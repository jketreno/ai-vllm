"""FastAPI wrapper around a torchao fp8-quantized Qwen-Image-Edit-2511 pipeline.

Loading is split into instrumented sections (transformer, text_encoder, pipeline
assembly). Before each section, free host memory is checked against that section's
expected requirement -- GB10 is a unified-memory system, so host RAM and GPU memory
are the same physical pool, and MemAvailable is a fair proxy for real headroom (which
has repeatedly measured lower than vLLM's nominal `1 - gpu-memory-utilization` figure
would suggest). If headroom looks insufficient, loading stops before the risky
allocation instead of attempting it and potentially OOMing the whole host. Progress
and measured memory deltas are persisted to disk after every section so a stalled or
aborted load leaves a clear record, and so future runs can use measured actuals
instead of guessed size requirements.
"""

import base64
import io
import json
import os
import threading
import time

import torch
from diffusers import AutoModel, QwenImageEditPlusPipeline, TorchAoConfig
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from prometheus_client import Histogram, start_http_server
from torchao.quantization import Float8WeightOnlyConfig
from transformers import Qwen2_5_VLForConditionalGeneration

MODEL_ID = os.environ.get("QWEN_IMAGE_EDIT_MODEL", "Qwen/Qwen-Image-Edit-2511")
METRICS_PORT = int(os.environ.get("QWEN_IMAGE_EDIT_METRICS_PORT", "9093"))
STATUS_PATH = os.environ.get("QWEN_IMAGE_EDIT_STATUS_PATH", "/app/state/load_status.json")
# Quantizing the transformer from bf16 to fp8 on every startup takes several minutes
# and briefly holds both the bf16 source and the fp8 result in memory. Cache the
# already-quantized weights on first run and load them directly on subsequent starts
# -- torchao persists real fp8 tensors here (not bf16 + a re-quantize instruction), so
# reload skips quantization compute entirely. Lives on the shared HF cache volume,
# alongside the other cached model repos, not the load-status state volume.
QUANTIZED_TRANSFORMER_PATH = os.environ.get(
    "QWEN_IMAGE_EDIT_QUANTIZED_TRANSFORMER_PATH",
    "/root/.cache/huggingface/qwen-image-edit-2511-transformer-fp8",
)

# Expected host MemAvailable required for each section, in GiB, before attempting it.
# Defaults are measured actuals from a successful concurrent run alongside vllm-engine
# on this GB10 node (see STATUS_PATH history / project plan), rounded up slightly for
# margin: transformer 12.81, text_encoder 13.38, pipeline 0.66 GiB actually used.
# Override via env if a future run's STATUS_PATH actuals differ meaningfully.
REQUIRED_GIB = {
    "transformer": float(os.environ.get("QWEN_IMAGE_EDIT_REQUIRED_TRANSFORMER_GIB", "14")),
    "text_encoder": float(os.environ.get("QWEN_IMAGE_EDIT_REQUIRED_TEXT_ENCODER_GIB", "15")),
    "pipeline": float(os.environ.get("QWEN_IMAGE_EDIT_REQUIRED_PIPELINE_GIB", "2")),
}
SAFETY_MARGIN_GIB = float(os.environ.get("QWEN_IMAGE_EDIT_SAFETY_MARGIN_GIB", "4"))

EDIT_LATENCY = Histogram(
    "qwen_image_edit_inference_seconds", "Time spent running one edit inference"
)

app = FastAPI(title="ai-vllm Qwen-Image-Edit API", version="1.0.0")
_pipeline = None
_pipeline_lock = threading.Lock()


def _mem_available_gib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


def _cuda_mem_gib():
    return {
        "allocated_gib": torch.cuda.memory_allocated() / (1024**3),
        "reserved_gib": torch.cuda.memory_reserved() / (1024**3),
    }


class LoadAborted(RuntimeError):
    pass


class _LoadStatus:
    """Persists load progress to disk after every section, and gates each section on
    measured free memory before attempting it, aborting cleanly if headroom is short."""

    def __init__(self, path):
        self.path = path
        self.sections = []
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._write({"state": "starting", "sections": []})

    def _write(self, extra):
        payload = {"sections": self.sections, "updated_at": time.time(), **extra}
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, self.path)

    def gate(self, section):
        required = REQUIRED_GIB[section]
        available = _mem_available_gib()
        threshold = required + SAFETY_MARGIN_GIB
        record = {
            "section": section,
            "required_gib": round(required, 2),
            "safety_margin_gib": SAFETY_MARGIN_GIB,
            "available_before_gib": round(available, 2),
        }
        if available < threshold:
            record["decision"] = "aborted_insufficient_memory"
            self.sections.append(record)
            self._write({"state": "aborted"})
            raise LoadAborted(
                f"Refusing to load '{section}': need ~{required:.1f} GiB + "
                f"{SAFETY_MARGIN_GIB:.1f} GiB safety margin = {threshold:.1f} GiB, "
                f"only {available:.1f} GiB host MemAvailable."
            )
        record["decision"] = "proceeding"
        self._pending = record
        self._write({"state": f"loading_{section}"})
        return record

    def note(self, key, value):
        self._pending[key] = value

    def record_done(self, section, extra_timing_s):
        available_after = _mem_available_gib()
        record = self._pending
        record["available_after_gib"] = round(available_after, 2)
        record["actual_used_gib"] = round(
            record["available_before_gib"] - available_after, 2
        )
        record["duration_s"] = round(extra_timing_s, 1)
        record.update(_cuda_mem_gib())
        self.sections.append(record)
        self._write({"state": f"loaded_{section}"})


def _load_transformer_cached(status):
    marker = os.path.join(QUANTIZED_TRANSFORMER_PATH, "config.json")
    t0 = time.time()
    if os.path.exists(marker):
        transformer = AutoModel.from_pretrained(
            QUANTIZED_TRANSFORMER_PATH,
            torch_dtype=torch.bfloat16, use_safetensors=False, device_map="cuda",
        )
        status.note("transformer_source", "cache")
    else:
        quant_config = TorchAoConfig(quant_type=Float8WeightOnlyConfig())
        transformer = AutoModel.from_pretrained(
            MODEL_ID, subfolder="transformer",
            quantization_config=quant_config, torch_dtype=torch.bfloat16,
            device_map="cuda",
        )
        status.note("transformer_source", "quantized_fresh")
        # torchao's Float8WeightOnlyConfig tensor-subclass format isn't safetensors-
        # compatible yet, hence safe_serialization=False -- persists actual fp8
        # tensors so the next startup can load them directly (see comment above).
        transformer.save_pretrained(QUANTIZED_TRANSFORMER_PATH, safe_serialization=False)
    return transformer, time.time() - t0


def _load_pipeline():
    status = _LoadStatus(STATUS_PATH)

    status.gate("transformer")
    transformer, elapsed = _load_transformer_cached(status)
    status.record_done("transformer", elapsed)

    status.gate("text_encoder")
    t0 = time.time()
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, subfolder="text_encoder",
        torch_dtype=torch.bfloat16, device_map="cuda",
    )
    status.record_done("text_encoder", time.time() - t0)

    status.gate("pipeline")
    t0 = time.time()
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID, transformer=transformer, text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
    )
    pipeline.to("cuda")  # only moves the small remaining components (vae, tokenizer)
    status.record_done("pipeline", time.time() - t0)

    status._write({"state": "ready"})
    return pipeline


@app.on_event("startup")
def startup():
    global _pipeline
    start_http_server(METRICS_PORT)
    try:
        _pipeline = _load_pipeline()
    except LoadAborted as error:
        # Leave _pipeline as None; /health reports not-ready and /v1/edit returns 503.
        # The process stays up so the status file and logs remain inspectable rather
        # than the container silently restart-looping.
        print(f"qwen-image-edit: {error}", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipeline is not None}


@app.get("/v1/load-status")
def load_status():
    try:
        with open(STATUS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(404, "No load status recorded yet")


@app.post("/v1/edit")
async def edit(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    num_inference_steps: int = Form(20),
    true_cfg_scale: float = Form(4.0),
    seed: int = Form(0),
):
    if _pipeline is None:
        raise HTTPException(503, "Model not loaded (see /v1/load-status)")
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(400, "Invalid image") from error
    if not prompt.strip():
        raise HTTPException(400, "prompt must not be empty")
    num_inference_steps = max(1, min(num_inference_steps, 100))

    generator = torch.Generator(device="cuda").manual_seed(seed)
    with _pipeline_lock, torch.inference_mode(), EDIT_LATENCY.time():
        result = _pipeline(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            generator=generator,
        ).images[0]

    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    return {
        "width": result.width,
        "height": result.height,
        "image_png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }

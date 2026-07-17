"""FastAPI wrapper around a torchao fp8-quantized Qwen-Image-Edit-2511 pipeline."""

import base64
import io
import os
import threading

import torch
from diffusers import AutoModel, QwenImageEditPlusPipeline, TorchAoConfig
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from prometheus_client import Histogram, start_http_server
from torchao.quantization import Float8WeightOnlyConfig
from transformers import Qwen2_5_VLForConditionalGeneration

MODEL_ID = os.environ.get("QWEN_IMAGE_EDIT_MODEL", "Qwen/Qwen-Image-Edit-2511")
METRICS_PORT = int(os.environ.get("QWEN_IMAGE_EDIT_METRICS_PORT", "9093"))

EDIT_LATENCY = Histogram(
    "qwen_image_edit_inference_seconds", "Time spent running one edit inference"
)

app = FastAPI(title="ai-vllm Qwen-Image-Edit API", version="1.0.0")
_pipeline = None
_pipeline_lock = threading.Lock()


def _load_pipeline():
    # Load the transformer and text_encoder straight to GPU via device_map instead of
    # loading to CPU then calling pipeline.to("cuda"): a post-hoc .to() briefly holds
    # both the CPU and GPU copy of each parameter during the transfer, which spiked
    # peak memory enough to OOM the ~16.6 GB bf16 text_encoder when GPU headroom was
    # tight (vllm-engine running concurrently). device_map="cuda" streams weights
    # directly to the target device as they're read, with no CPU-resident stage.
    quant_config = TorchAoConfig(quant_type=Float8WeightOnlyConfig())
    transformer = AutoModel.from_pretrained(
        MODEL_ID, subfolder="transformer",
        quantization_config=quant_config, torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, subfolder="text_encoder",
        torch_dtype=torch.bfloat16, device_map="cuda",
    )
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID, transformer=transformer, text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
    )
    pipeline.to("cuda")  # only moves the small remaining components (vae, tokenizer)
    return pipeline


@app.on_event("startup")
def startup():
    global _pipeline
    start_http_server(METRICS_PORT)
    _pipeline = _load_pipeline()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipeline is not None}


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
        raise HTTPException(503, "Model not loaded yet")
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

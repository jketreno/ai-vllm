"""Qwen3.5 MoE QLoRA trainer producing immutable CLARE adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import random
import time
from datetime import datetime, timezone
from typing import Any

from unsloth import FastModel
from datasets import Dataset
from peft import PeftConfig
import torch
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from mlflow_tracking import TrainingTracker

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
REQUESTED_LOAD_IN_4BIT = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--base_model_id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--adapter_id", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--project_id", default="global")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dependency_lock", default="/app/requirements.lock")
    return parser.parse_args()


def file_hash(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def memory_snapshot(stage: str) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"stage": stage, "timestamp": datetime.now(tz=timezone.utc).isoformat()}
    try:
        meminfo = pathlib.Path("/proc/meminfo").read_text(encoding="utf-8")
        for line in meminfo.splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                snapshot[key] = value.strip()
    except OSError:
        pass
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        snapshot["cuda_free_bytes"] = free
        snapshot["cuda_total_bytes"] = total
        snapshot["cuda_allocated_bytes"] = torch.cuda.memory_allocated()
        snapshot["cuda_reserved_bytes"] = torch.cuda.memory_reserved()
    return snapshot


def effective_training_mode(model: Any, base_model_id: str) -> str:
    if bool(getattr(model, "is_loaded_in_4bit", False)):
        return "qlora-4bit"
    if "fp8" in base_model_id.casefold():
        return "fp8-16bit-lora"
    return "other"


def inference_base_from_env(
    *,
    train_base: dict[str, Any],
) -> dict[str, Any]:
    inference_model = os.environ.get("CLARE2_INFERENCE_MODEL", train_base["model_id"])
    inference_revision = os.environ.get("CLARE2_INFERENCE_REVISION", train_base["revision"])
    return {
        "model_id": inference_model,
        "revision": inference_revision,
        "architecture": os.environ.get(
            "CLARE2_BASE_ARCHITECTURE",
            train_base.get("architecture", "unknown"),
        ),
        "config_hash": os.environ.get("CLARE2_BASE_CONFIG_HASH", train_base["config_hash"]),
        "tokenizer_hash": os.environ.get("CLARE2_TOKENIZER_HASH", train_base["tokenizer_hash"]),
        "inference_quantization": os.environ.get("CLARE2_INFERENCE_QUANTIZATION", "fp8"),
    }


def _process_line(
    line: str,
    tokenizer: Any,
    max_seq_length: int,
    skipped: dict[str, int],
) -> dict[str, Any] | None:
    # Keep prompt/completion as separate columns: SFTTrainer only masks the
    # prompt out of the loss when it sees this shape, not a flattened "text" field.
    if not line.strip():
        skipped["blank"] = skipped.get("blank", 0) + 1
        return None
    try:
        source = json.loads(line)
    except json.JSONDecodeError:
        skipped["malformed_json"] = skipped.get("malformed_json", 0) + 1
        return None
    prompt = source.get("prompt")
    completion = source.get("completion")
    if not isinstance(prompt, str) or not prompt.strip():
        skipped["missing_prompt"] = skipped.get("missing_prompt", 0) + 1
        return None
    if not isinstance(completion, str) or not completion.strip():
        skipped["missing_completion"] = skipped.get("missing_completion", 0) + 1
        return None
    prompt_messages = [{"role": "user", "content": prompt}]
    completion_messages = [{"role": "assistant", "content": completion}]
    full_text = tokenizer.apply_chat_template(
        prompt_messages + completion_messages, tokenize=False, add_generation_prompt=False
    )
    tokens = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if not tokens:
        skipped["empty_tokens"] = skipped.get("empty_tokens", 0) + 1
        return None
    if len(tokens) > max_seq_length:
        skipped["over_length"] = skipped.get("over_length", 0) + 1
        return None
    return {
        "prompt": prompt_messages,
        "completion": completion_messages,
        "category": source.get("category", "general"),
    }


def load_corpus(
    train_file: pathlib.Path,
    tokenizer: Any,
    max_seq_length: int,
) -> tuple[Dataset, dict[str, int]]:
    records: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    with train_file.open(encoding="utf-8") as handle:
        for line in handle:
            record = _process_line(line, tokenizer, max_seq_length, skipped)
            if record is not None:
                records.append(record)
    if not records:
        raise ValueError(f"no valid training records; skipped={skipped}")
    return Dataset.from_list(records), skipped


class FiniteLossCallback(TrainerCallback):
    def __init__(self, tracker: TrainingTracker | None = None) -> None:
        self.loss_history: list[dict[str, float]] = []
        self.tracker = tracker

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss = float(logs["loss"])
            if not math.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss: {loss}")
            self.loss_history.append({"step": float(state.global_step), "loss": loss})
            if self.tracker is not None:
                self.tracker.log_metric("train.loss", loss, step=state.global_step)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    train_file = pathlib.Path(args.train_file)
    started = time.monotonic()
    memory_snapshots = [memory_snapshot("before_model_load")]
    tracker = TrainingTracker(
        lifecycle_run_id=args.run_id,
        adapter_id=args.adapter_id,
        project_id=args.project_id,
    )
    corpus_hash = file_hash(train_file)
    lock_path = pathlib.Path(args.dependency_lock)
    mlflow_run_id = tracker.start(
        {
            "base_model": args.model_name,
            "base_model_id": args.base_model_id,
            "base_revision": args.revision,
            "corpus_hash": corpus_hash,
            "dependency_lock_hash": file_hash(lock_path),
            "seed": args.seed,
            "lora_rank": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": TARGET_MODULES,
            "max_seq_length": args.max_seq_length,
            "epochs": args.num_train_epochs,
            "batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "bf16": True,
            "requested_load_in_4bit": REQUESTED_LOAD_IN_4BIT,
        },
        {
            "clare2.stage": "training",
            "clare2.framework": "unsloth",
            "clare2.quantization.requested": "qlora-4bit",
        },
    )

    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=args.model_name,
            revision=args.revision,
            max_seq_length=args.max_seq_length,
            load_in_4bit=REQUESTED_LOAD_IN_4BIT,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        memory_snapshots.append(memory_snapshot("after_model_load"))
        training_mode = effective_training_mode(model, args.base_model_id)
        text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        config_hash = text_hash(model.config.to_json_string())
        tokenizer_hash = text_hash(json.dumps(text_tokenizer.init_kwargs, sort_keys=True, default=str))
        architecture = getattr(model.config, "architectures", None) or [model.config.__class__.__name__]
        train_base = {
            "model_id": args.base_model_id,
            "revision": args.revision,
            "architecture": architecture[0],
            "config_hash": config_hash,
            "tokenizer_hash": tokenizer_hash,
        }
        inference_base = inference_base_from_env(train_base=train_base)
        tracker.log_params(
            {
                "effective_training_mode": training_mode,
                "base_config_hash": config_hash,
                "tokenizer_hash": tokenizer_hash,
                "inference_model_id": inference_base["model_id"],
                "inference_revision": inference_base["revision"],
            }
        )
        model = FastModel.get_peft_model(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=TARGET_MODULES,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )
        dataset, skipped = load_corpus(train_file, text_tokenizer, args.max_seq_length)
        tracker.log_metric("corpus.training_records", float(len(dataset)))
        for reason, count in skipped.items():
            tracker.log_metric(f"corpus.skipped.{reason}", float(count))
        tracker.log_dict(skipped, "corpus/skipped_records.json")
        callback = FiniteLossCallback(tracker)
        training_args = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=1,
            save_strategy="epoch",
            bf16=True,
            max_length=args.max_seq_length,
            completion_only_loss=True,
            report_to="none",
            seed=args.seed,
        )
        trainer = SFTTrainer(
            model=model,
            processing_class=text_tokenizer,
            train_dataset=dataset,
            args=training_args,
            callbacks=[callback],
        )
        result = trainer.train()
        if not math.isfinite(float(result.training_loss)):
            raise FloatingPointError("final training loss is not finite")
        memory_snapshots.append(memory_snapshot("after_training"))

        model.save_pretrained(str(output_dir), safe_serialization=True)
        text_tokenizer.save_pretrained(str(output_dir))
        PeftConfig.from_pretrained(str(output_dir))

        metadata = {
            "adapter_id": args.adapter_id,
            "run_id": args.run_id,
            "mlflow_run_id": mlflow_run_id,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "corpus_hash": corpus_hash,
            "base": train_base,
            "train_base": train_base,
            "inference_base": inference_base,
            "effective_training_mode": training_mode,
            "memory_snapshots": memory_snapshots,
            "dependency_lock_hash": file_hash(lock_path),
            "seed": args.seed,
            "hyperparameters": {
                "rank": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "max_seq_length": args.max_seq_length,
                "epochs": args.num_train_epochs,
                "learning_rate": args.learning_rate,
                "bf16": True,
                "requested_load_in_4bit": REQUESTED_LOAD_IN_4BIT,
            },
            "target_modules": TARGET_MODULES,
            "training_records": len(dataset),
            "skipped_records": skipped,
            "loss_history": callback.loss_history,
            "final_loss": float(result.training_loss),
            "duration_seconds": round(time.monotonic() - started, 3),
        }
        (output_dir / "training_meta.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        candidate = {
            "id": args.adapter_id,
            "directory": args.adapter_id,
            "created_at": metadata["created_at"],
            "corpus_hash": corpus_hash,
            "base": metadata["base"],
            "train_base": metadata["train_base"],
            "inference_base": metadata["inference_base"],
            "effective_training_mode": training_mode,
            "peft": {
                "rank": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
            },
            "target_modules": TARGET_MODULES,
            "evaluation": None,
            "project_scope": args.project_id,
            "capabilities": ["code", "review"],
            "status": "candidate",
            "mlflow_run_id": mlflow_run_id,
        }
        (output_dir / "candidate_manifest.json").write_text(
            json.dumps(candidate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tracker.log_metric("train.final_loss", float(result.training_loss))
        tracker.log_metric("train.duration_seconds", metadata["duration_seconds"])
        tracker.log_dict(metadata, "training/training_meta.json")
        tracker.log_adapter_artifacts(output_dir)
        tracker.finish()
    except Exception:
        tracker.finish("FAILED")
        raise


if __name__ == "__main__":
    main()

"""Qwen3.5 MoE QLoRA trainer producing immutable CLARE adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import random
import time
from datetime import datetime, timezone
from typing import Any

from datasets import Dataset
from peft import PeftConfig
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastModel

from mlflow_tracking import TrainingTracker

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
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


def _tokenize_pair(
    tokenizer: Any,
    prompt: str,
    completion: str,
) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )


def _process_line(
    line: str,
    tokenizer: Any,
    max_seq_length: int,
    skipped: dict[str, int],
) -> dict[str, Any] | None:
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
    text = _tokenize_pair(tokenizer, prompt, completion)
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not tokens:
        skipped["empty_tokens"] = skipped.get("empty_tokens", 0) + 1
        return None
    if len(tokens) > max_seq_length:
        skipped["over_length"] = skipped.get("over_length", 0) + 1
        return None
    return {"text": text, "category": source.get("category", "general")}


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
            "load_in_4bit": True,
        },
        {
            "clare2.stage": "training",
            "clare2.framework": "unsloth",
            "clare2.quantization": "qlora-4bit",
        },
    )

    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=args.model_name,
            revision=args.revision,
            max_seq_length=args.max_seq_length,
            load_in_4bit=True,
            dtype="bfloat16",
            trust_remote_code=True,
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
        dataset, skipped = load_corpus(train_file, tokenizer, args.max_seq_length)
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
            dataset_text_field="text",
            report_to="none",
            seed=args.seed,
        )
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=training_args,
            callbacks=[callback],
        )
        result = trainer.train()
        if not math.isfinite(float(result.training_loss)):
            raise FloatingPointError("final training loss is not finite")

        model.save_pretrained(str(output_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(output_dir))
        PeftConfig.from_pretrained(str(output_dir))

        config_hash = text_hash(model.config.to_json_string())
        tokenizer_hash = text_hash(json.dumps(tokenizer.init_kwargs, sort_keys=True, default=str))
        tracker.log_params(
            {
                "base_config_hash": config_hash,
                "tokenizer_hash": tokenizer_hash,
            }
        )
        metadata = {
            "adapter_id": args.adapter_id,
            "run_id": args.run_id,
            "mlflow_run_id": mlflow_run_id,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "corpus_hash": corpus_hash,
            "base": {
                "model_id": args.model_name,
                "revision": args.revision,
                "config_hash": config_hash,
                "tokenizer_hash": tokenizer_hash,
            },
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
                "load_in_4bit": True,
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

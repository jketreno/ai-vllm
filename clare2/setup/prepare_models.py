"""Resolve, download, and fingerprint CLARE₂ Qwen model snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib

from huggingface_hub import HfApi, snapshot_download
from transformers import AutoConfig, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-model", required=True)
    parser.add_argument("--training-model", required=True)
    parser.add_argument("--inference-revision")
    parser.add_argument("--training-revision")
    parser.add_argument("--cache-dir", default="/cache")
    parser.add_argument("--token-file", default="/run/secrets/huggingface_token")
    parser.add_argument("--output", default="/output/model.env")
    return parser.parse_args()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def revision(api: HfApi, model: str, configured: str | None, token: str) -> str:
    requested = (configured or "").strip()
    if requested.startswith("REPLACE_WITH_"):
        requested = ""
    return api.model_info(model, revision=requested or None, token=token).sha


def main() -> None:
    args = parse_args()
    token = pathlib.Path(args.token_file).read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("Hugging Face token is empty")

    api = HfApi()
    inference_revision = revision(
        api,
        args.inference_model,
        args.inference_revision,
        token,
    )
    training_revision = revision(
        api,
        args.training_model,
        args.training_revision,
        token,
    )

    print(f"Downloading {args.inference_model}@{inference_revision}", flush=True)
    snapshot_download(
        repo_id=args.inference_model,
        revision=inference_revision,
        cache_dir=args.cache_dir,
        token=token,
    )
    print(f"Downloading {args.training_model}@{training_revision}", flush=True)
    training_path = snapshot_download(
        repo_id=args.training_model,
        revision=training_revision,
        cache_dir=args.cache_dir,
        token=token,
    )

    config = AutoConfig.from_pretrained(
        training_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        training_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    values = {
        "CLARE2_INFERENCE_REVISION": inference_revision,
        "CLARE2_TRAIN_REVISION": training_revision,
        "CLARE2_BASE_CONFIG_HASH": digest(config.to_json_string()),
        "CLARE2_TOKENIZER_HASH": digest(
            json.dumps(tokenizer.init_kwargs, sort_keys=True, default=str)
        ),
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()),
        encoding="utf-8",
    )
    os.chmod(output, 0o600)


if __name__ == "__main__":
    main()

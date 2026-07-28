# Phase 06 — Supported QLoRA Base and Training Canary

Status: proposed  
Depends on: Phase 05  
Produces: a proven training configuration and fail-fast canary

## Goal

Replace unsupported FP8 training with a pinned, compatible non-FP8 training
base and prove that the adapter learns before committing to a full run.

## Required Human Decision

Select and pin the exact non-FP8 Qwen3.6-27B artifact corresponding to the FP8
inference model. Confirm its license, architecture, tokenizer, and weight
lineage.

If no compatible artifact exists, stop this phase and select a different
training/inference pair. Do not bypass fingerprint checks.

## Training Mode Contract

The trainer must assert:

- requested mode is `qlora-4bit`;
- effective mode is `qlora-4bit`;
- quantization config is supported;
- trainable parameter count and names match policy;
- all expected target modules exist;
- base/tokenizer fingerprints match the approved compatibility manifest; and
- no unexpected quantization-scale keys were discarded.

Silent fallback to FP8 16-bit LoRA is a hard failure.

## Canary Stages

1. Load tokenizer and validate chat template/token IDs.
2. Load base and run known-text base loss/perplexity sanity.
3. Build adapters and enumerate trainable parameters.
4. Run one forward/backward step on a fixed mini-corpus.
5. Assert finite loss and gradients.
6. Assert nonzero finite gradient norms in expected adapter modules.
7. Run an optimizer update.
8. Assert adapter parameters changed while base parameters did not.
9. Repeat the same mini-batch and require loss improvement within a reviewed
   tolerance.
10. Save, reload, and validate the adapter.
11. Load the adapter over the FP8 inference runtime and compare deterministic
    probe output.

The checkpoint mismatch must be solved by using a supported stack or targeted
architecture-aware checkpointing. Do not disable safety checks or set
checkpoint determinism validation to `none` merely to suppress the exception.

## Resource Guardrails

- Configurable micro-batch and accumulation.
- Pre-load and peak memory thresholds.
- Host available-memory floor.
- Timeout per canary stage.
- No full training after canary failure.
- Clean trainer exit and vLLM recovery.

## Primary File Targets

- `clare2/train/train.py`
- `clare2/train/requirements.lock`
- `clare2/train/Dockerfile`
- `clare2/train/train.sh`
- model setup and compatibility manifests
- MLflow tracking and training tests
- Compose and environment documentation

## Tests

CPU/unit tests:

- effective-mode mismatch fails;
- missing target modules fail;
- frozen/trainable parameter invariants;
- canary result schema;
- loss/gradient/parameter-change guards;
- compatibility-manifest validation.

GPU acceptance:

- supported small-model fixture;
- selected 27B training base;
- three clean-container repetitions;
- adapter save/reload;
- FP8 inference load compatibility;
- bounded memory and clean recovery.

## Production Validation

Run the canary manually with nightly training disabled. Record:

- image and dependency hashes;
- model fingerprints;
- effective quantization;
- memory snapshots;
- base loss;
- pre/post-update loss;
- gradient norms;
- changed parameter hashes;
- duration; and
- FP8 inference compatibility result.

## Rollback

Remove the candidate training image from selection and retain the previous
base-only inference configuration. Do not fall back to FP8 LoRA training.

## Exit Criteria

- Three consecutive clean canaries pass.
- QLoRA remains enabled as requested.
- Loss and adapter parameters demonstrably change.
- The adapter loads over the intended FP8 inference base.
- Memory guardrails work.
- `verify-ci.sh` passes.


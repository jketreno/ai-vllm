# Phase 00 — Containment and Reproducible Baseline

Status: proposed  
Depends on: none  
Produces: safe training admission control, immutable audit baseline, and
pre-change production evidence

## Goal

Stop known-bad FP8 training from consuming the nightly window while preserving
capture and distillation. Establish a repeatable baseline against which every
later phase can be measured.

## Scope

1. Add an explicit training-enabled admission setting, defaulting to disabled
   until Phase 09.
2. Distinguish `disabled_by_operator`, `skipped_no_eligible_corpus`, and
   `failed` lifecycle outcomes.
3. Preserve capture, sync, distillation, summarization, assembly, and base-model
   inference while training is disabled.
4. Add a read-only audit command/report for corpus, summaries, MLflow, registry,
   lifecycle, and trainer compatibility.
5. Persist full exception type, project, adapter ID when known, MLflow run ID
   when known, and traceback artifact for training failures.
6. Record the current production hashes and observed failure history without
   mutating historical artifacts.

## Primary File Targets

- `clare2/pipeline/app/main.py`
- `clare2/pipeline/app/lifecycle.py`
- `clare2/pipeline/app/notify.py`
- `clare2/pipeline/app/metrics.py`
- `clare2/train/train.sh`
- `clare2/train/train.py`
- `docker-compose.yml`
- `.env.example`
- `README.md`
- focused tests under `clare2/pipeline/tests` and `clare2/train/tests`

## Design

Introduce `CLARE2_TRAINING_ENABLED`, parsed strictly as a boolean. The scheduler
may still invoke the lifecycle entry point, but admission returns a durable,
observable skip outcome before maintenance mode, vLLM drain, or trainer start.

The baseline report should include:

- repository and container image hashes;
- dependency lock hash;
- base and tokenizer fingerprints;
- requested and effective training modes;
- project corpus hashes and record counts;
- summary and theme counts;
- MLflow status totals;
- registry status totals and aliases;
- stale candidates/runs;
- last trainer traceback fingerprint; and
- whether production meets training admission prerequisites.

The report must redact secrets and avoid copying raw corpus text by default.

## Tests

- Disabled admission never enters maintenance or stops vLLM.
- Capture/distillation schedules remain registered.
- Invalid boolean configuration fails safely.
- Disabled runs are not reported as training failures.
- Failure context preserves known adapter and MLflow IDs.
- Audit output is deterministic for identical fixtures.
- Audit output does not expose configured secrets.

## Production Validation

1. Deploy with training disabled.
2. Confirm the next scheduled admission reports `disabled_by_operator`.
3. Confirm vLLM remains available through the training window.
4. Confirm capture, sync, and distillation continue.
5. Generate and archive the baseline report.

## Rollback

Revert the admission-control deployment. Do not enable the old trainer as part
of rollback; base inference remains the safe operating state.

## Exit Criteria

- Known-bad nightly GPU training cannot start accidentally.
- The production baseline is reproducible and archived.
- Failure notifications identify the actual project and retained MLflow run.
- Existing corpus and adapters are unchanged.
- `verify-ci.sh` passes.

## Not in Scope

- Fixing model training.
- Changing corpus meaning.
- Migrating existing records.
- Promoting any adapter.


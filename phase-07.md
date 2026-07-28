# Phase 07 — Project-Isolated Training Orchestration

Status: proposed  
Depends on: Phase 06  
Produces: independent project outcomes and complete run observability

## Goal

Ensure one project's crash, invalid corpus, or rejection cannot prevent other
eligible projects from completing.

## Batch Model

A nightly batch owns multiple independent project runs:

```text
batch
  Zoo-Code -> skipped/failed/trained
  ai-vllm  -> skipped/failed/trained
  auto-sam -> skipped/failed/trained
```

Each project has its own:

- run ID;
- adapter ID;
- corpus hash;
- MLflow run;
- lifecycle phase;
- timeout;
- result;
- traceback/error fingerprint; and
- evaluation handoff.

The batch outcome summarizes projects but does not overwrite their outcomes.

## Scheduling

- Run the Phase 06 canary once per image/base fingerprint before the batch.
- Admit only projects whose Phase 05 manifests pass.
- Continue after a project failure when hardware remains healthy.
- Stop remaining projects only for batch-global faults such as failed canary,
  unsafe memory state, vLLM recovery failure, or host health failure.
- Make ordering explicit and configurable; never depend on ASCII path order.

## State and Recovery

- Persist batch and project state atomically.
- Callbacks are idempotent per project run.
- Watchdog reconciliation distinguishes container exit, host restart, callback
  loss, and project failure.
- Restart can resume eligible unfinished projects without retraining completed
  ones.
- Failed partial adapter directories are marked and quarantined.
- Stale MLflow `RUNNING` entries are reconciled with an explicit terminal
  reason.

## Observability

Persist:

- full project result;
- effective training mode;
- canary result;
- skipped-record reasons;
- loss/gradient/memory history;
- traceback artifact;
- timestamps for every phase;
- callback delivery state; and
- batch-global recovery actions.

Emails and metrics report every project, including those after a failure.

## Primary File Targets

- `clare2/train/train.sh`
- `clare2/pipeline/app/lifecycle.py`
- `clare2/pipeline/app/main.py`
- `clare2/pipeline/app/notify.py`
- `clare2/pipeline/app/metrics.py`
- lifecycle state schemas and tests

## Tests

- First project fails; second trains; third skips.
- Global canary failure prevents all projects.
- Project callback retries are idempotent.
- Restart resumes only unfinished projects.
- Same corpus can retry after failed training but not after durable success.
- A stale trainer container is reconciled.
- VLLM returns after every terminal path.
- Notifications contain all project outcomes.
- Project ordering is explicit and deterministic.

## Production Validation

With fixture trainers:

1. Force Zoo-Code failure.
2. Confirm ai-vllm and another project still run.
3. Interrupt the batch between projects and resume.
4. Drop a callback and verify watchdog reconciliation.
5. Confirm base inference restoration.

Then perform one canary-approved real batch without enabling the recurring
schedule.

## Rollback

Disable batch training admission. Restore base inference. Preserve all batch and
project state for diagnosis.

## Exit Criteria

- One project cannot block another.
- Every project has a durable terminal outcome.
- Interrupted batches resume safely.
- Recovery restores inference.
- `verify-ci.sh` passes.


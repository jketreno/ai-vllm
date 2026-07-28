# Phase 09 — Migration, Shadow Rollout, and Production Enablement

Status: proposed  
Depends on: Phases 00–08  
Produces: migrated v2 corpus, demonstrated promotion, and supported operations

## Goal

Move from the current v1 corpus and disabled trainer to a monitored v2
production lifecycle without losing evidence or reintroducing known failure
modes.

## Migration Strategy

### Preserve v1

- Freeze v1 corpus, summaries, registry, and relevant MLflow metadata as
  immutable audit inputs.
- Hash and inventory every artifact.
- Do not relabel v1 narratives as verified corrections.

### Build v2

- Reprocess eligible raw sessions through v2 validation and distillation.
- Mark unverifiable source relationships explicitly.
- Apply disposition, expiry, deduplication, and sensitivity policy.
- Generate new summaries/themes and corpus manifests under a versioned root.
- Quarantine invalid or sensitive records with reason codes.

### Compare

For each project, report:

- v1 versus v2 example counts;
- verified correction share;
- disposition totals;
- expired/sensitive exclusions;
- duplicate reduction;
- held-out size;
- anchor ratio;
- estimated updates; and
- admission result.

## Shadow Rollout Stages

1. **Capture shadow:** v2 capture beside v1.
2. **Distillation shadow:** no v2 output used by training.
3. **Corpus shadow:** v2 manifests reviewed.
4. **Canary only:** manual GPU canary; no full candidate.
5. **Manual candidate:** one selected project, no automatic promotion.
6. **Evaluation shadow:** compare decision with human review.
7. **Manual promotion:** route limited project traffic.
8. **Soak:** monitor quality, latency, memory, and fallback.
9. **Rollback drill:** restore base/prior adapter.
10. **Scheduled enablement:** explicitly set training enabled.

Advancing stages requires an archived approval record.

## Initial Project Selection

Choose a project with:

- enough verified corrections;
- stable source-of-truth rules;
- meaningful held-out tests;
- low sensitivity;
- active development; and
- an owner available to review outputs.

Do not select a project solely because it sorts first or has the largest
narrative corpus.

## Operational Runbook

Document:

- admission status;
- canary triage;
- corpus quarantine review;
- summary partition retries;
- project training retries;
- MLflow stale-run reconciliation;
- evaluation inspection;
- promotion and rollback;
- base-model fallback;
- retention/tombstone processing;
- alert meanings; and
- incident evidence collection.

## Monitoring and Alerts

At minimum:

- capture and distillation lag;
- invalid/quarantined record rate;
- correction-backed corpus share;
- summary partition failures;
- admitted/skipped project count and reasons;
- canary success and effective training mode;
- loss, gradients, memory, and duration;
- project training outcomes;
- evaluation deltas and regression blocks;
- adapter load/routing failures;
- stale lifecycle/MLflow state; and
- rollback events.

## Tests and Drills

- Full clean-room fixture pipeline from capture to promotion.
- Production-like shadow replay.
- Host/container restart during every major lifecycle stage.
- Failed project followed by successful project.
- Invalid summary output with last-known-good preservation.
- Sensitive record tombstone propagation.
- Adapter load failure and base fallback.
- Promotion followed by rollback.
- Disaster recovery from manifests and immutable adapters.

## Enablement Checklist

- All previous phase exit criteria are signed off.
- Non-FP8 training artifact and compatibility manifest are approved.
- Anchor dataset license is approved.
- Corpus privacy policy is approved.
- At least three production-host canaries pass.
- Selected project corpus passes admission.
- Human review approves held-out evaluation quality.
- Manual candidate passes evaluation.
- Routing isolation and rollback are demonstrated.
- Alerts and on-call runbook are active.
- `CLARE2_TRAINING_ENABLED` is changed only in the explicit enablement change.

## Rollback

1. Disable training admission.
2. Route affected projects to the last approved adapter or base.
3. Preserve the failed adapter, corpus manifest, evaluation, and lifecycle
   state.
4. Revert active v2 pointers if the defect is data-related.
5. Keep v2 capture running unless privacy or schema integrity is implicated.

## Exit Criteria

- One v2 adapter is trained, evaluated, and promoted through the normal
  lifecycle.
- Routing selects it only for its intended project/capabilities.
- General and project held-out checks pass.
- Rollback succeeds.
- Another project's simulated failure does not affect it.
- Scheduled training is enabled only after explicit human approval.
- `verify-ci.sh` passes.


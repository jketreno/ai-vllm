# CLARE₂ Learning-Plane Recovery Plan

Status: proposed  
Scope: CLARE₂ capture, distillation, corpus construction, training, evaluation,
promotion, and operations  
Implementation model: one phase at a time, with a human-reviewed gate between
phases

## Objective

Turn CLARE₂ from a pipeline that trains on synthetic summaries through an
unsupported FP8 LoRA path into a reproducible learning system that:

1. learns from verified developer corrections;
2. separates durable behavior from volatile context and noise;
3. trains a compatible QLoRA adapter with measurable gradient and loss sanity;
4. evaluates held-out project behavior and general coding retention;
5. isolates failures by project; and
6. retains enough evidence to explain every training and promotion decision.

The control-plane strengths already present—immutable adapter IDs, pinned base
revisions, lifecycle recovery, MLflow tracking, safe routing defaults, and
project partitioning—should be preserved.

## Confirmed Production Findings Driving This Plan

- Qwen3.6-27B-FP8 is not being trained with QLoRA. Unsloth disables
  `load_in_4bit` and switches to 16-bit LoRA over the FP8 checkpoint.
- Recent runs fail on the first backward pass because Qwen3.6 checkpoint
  recomputation saves a different tensor structure from the forward pass.
- Completed FP8 LoRA runs held loss near 14.2–14.5; earlier supported
  Qwen3.5-4B QLoRA runs produced loss near 2.5.
- The corpus contains distilled behavioral descriptions and templated prose,
  not failed-code to verified-fix pairs.
- Production Zoo-Code and ai-vllm captures contain no explicit correction
  records.
- The recurrence gate trusts the distillation model's `evidence_count`.
- Current category weights are written to JSONL but ignored by the trainer.
- There is no anchor dataset, held-out split, or minimum optimizer-update gate.
- Weekly summarization can replace substantial input with an empty result after
  invalid structured output.
- No monthly summaries, quarterly summaries, or active themes currently exist.
- One project's training failure exits the complete multi-project batch.
- Generic keyword probes are unrelated to the trained project corpus.
- Rejected adapters do not retain their evaluation report.

## Target Architecture

```text
captured source events
  -> schema validation + stable event IDs
  -> verified correction/CI linkage
  -> deterministic evidence accounting
  -> semantic distillation with provenance
  -> stable/context/discard classification
  -> chunked summaries + durable themes
  -> deduplicated project examples
       + licensed general-coding anchors
       + held-out project evaluation set
  -> supported non-FP8-base QLoRA canary
  -> project-isolated training jobs
  -> project-specific + general regression evaluation
  -> persist report
  -> promote or reject
  -> route approved immutable adapter
```

## Non-Negotiable Invariants

1. A production training run must not begin unless its canary passes.
2. The trainer must fail if requested QLoRA silently becomes another mode.
3. Every learned project example must retain source event provenance.
4. `evidence_count` must be derived in code, never accepted from an LLM.
5. A correction may be high confidence without being mislabeled as recurrent.
6. Volatile or sensitive operational facts must not enter model weights.
7. Training and evaluation examples must be disjoint by semantic lineage.
8. General-capability anchors must have documented source, license, and hash.
9. A project failure must not prevent other eligible projects from completing.
10. Promotion and rejection must both persist the complete evaluation report.
11. Empty or invalid summarization output must never replace a last-known-good
    artifact.
12. The base model remains the routing fallback until an adapter is explicitly
    approved.

## Phase Sequence

| Phase | Document | Outcome | Depends on |
|---|---|---|---|
| 00 | [phase-00.md](phase-00.md) | Contain unsafe training and establish a reproducible baseline | none |
| 01 | [phase-01.md](phase-01.md) | Versioned capture and verified-correction data contract | 00 |
| 02 | [phase-02.md](phase-02.md) | Deterministic recurrence, provenance, and distillation quality gates | 01 |
| 03 | [phase-03.md](phase-03.md) | Stable/context/discard routing with expiry and sensitivity controls | 02 |
| 04 | [phase-04.md](phase-04.md) | Chunked, failure-safe summaries and durable themes | 02–03 |
| 05 | [phase-05.md](phase-05.md) | Deduplicated corpus, anchor blend, weights, splits, and admission rules | 03–04 |
| 06 | [phase-06.md](phase-06.md) | Supported QLoRA base, training canary, and gradient/loss validation | 05 |
| 07 | [phase-07.md](phase-07.md) | Project-isolated training orchestration and complete observability | 06 |
| 08 | [phase-08.md](phase-08.md) | Held-out project evaluation and explainable promotion | 05–07 |
| 09 | [phase-09.md](phase-09.md) | Corpus migration, shadow rollout, production enablement, and operations | 00–08 |

## Cross-Phase Data Contracts

All persisted records introduced by this work should include:

- `schema_version`;
- immutable record ID;
- project ID;
- source provider and session ID;
- source event IDs;
- creation and observation timestamps;
- content hash;
- sensitivity classification;
- stability classification;
- source-of-truth reference when applicable; and
- lineage IDs linking capture, distilled pattern, training example, evaluation
  probe, training run, and adapter.

JSONL remains acceptable for append-only capture and human inspection. Indexes,
manifests, and lifecycle documents must be atomically replaced and
schema-validated.

## Human Decision Gates

The following decisions must be made before their dependent phase is accepted:

1. Exact non-FP8 Qwen3.6-27B training artifact and pinned revision.
2. Proof that its LoRA adapter is compatible with the FP8 inference artifact.
3. General coding anchor source, license, allowed redistribution, and mixing
   policy.
4. Retention and redaction policy for code diffs and developer corrections.
5. Whether an LLM judge is allowed in evaluation; deterministic checks remain
   mandatory regardless.
6. Minimum evidence, corpus size, update count, and promotion thresholds.
7. Provider hook changes in paths marked `humans-only`.

## Autonomy Boundaries

`clare2` is supervised and requires human review. The following relevant paths
are `humans-only` and must not be generated by an AI implementation phase:

- `clare2/templates/hooks`;
- `clare2/scripts/clare2-install-hooks.sh`;
- all files under `clare`, including
  `clare/scripts/clare2-capture-event.sh`.

If provider hook changes are necessary, the phase must document the required
contract and stop for a human-supplied change. Changes to agent configuration
must obey the repository's cross-agent synchronization rule.

## Verification Policy

Every implementation phase that edits files must:

1. add focused unit and integration tests for its invariants;
2. test corrupt, empty, duplicate, stale, and interrupted inputs;
3. run the applicable service tests;
4. run `./clare/verify-ci.sh` according to the repository's terminating
   two-attempt rule; and
5. attach production or fixture evidence listed in that phase's exit criteria.

GPU validation belongs on production or a designated compatible host. Unit
tests must not require a GPU.

## Rollout Policy

- No phase directly enables nightly training.
- New schemas and assemblers run in shadow mode beside v1 until compared.
- Existing corpus and adapters are preserved, not rewritten in place.
- Every migration writes to a versioned destination and produces a manifest.
- Enabling production training is an explicit Phase 09 human action.
- Rollback always restores base-model routing and the last known-good corpus
  pointer without deleting evidence.

## Definition of Complete

This plan is complete only when:

- at least one project produces a valid v2 correction-backed corpus;
- the supported QLoRA canary passes repeatedly from a clean container;
- the candidate shows learning on held-out project probes without general
  regression;
- the full evaluation report is persisted whether promoted or rejected;
- one failing project does not block another;
- a candidate is promoted through the normal lifecycle;
- routing selects it only for the intended project and capabilities;
- rollback to the base or prior approved adapter is demonstrated; and
- operations documentation explains monitoring, failure triage, retention,
  and recovery.


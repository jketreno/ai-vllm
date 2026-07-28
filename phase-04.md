# Phase 04 — Failure-Safe Hierarchical Summarization

Status: proposed  
Depends on: Phases 02 and 03  
Produces: scalable summaries and durable themes with last-known-good safety

## Goal

Make long-term corpus memory reliable before it is used for training.

## Design

Replace single-call whole-period summarization with bounded map/reduce:

1. Partition by project, category, disposition, and bounded token count.
2. Summarize each partition with structured output.
3. Validate provenance and evidence totals.
4. Deterministically merge exact identities.
5. Run bounded semantic merge over remaining candidates.
6. Validate the reduction.
7. Publish atomically only when every required partition succeeds.

Summaries must preserve source pattern IDs and must never fabricate additional
evidence.

## Publication Rules

- Write a candidate summary under a run-specific staging path.
- Produce a manifest with input hashes, prompt/model versions, partitions,
  success/failure counts, and output hash.
- Promote the candidate pointer only after complete validation.
- Keep the previous active summary on empty, invalid, partial, or timed-out
  output.
- Treat a legitimately empty summary as a typed outcome with exclusion reasons,
  not merely an empty file.
- Support retry of failed partitions without redoing successful immutable ones.

## Theme Policy

Remove the requirement to wait a calendar quarter before any durable theme can
exist. Promote a theme when evidence policy is met across distinct sessions and
configured observation time, while still producing weekly/monthly/quarterly
views for compression and review.

Theme promotion must consider disposition and expiry. Only active `weights`
records can become weight themes.

## Primary File Targets

- `clare2/pipeline/app/summarizer.py`
- `clare2/pipeline/app/structured_output.py`
- `clare2/pipeline/app/metrics.py`
- summarizer prompts and manifests
- `clare2/pipeline/tests/test_policy.py` or dedicated summarizer tests

## Tests

- A 100+ record fixture partitions into bounded calls.
- One failed partition prevents publication but retains last known good.
- Retry reuses successful partitions.
- Evidence counts equal the union of source event IDs.
- Duplicate source IDs are counted once.
- Empty-by-policy differs from parse failure.
- Active theme pointers update atomically.
- Expired/context/discard records cannot become weight themes.
- Theme promotion is not calendar-quarter dependent.

## Production Validation

Shadow-run the new summarizer over the period where production yielded:

- Zoo-Code `12 → 11`;
- ai-vllm `94 → 0`;
- auto-sam `22 → 0`; and
- webserver `8 → 0`.

The v2 run must complete all projects without replacing valid output with a
parse failure. Human review compares merged themes with source patterns.

## Rollback

Move the active pointer back to the previous immutable summary manifest. Do not
delete failed candidates or partition diagnostics.

## Exit Criteria

- Large-project summarization succeeds reproducibly.
- Invalid output cannot erase good state.
- At least one reviewed durable theme can be produced without waiting a
  quarter.
- `verify-ci.sh` passes.


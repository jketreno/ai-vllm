# Phase 05 — Training Corpus Construction and Admission

Status: proposed  
Depends on: Phases 03 and 04  
Produces: versioned train/validation/evaluation corpora with anchor blending

## Goal

Build a dataset that is large enough, traceable, nonduplicative, appropriately
weighted, and safe for specialization without avoidable general-capability
loss.

## Corpus Components

1. Verified correction-backed project examples.
2. Recurrent stable project weight patterns.
3. Reviewed durable themes.
4. A licensed general coding anchor dataset.
5. A held-out project evaluation split.
6. A held-out general regression split.

Volatile context and discarded records are excluded.

## Example Construction

Prefer real task/attempt/correction examples. Pattern-only records may produce
behavioral SFT examples, but targets must be reviewed or grounded in verified
source content. Remove generic filler templates that merely repeat the pattern.

Each example retains lineage and includes:

- example ID and schema version;
- source pattern/correction IDs;
- prompt and target;
- category and disposition;
- sampling weight;
- project;
- content and semantic-cluster hashes;
- split assignment;
- source license where relevant; and
- token counts from the real tokenizer.

## Deduplication and Splitting

- Exact deduplication by normalized content hash.
- Semantic clustering before split assignment.
- All examples in one lineage/cluster stay in one split.
- Held-out examples must not appear as themes, paraphrases, or canonical
  excerpts in training.
- Deterministic split assignment from versioned seed and stable IDs.

## Weighting

Use an explicit sampler or deterministic replication strategy supported by the
trainer. Validate effective sample counts. Category weights alone are
insufficient; correction confidence, recurrence, freshness, and anchor ratio
must be represented.

## Admission Rules

Reject training when any configured condition fails:

- insufficient project examples;
- insufficient correction-backed signal;
- insufficient estimated optimizer updates;
- excessive duplicate rate;
- missing anchor license/hash;
- empty held-out split;
- tokenization rejection above threshold;
- sensitivity or expiry violation; or
- schema/lineage validation failure.

Use actual tokenizer counts, not the four-characters-per-token estimate.

## Human Decisions

- Anchor dataset and license.
- Project-to-anchor mixing ratio.
- Minimum records and optimizer updates.
- Split percentages.
- Whether low-confidence pattern-only examples are allowed.

## Primary File Targets

- `clare2/pipeline/app/corpus.py`
- new corpus schemas, sampler manifests, and validators
- `clare2/train/train.py`
- corpus and trainer tests
- configuration and operator reporting

## Tests

- Weights affect effective sampling.
- Exact and semantic duplicates do not cross splits.
- Lineage does not leak into evaluation.
- Context/discard/expired records are excluded.
- Actual tokenizer counts govern over-length decisions.
- Tiny Zoo-Code-like data is rejected for insufficient updates.
- Missing or unlicensed anchor data blocks admission.
- Same inputs and seed produce byte-identical manifests.
- Corrupt source records quarantine without silently shrinking below admission.

## Production Validation

Build v2 shadow corpora for all projects and report:

- accepted/excluded counts by reason;
- correction-backed share;
- anchor share;
- duplicate clusters;
- train/validation/evaluation counts;
- token distributions;
- estimated optimizer updates; and
- sensitivity and expiry exclusions.

No GPU training occurs in this phase.

## Rollback

Restore the prior corpus pointer. Keep v2 manifests and exclusion reports for
audit.

## Exit Criteria

- A corpus cannot be admitted with one or a handful of examples.
- Sampling weights demonstrably affect input frequency.
- Held-out and training lineages are disjoint.
- Anchor provenance is approved.
- `verify-ci.sh` passes.


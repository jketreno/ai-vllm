# Phase 03 — Knowledge Routing, Expiry, and Sensitivity

Status: proposed  
Depends on: Phase 02  
Produces: deterministic `weights`, `context`, or `discard` disposition

## Goal

Prevent volatile operational details, secrets, temporary incidents, and
obsolete bugs from entering adapter weights.

## Dispositions

### Weights

Stable behavioral preferences, coding conventions, architectural boundaries,
and durable domain reasoning that should generalize without retrieval.

### Context

Facts that can change independently of model behavior:

- paths and hostnames;
- deployment topology;
- current API fields or versions;
- operational runbooks;
- active incidents;
- environment capacities;
- service health state; and
- source-of-truth facts best retrieved at request time.

### Discard

- secrets or sensitive data;
- unsupported inference;
- duplicate or contradictory material;
- obsolete incident state;
- assistant self-description without correction or recurrence;
- generic behavior already supplied by system policy;
- incomplete/abandoned task narration; and
- unverified claims.

## Classification

Use deterministic rules for clear cases, followed by a structured LLM
recommendation for ambiguous cases, followed by policy validation. Store both
the recommendation and final disposition.

Each retained record includes:

- disposition;
- rationale code;
- confidence;
- validity interval;
- expiry/review date;
- sensitivity level;
- source-of-truth reference;
- supersedes/superseded-by lineage; and
- reviewer when manually overridden.

LLM output cannot downgrade sensitivity or extend expiry beyond policy limits.

## Conflict Handling

- Exact source-of-truth references outrank inferred patterns.
- Newer facts do not silently erase older records; they supersede them.
- Contradictory weight candidates are quarantined for review.
- Context records must be retrievable by project and validity time.
- Expired records are excluded from assembly but retained for audit.

## Primary File Targets

- new routing/policy module under `clare2/pipeline/app`
- versioned configuration under `clare2/pipeline/config`
- corpus manifest and metrics integration
- tests for classification, expiry, conflicts, and sensitivity
- operator read APIs for disposition and quarantine inspection

## Tests

- Hostnames, absolute production paths, and current incidents route to context.
- Durable naming and architectural rules can route to weights.
- Seeded secrets always discard regardless of LLM recommendation.
- Expired context is excluded.
- Superseded facts are not simultaneously active.
- Manual override is audited.
- Contradictory weight rules quarantine.
- Policy configuration changes alter disposition version, not historical data.

## Production Validation

Classify the current Zoo-Code and ai-vllm corpora in shadow mode. The report
must specifically identify:

- the “training is expected to crash” record;
- production hostnames and absolute paths;
- historical CI-debt descriptions;
- stable CLARE verification rules;
- duplicate verify-ci patterns; and
- durable project architecture.

Human review samples each disposition and records precision targets before
Phase 05 can use the results.

## Rollback

Retain classifications as shadow metadata. Continue reading v1 episodes without
using v2 disposition for production training.

## Exit Criteria

- Sensitive or volatile examples cannot reach a weight corpus.
- Expiry and supersession are deterministic.
- Shadow classification meets the agreed review threshold.
- `verify-ci.sh` passes.


# Phase 08 — Held-Out Evaluation and Explainable Promotion

Status: proposed  
Depends on: Phases 05 and 07  
Produces: project-aware evaluation with durable promotion/rejection evidence

## Goal

Measure whether an adapter learned its project behavior while preserving
general coding ability, then persist an explainable decision.

## Evaluation Suites

### Project Held-Out

Derived from Phase 05 evaluation examples and excluded semantic lineages:

- correction replay;
- architectural choice;
- antipattern avoidance;
- project terminology;
- code generation or patch behavior; and
- relevant verification behavior.

### General Regression

From the approved anchor evaluation split:

- syntax and basic coding;
- instruction following;
- multi-step reasoning;
- safe tool-use behavior where applicable; and
- language/framework coverage relevant to the deployment.

### Operational Safety

- base fallback;
- project routing isolation;
- adapter load/unload;
- deterministic generation settings;
- timeout/failure handling; and
- no cross-project leakage.

## Scoring

Prefer deterministic scoring:

- tests compile/pass;
- exact structured fields;
- schema validation;
- forbidden/required behavior checks;
- patch application;
- repository verification fixtures.

An LLM rubric judge may supplement but never replace deterministic mandatory
checks. If used, pin judge model/prompt and measure judge agreement on a
human-labeled calibration set.

No probe passes merely because its expected keyword is empty.

## Promotion Policy

The policy should include:

- all mandatory safety checks pass;
- project score exceeds an absolute threshold;
- project score improves over base by a minimum margin;
- no protected general category regresses beyond tolerance;
- no category has insufficient sample size;
- training canary and corpus manifest are valid;
- base/inference fingerprints match; and
- complete report persistence succeeds before registry transition.

Thresholds are versioned configuration and included in the decision hash.

## Persistence

Both promotion and rejection store:

- full report;
- per-probe candidate and baseline outputs or approved hashes/redacted forms;
- scorer versions;
- corpus/split manifest hashes;
- thresholds;
- decision reasons;
- model and adapter fingerprints; and
- evaluation duration/errors.

Registry state must never say `rejected` with `evaluation: null`.

## Primary File Targets

- `clare2/pipeline/app/evaluator.py`
- `clare2/pipeline/prompts/eval_probes.jsonl` or versioned replacement
- `clare2/pipeline/app/lifecycle.py`
- `clare2/pipeline/app/registry.py`
- evaluation schemas, fixtures, and tests
- `clare2/scripts/clare2-ab-evaluate.py`

## Tests

- Project A probes never evaluate Project B's adapter for promotion.
- Training lineage cannot appear in held-out probes.
- Empty expectations fail schema validation.
- A keyword-containing wrong answer does not pass deterministic scoring.
- General regression blocks promotion.
- Rejection persists the full report.
- Persistence failure blocks status transition.
- Re-evaluation is reproducible from hashes and configuration.
- Manual promotion requires a stored report and audited override reason.

## Production Validation

Evaluate:

1. base versus base as a reproducibility control;
2. a known no-op adapter;
3. a deliberately overfit fixture;
4. a canary-trained candidate; and
5. cross-project routing isolation.

Human review compares automated results with expected decisions.

## Rollback

Restore the previous evaluation policy version and base routing. Retain all
reports and decision records.

## Exit Criteria

- Evaluation measures actual held-out project behavior.
- General regression is protected.
- Promotion and rejection are fully explainable.
- No rejected adapter lacks a report.
- `verify-ci.sh` passes.


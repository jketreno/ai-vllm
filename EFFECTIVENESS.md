# CLARE2 LoRA Effectiveness Report

Generated: 2026-07-06

## Summary

I ran the CLARE2 dream-training flow against the current assembled `ai-vllm` corpus and
then ran the built-in candidate-vs-base evaluator plus the external proxy A/B script.

The FP8 direct LoRA path can produce an adapter that vLLM loads over
`Qwen/Qwen3.6-27B-FP8`, but the adapter was not effective and was correctly rejected.

## Final Run

- Dream run: `run-20260706T213320Z-dream`
- Adapter: `clare-ai-vllm-20260706T213337Z-7df5d0dfae13`
- Project: `ai-vllm`
- Corpus: `/corpus/training/ai-vllm/current.jsonl`
- Training records: `5`
- Corpus hash: `7df5d0dfae13fc8554c88828765420075e140e9265f5cf32f395761347ef5e38`
- Train base: `Qwen/Qwen3.6-27B-FP8`
- Inference base: `Qwen/Qwen3.6-27B-FP8`
- Revision: `e89b16ebf1988b3d6befa7de50abc2d76f26eb09`
- Effective training mode: `fp8-16bit-lora`
- Final training loss: `14.2742`
- Lifecycle outcome: `rejected`

## What Happened

The first dream run exposed a real fingerprint bug: the registry base said
`Qwen3_5MoeForConditionalGeneration`, while vLLM and the trained adapter reported
`Qwen3_5ForConditionalGeneration`. I fixed the Compose/runtime base architecture and
allowed registry base refresh when all existing adapters are inactive.

After that fix, the second run loaded the adapter successfully through vLLM:

- `/v1/load_lora_adapter`: `200 OK`
- Adapter status after evaluation: `rejected`
- Current alias: `null`
- Rollback alias: `null`

## Training Loss

| Step | Loss |
| ---: | ---: |
| 1 | 14.2891 |
| 2 | 14.2438 |
| 3 | 14.2898 |

The loss stayed high and flat. With only five records and three optimizer steps,
this is not evidence of useful learning.

## Memory

Dream mode freed the expected memory by stopping the main AI services while keeping
spam classification and observability awake.

| Stage | MemAvailable |
| --- | ---: |
| before sleep | 12.0 GB |
| before training | 104.3 GB |
| after training | 103.8 GB |
| after wake | 14.5 GB |

Inside the trainer, CUDA free memory dropped from about `96.96 GB` before model load
to about `30.49 GB` after training.

## Built-In Evaluation

The lifecycle evaluator compared the candidate against the base model on 20 probes.

| Model | Passed | Total | Pass rate |
| --- | ---: | ---: | ---: |
| Base `Qwen/Qwen3.6-27B-FP8` | 2 | 20 | 0.10 |
| Candidate LoRA | 2 | 20 | 0.10 |

Outcome:

- `approved`: `false`
- `mandatory_pass`: `false`
- `no_category_regression`: `true`

The candidate did not improve on the base.

## Proxy A/B Test

I ran:

```bash
python3 clare2/scripts/clare2-ab-evaluate.py --output logs/clare2/dream/ab-run-20260706T213320Z.json
```

The script refused to run routed LoRA prompts because no approved or loaded adapter
matched `project='ai-vllm'` and `capabilities=['code', 'review']`.

That is the correct safety behavior. Since the adapter was rejected, CLARE2 did not
offer it through project routing. The built-in evaluator is therefore the completed
A/B result for this run.

## Recommendation

Do not promote `clare-ai-vllm-20260706T213337Z-7df5d0dfae13`.

The next useful experiment is not more FP8 direct LoRA with the same corpus. The better
next step is to cache a matching non-FP8 Qwen3.6 training base, train QLoRA from that
base, and validate whether its adapter can load over the FP8 inference model. Also
increase the corpus substantially; five SFT records is enough to test machinery, not
to teach durable behavior.

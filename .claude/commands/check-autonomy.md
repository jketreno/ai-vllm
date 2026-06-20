# /project:check-autonomy — Check the autonomy level for a file path

Given a file path, look up its autonomy level in `clare/autonomy.yml` and explain what it means for AI-generated changes.

## Usage

`/project:check-autonomy [path]`

Example: `/project:check-autonomy src/payment/processor.ts`

## Instructions

1. Read `clare/autonomy.yml`
2. Find the most specific matching path for the given argument (longest prefix match wins)
3. Report the level, reason, and what action is appropriate

## Response Format

```
## Autonomy Check: [path]

Matched rule: [matched path pattern]
Level: [full-autonomy | supervised | humans-only]
Reason: [reason from autonomy.yml]

### What this means:
[Explain the appropriate AI behavior for this level]

### Source of Truth (if applicable):
[If the path relates to a declared domain concept, show the source_of_truth entry]
```

## Level Explanations

**full-autonomy**: AI can proceed normally. No special restrictions or reminders needed.

**supervised**: AI can generate code but must add: "⚠ Human review required — this path is marked `supervised` in `clare/autonomy.yml`." at the end of the response.

**humans-only**: AI must NOT generate code. Respond with: "This path is marked `humans-only` in `clare/autonomy.yml`. I won't generate code here. Please make this change yourself, or update the autonomy level if that boundary no longer applies."

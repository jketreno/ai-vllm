# /project:distill — Trigger an on-demand CLARE₂ distillation pass

Sends a distillation request to the CLARE₂ pipeline container, which processes
today's session files and writes extracted patterns to the episode store.
Can be run any time; the nightly cron also runs at 22:00 UTC automatically.

## Instructions

```bash
AI_VLLM_ROOT="${CLARE2_ROOT:-../ai-vllm}"
PIPELINE_URL="${CLARE2_PIPELINE_URL:-http://127.0.0.1:8000}"
TOKEN_FILE="${CLARE2_OPERATOR_TOKEN_FILE:-${AI_VLLM_ROOT}/secrets/clare2_operator_token}"
TOKEN=$(<"${TOKEN_FILE}")

# Trigger the distillation pass
RESPONSE=$(curl -sf -X POST "${PIPELINE_URL}/distill/trigger" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" 2>&1)

if [[ $? -ne 0 ]]; then
  echo "❌ Could not reach the CLARE₂ pipeline at ${PIPELINE_URL}"
  echo "   Is the pipeline container running?"
  echo "   Start it with: docker compose -f ${AI_VLLM_ROOT}/docker-compose.yml up -d clare2-policy"
  exit 1
fi

echo "✅ Distillation pass triggered."
echo "   The pipeline is processing today's sessions in the background."
echo ""

# Poll for status
sleep 5
STATUS=$(curl -sf "${PIPELINE_URL}/distill/status" \
  -H "Authorization: Bearer ${TOKEN}" 2>/dev/null || echo '{}')
echo "Corpus stats:"
echo "$STATUS" | python3 -m json.tool 2>/dev/null || echo "$STATUS"
```

After running, report:
1. Whether the pipeline accepted the request
2. The current corpus stats (episodes by category, last distillation timestamp)
3. If the pipeline is unreachable, how to start it

## Notes

- Distillation is idempotent for the same day's sessions — re-running appends
  only new patterns that weren't already in the episode file.
- The recurrence gate (evidence_count ≥ 2) applies during distillation.
  Single-occurrence patterns are dropped.
- Full nightly schedule: distill at 22:00 → summarize at 22:30 → assemble
  corpus at 23:30 → train at 00:00 → new Temper live by 06:00.

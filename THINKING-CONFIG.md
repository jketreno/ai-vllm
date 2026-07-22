# Qwen Thinking Configuration

CLARE serves Qwen through the authenticated policy proxy at
`/v1/chat/completions`. The proxy selects the approved base model or scoped LoRA
adapter, but it does not add, remove, or clamp thinking options. Request behavior
therefore matches direct vLLM behavior for the same payload.

Raw vLLM remains private to Docker networks. Application callers should send
requests to the CLARE policy proxy.

## Caller Fields

Callers can pass vLLM's Qwen options directly:

- `chat_template_kwargs.enable_thinking` enables or disables thinking.
- `chat_template_kwargs.preserve_thinking` controls whether prior thinking is
  retained by the chat template.
- `thinking_token_budget` limits reasoning tokens. Requires vLLM with
  spec-decode-aware thinking-budget support (upstream vLLM PR #34668,
  first in v0.20.2); confirmed enforced on this deployment's `26.06-py3`
  vLLM image. On `26.04-py3` this field was silently ignored whenever MTP
  speculative decoding was active — the model could burn its entire
  `max_tokens` allowance on reasoning regardless of the configured budget.
- `max_tokens` limits the complete generated output, including reasoning.

The proxy preserves these values exactly. If they are omitted, vLLM and the
model's chat template determine their defaults.

## Client Compatibility

Clients must understand streamed `delta.reasoning` events to use thinking mode.
A client that only recognizes `delta.content` can report an empty assistant
message when reasoning consumes the entire `max_tokens` budget.

For clients such as Roo Code that do not reliably consume reasoning events,
disable thinking in the request:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

For clients that support reasoning events, reserve enough total output for both
reasoning and the visible answer:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": true
  },
  "thinking_token_budget": 512,
  "max_tokens": 4096
}
```

### Clients That Cannot Set a Custom Request Body

Some clients only expose a UI for custom HTTP headers and give no way to add
fields to the JSON body (Roo Code / Zoo Code's "Custom Headers" setting is a
common example). For these, send the same fields as a JSON object in the
`X-CLARE2-Params` header instead; the proxy deep-merges it into the request
body before forwarding upstream. Nested objects are folded key by key rather
than replaced wholesale — a body `chat_template_kwargs.preserve_thinking` and
a header `chat_template_kwargs.enable_thinking` both survive in the merged
result, with the header winning only on keys present in both.

Header name: `X-CLARE2-Params`

To disable thinking entirely (no reasoning phase at all — the deterministic
option for clients that never consume `delta.reasoning`):

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

To keep thinking on but cap it, so a turn can never come back with an empty
`content` purely from exhausting `max_tokens` mid-thought (requires vLLM with
spec-decode-aware thinking-budget support; confirmed enforced on this
deployment's `26.06-py3`, not on the prior `26.04-py3`):

```json
{"chat_template_kwargs": {"enable_thinking": true}, "thinking_token_budget": 512}
```

Don't combine `enable_thinking: false` with `thinking_token_budget` — with
thinking disabled there is no reasoning phase for the budget to bound, so the
field is a no-op.

The header value must be valid JSON and must be an object, or the proxy
returns `400`. Like the JSON body's `model` field, `model` cannot be set
through this header — the proxy always overwrites it with the resolved route
or base model.

## Routing Behavior

The proxy overwrites the request's `model` field with either the configured base
model or the LoRA adapter selected by a valid CLARE route. It also handles
authentication, endpoint allowlisting, maintenance mode, and adapter lifecycle.
All other JSON request fields and all upstream response bytes pass through
unchanged.

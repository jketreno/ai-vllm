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
- `thinking_token_budget` limits reasoning tokens when supported by the deployed
  vLLM/model combination.
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

## Routing Behavior

The proxy overwrites the request's `model` field with either the configured base
model or the LoRA adapter selected by a valid CLARE route. It also handles
authentication, endpoint allowlisting, maintenance mode, and adapter lifecycle.
All other JSON request fields and all upstream response bytes pass through
unchanged.

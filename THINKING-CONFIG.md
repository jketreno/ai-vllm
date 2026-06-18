# CLARE Thinking Configuration

CLARE serves Qwen through the authenticated policy proxy at
`/v1/chat/completions`. The deployment keeps Qwen thinking enabled by default,
but bounded, so callers get useful reasoning without letting agent tasks spend
unbounded time inside `<think>` output.

The balanced default reasoning budget is `1024` tokens, with a deployment
maximum of `2048` tokens. Callers may override the budget per request with
OpenAI-compatible extra body fields when they need faster responses, deeper
reasoning, or no thinking at all.

Raw vLLM remains private to Docker networks. Application callers should send
these options to the CLARE policy proxy, not directly to `vllm-engine`.

## Caller Fields

`thinking_token_budget`

: Per-request cap for reasoning tokens. vLLM counts from the reasoning start
  marker and forces the reasoning block to end when this budget is reached.
  Use this to bound latency while keeping thinking enabled.

`chat_template_kwargs.enable_thinking`

: Explicit thinking-mode switch for Qwen chat templates. Set it to `true` to
  request thinking, or `false` for quick classification, formatting, routing,
  and other low-reasoning calls.

`max_tokens`

: Total generated output budget for the request. This is separate from the
  reasoning budget and should still be sized for the visible answer you need.

`temperature`

: Sampling control. Lower values can make answers more direct and repeatable,
  but this is not a hard thinking limit.

`seed`

: Reproducibility control for deterministic-style calls. This is useful for
  tests and evaluations, but it does not bound thinking.

## Budget Profiles

Use `512` for fast, simple tasks where latency matters more than extended
reasoning.

Use `1024` for normal agent and assistant calls. This is the balanced default.

Use `2048` for deeper reasoning when the caller expects multi-step analysis but
still needs a hard stop.

Use `chat_template_kwargs.enable_thinking=false` for no-thinking calls. Some
vLLM/model combinations also treat a `thinking_token_budget` of `0` as no
reasoning budget, but the explicit chat-template switch is the clearest way to
request no thinking from Qwen.

## Precedence

If the caller omits thinking configuration, the proxy supplies bounded-thinking
defaults.

If the caller provides `thinking_token_budget`, the proxy honors it unless it is
above the deployment maximum, in which case the proxy may clamp it.

If the caller provides `chat_template_kwargs.enable_thinking=false`, that
request disables thinking and should not receive a default thinking budget.

If the caller provides `chat_template_kwargs.enable_thinking=true`, thinking is
enabled for that request and the caller's budget, or the proxy default, applies.

The deployment defaults are configurable with
`CLARE2_DEFAULT_ENABLE_THINKING`, `CLARE2_DEFAULT_THINKING_TOKEN_BUDGET`, and
`CLARE2_MAX_THINKING_TOKEN_BUDGET`.

## LangGraph and LangChain

When using `langchain_openai.ChatOpenAI`, pass vLLM-specific fields through the
OpenAI extra body. CLARE ignores the caller's `model` value and routes to the
approved base model or adapter.

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="ignored-by-clare",
    base_url="http://127.0.0.1:8000/v1",
    api_key=clare2_proxy_token,
    temperature=0.4,
    model_kwargs={
        "extra_body": {
            "thinking_token_budget": 1024,
            "chat_template_kwargs": {"enable_thinking": True},
        }
    },
)
```

For a faster node:

```python
fast_llm = llm.bind(
    extra_body={
        "thinking_token_budget": 512,
        "chat_template_kwargs": {"enable_thinking": True},
    }
)
```

For a no-thinking formatting or classification node:

```python
formatting_llm = llm.bind(
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
    }
)
```

## Raw HTTP

Send the same fields in the JSON body when calling the CLARE policy proxy.

```bash
CLARE_TOKEN=$(<secrets/clare2_proxy_token)

curl --fail --silent \
  -H "Authorization: Bearer ${CLARE_TOKEN}" \
  -H "Content-Type: application/json" \
  --data '{
    "model": "ignored-by-clare",
    "messages": [
      {
        "role": "user",
        "content": "Think through the deployment options, then recommend one."
      }
    ],
    "thinking_token_budget": 1024,
    "chat_template_kwargs": {"enable_thinking": true},
    "max_tokens": 1024,
    "temperature": 0.4
  }' \
  http://127.0.0.1:8000/v1/chat/completions
```

No-thinking request:

```json
{
  "model": "ignored-by-clare",
  "messages": [
    {"role": "user", "content": "Format this JSON object without commentary."}
  ],
  "chat_template_kwargs": {"enable_thinking": false},
  "max_tokens": 512,
  "temperature": 0
}
```

## Notes

The spam classifier is separate from this caller contract. It already disables
thinking internally and uses constrained JSON output for deterministic
classification.

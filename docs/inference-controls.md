# Steering at the inference layer

Your client (opencode, or anything else speaking `/v1`) is not the only place
behaviour is decided. `ds4-server` applies a chat template, defaults to thinking
mode, and overrides sampling in ways that will surprise you if you assume
llama.cpp or vLLM semantics. This is what is actually available on the wire.

## System prompts work, and they work well

Standard `role: system` messages are honoured — no special flag, no template
wrangling:

```bash
curl -s localhost:8888/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "ds4",
  "messages": [
    {"role": "system", "content": "You are a terse assistant. Answer with a single word, lowercase, no punctuation."},
    {"role": "user",   "content": "What is the capital of France?"}
  ]}' | jq -r '.choices[0].message.content'
```
```
paris
```

Instruction-following is tight enough that a system prompt is the right place to
pin output format for machine consumers. The benchmark harness relies on this:
`tools/bench.py needle` pins the answer format with a system prompt, and that
alone took retrieval from an apparent 2/6 to a true 6/6 (see below).

## Thinking mode is ON by default

This is the single most surprising default. DeepSeek-compatible chat requests
default to thinking mode, so the model deliberates before answering:

| request | completion tokens | `content` |
|---|---:|---|
| default (thinking on) | 56 | `paris` |
| `"thinking": {"type": "disabled"}` | **2** | `paris` |
| `"model": "deepseek-chat"` | **2** | `paris` |

Same answer, **28x the tokens**. At 20-27 tok/s that is the difference between a
0.1 s reply and a 2.5 s one.

Reasoning is returned out-of-band in `message.reasoning_content`, so `content`
stays clean — you do not have to strip `<think>` tags yourself.

Three equivalent ways to turn it off:

```jsonc
{"thinking": {"type": "disabled"}}   // Anthropic-style
{"think": false}                     // shorthand
{"model": "deepseek-chat"}           // model-name selector
```

And to turn it up, `reasoning_effort` (or `output_config.effort`) takes `low`,
`high` or `max`, honoured at any `--ctx`.

## Thinking mode ignores your sampling knobs

From `ds4-server --help`:

> API defaults are temperature=1, top_p=1, min_p=0.05, and no top-k cap.
> **In thinking mode, client sampling knobs are ignored like the official API.**

So `temperature: 0` in a thinking-mode request does nothing. Runs are not
reproducible, and any benchmark that assumes greedy decoding is lying to you.
This is why `tools/bench.py` disables thinking by default and only enables it
behind `--thinking`.

If you need determinism, you need non-thinking mode.

## The truncation trap

Thinking mode plus a small `max_tokens` produces garbage, and it is worth
recognising the failure signature. If the token budget runs out before the
reasoning block closes, the server logs:

```
ds4-server: thinking not closed, ignoring DSML in reasoning
```

and emits the raw, unfinished deliberation as `content`. It looks like the model
ignored your question:

```
depth  measured  place  found  answer
32768     32180    50%     NO   The user is asking for the vault combination for
```

That row is not a retrieval failure. That is a 32-token cap cutting off a model
that was still thinking. With thinking disabled and the format pinned by a system
prompt, the identical prompts give:

```
32768     32198    10%    yes   43-64-12
32768     32198    50%    yes   46-15-15
32768     32196    90%    yes   41-29-81
131072   128578    10%    yes   56-39-95
131072   128576    50%    yes   31-25-62
131072   128579    90%    yes   43-80-30
needle: 6/6 recovered
```

**6/6 at 128k tokens, including the needle at 90% depth.** The window is not just
held, it is used. Any short-output automation — classifiers, routers, tool-call
extractors, retrieval — should disable thinking or budget several hundred tokens
for it.

## Practical guidance for coding agents

- **Tool-call and edit loops: thinking off.** These are format-bound, not
  reasoning-bound. You get the same edits 28x cheaper, and you get determinism
  back.
- **Planning and debugging turns: thinking on**, optionally
  `reasoning_effort: high`. This is what the mode is for.
- **Put machine-facing format contracts in the system prompt.** They hold.
- **Do not cap `max_tokens` tightly while thinking is on.** Either disable it or
  leave real headroom, or you will get truncated deliberation as your answer and
  misdiagnose it as a model quality problem.
- ds4 also honours `tools`, `tool_choice`, `stop`, `seed`, `top_p`, `top_k`,
  `min_p` and `stream` — the full list is in `/v1/models` under
  `supported_parameters`.

## Server-side defaults

A few knobs move the default for every request, which is useful when the client
cannot be changed:

| variable | effect |
|---|---|
| `DS4_SERVER_DEFAULT_TEMP` | default sampling temperature |
| `DS4_THINK_NONE` / `_LOW` / `_HIGH` / `_MAX` | thinking-mode presets |
| `DS4_REASONING_EFFORT_HIGH_PREFIX`, `DS4_REASONING_EFFORT_MAX_PREFIX` | prefixes injected for those effort levels |

The chat template itself is baked into the GGUF, so switching between `stock` and
`abliterated` does not change the wire format — only the behaviour past refusal.

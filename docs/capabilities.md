# Capabilities: what this box can do, measured

[`performance.md`](performance.md) is about how fast it is. This is about what it
can actually do, and where the sharp edges are. Everything here was measured on
this hardware; raw rows are in [`../bench/results/`](../bench/results).

## Is the abliterated default as capable as stock?

This repo defaults to abliterated weights, and the README's argument that
abliteration "costs nothing we can measure" cites work
([Arditi et al.](https://arxiv.org/abs/2406.11717),
[Kim et al.](https://arxiv.org/abs/2607.28607)) on **full-precision** models. Our
artifact is abliterated at λ=3.5 across layers 10–42 **and then quantized to ~2
bits**. Nobody had measured that combination, so the default was chosen on a
citation rather than evidence.

`ds4-eval` ships embedded GPQA Diamond, audited SuperGPQA, AIME2025 and COMPSEC
questions with grading. Both weight sets, identical settings — same 16 questions,
`--temp 0 --seed 1 -n 1536`:

| | stock | abliterated |
|---|---:|---:|
| passed | **13/16** | **13/16** |
| items failed | 4, 9, 15 | **4, 9, 15** |
| generated tokens (total) | 12,105 | 13,555 (**+12%**) |
| runtime | 9 min | 10 min |

**No detectable capability difference, and the agreement is item-level: both
models failed exactly the same three questions.** That is a stronger signal than
the matching scores, because two models with equal-but-different competence would
be unlikely to fail the same items.

All three shared failures hit the 1536-token cap (`gen=1536`), so they were
budget truncations mid-reasoning rather than wrong answers — two AIME2025
problems and one GPQA Diamond. What this measures is therefore *accuracy under a
1536-token budget*, which is the practically relevant number for an agent loop
but is not the same as raw capability.

**Read this result honestly: N=16 has very low statistical power.** At 13/16 each,
the confidence interval on the difference spans roughly ±25 points. This rules out
a *large* regression from abliteration-plus-2-bit; it cannot rule out a small one.
It is evidence, not proof, and it is the first local evidence either way.

The one real difference is verbosity: abliterated generated **12% more tokens**
for the same score, and used more on 9 of 16 items. At 20-27 tok/s that is a
latency cost with no accuracy benefit. Worth knowing if you are optimizing turn
time.

```bash
make eval                  # both weight sets, bounded, writes traces
make eval MODEL=stock      # one of them
```

Traces (every question, output and grading decision) land in
`bench/results/eval/`. For a stronger result, raise `EVAL_QUESTIONS` and `-n`
and run it overnight — the defaults here are sized to finish in ~20 minutes.

## Retrieval holds at 514k tokens

Throughput at depth is worthless if the model cannot see the middle of the
window. Needles at three placements, thinking disabled, format pinned by a
system prompt:

| depth | 10% | 50% | 90% |
|---:|---|---|---|
| 32,198 | yes | yes | yes |
| 128,578 | yes | yes | yes |
| **514,088** | **yes** | **yes** | **yes** |

**9/9.** The 514k row matters most because that is the regime this box is sold on
and the one that was previously untested. The window is used, not merely held.

## Fan-out: N agents against one shared context

The `concurrency` suite gives every stream a unique prefix, which isolates decode
but describes nobody's real workload. The `fanout` suite is the opposite: one deep
context, N agents asking different questions of it. That is the coding-agent
shape.

128k-token shared context, 128 generated tokens each:

| agents | TTFT p50 | aggregate decode | per agent | agents reusing prefix |
|---:|---:|---:|---:|---:|
| 1 | 1,023 ms | 8.5 tok/s | 8.50 | 1 of 1 |
| 2 | 3,802 ms | 22.0 tok/s | 11.00 | 2 of 2 |
| 4 | 5,252 ms | 31.0 tok/s | 7.75 | **4 of 4** |

Against a cold prefill of **131,095 ms** for that same context. So four agents
interrogating a 128k context reach first token in ~5 s, where four independent
cold prefills would cost ~524 s. `tokens_prefilled_cached` was 899,584 against
128,668 computed — a 7x reuse ratio.

**The mechanism is not what I predicted.** I expected in-memory bank forks
(`admits_fork`). Forks stayed at 0; the reuse came through
`admits_partial_truncate` and **disk-checkpoint restores into separate banks**
(`kv cache bank restore hit ... load=124.8 ms`, `... 236.0 ms`). Each agent gets
its own bank rehydrated from the same content-addressed file. That is arguably
better than forking, because it does not require the source bank to still be live.

```bash
make bench-fanout          # writes bench/results/<model>-fanout.jsonl
```

## Whole-repo-in-context

The window is large enough to stop doing retrieval. Packed with
`tools/repo-context.sh`:

| target | packed | est. tokens | cold prefill | disk checkpoint |
|---|---:|---:|---:|---:|
| this repo | 0.18 MiB | ~47k | ~50 s | 0.21 GiB |
| the whole ds4 engine (168k lines of C/CUDA) | 2.04 MiB | ~534k | ~9.4 min | 2.34 GiB |

So the entire inference engine fits in the window with room left over, and after
the first prefill every question against it is a warm admit.

```bash
tools/repo-context.sh size .                  # estimate before committing
tools/repo-context.sh pack . -o ctx.txt
tools/repo-context.sh warm ctx.txt            # pay the prefill once
tools/repo-context.sh ask ctx.txt "where is admission decided?"
```

The constraint is that the prefix must be **byte-identical** between turns. The
script always emits files in sorted order with stable headers and sends the packed
file verbatim before the question, so the prefix is stable as long as the file is.

## The tool-loop trap (this one will cost you minutes)

`misc/ANTHROPIC_LIVE_CONTINUATION.md` states the contract:
`tool_use_id -> exact sampled DSML/KV frontier`. A tool-result turn bound to a
live call id appends **only the suffix** to the live KV. Measured on the OpenAI
surface with a 2-turn tool loop:

| how the client replays the assistant turn | prompt reuse on the tool-result turn |
|---|---:|
| **verbatim, ids intact** (as returned by the server) | **395 of 412 tokens — 96%** |
| reconstructed, id missing or altered | **0 of 402 — full re-prefill** |

At 400 tokens nobody notices. **With a 300k-token repo context in the prefix, that
is the difference between ~0.5 s and 5+ minutes per tool call.** Any client that
normalizes, re-renders, or regenerates assistant tool-call messages instead of
echoing them back exactly will silently re-prefill the entire context on every
single tool call.

If you are wiring an agent to this server: echo the assistant message back byte
for byte, including `tool_calls[].id`. The engine's own note is blunt about the
failure mode — *"Unknown IDs are not silently repairable if the request does not
replay enough prior history."*

## Determinism

| mode | reproducible? |
|---|---|
| `temperature: 0`, thinking disabled | yes — identical output across runs; `seed` is a no-op |
| `temperature: 1` + `seed: N` | yes — same seed gives identical output, different seed differs |
| thinking enabled (ds4's default) | **no** — sampling knobs are ignored entirely |

So you can regression-test agent behaviour across config changes, but only
outside thinking mode. See [`inference-controls.md`](inference-controls.md).

## Context compaction: order your prompt so the expensive part never moves

`misc/COMPACT.md` specifies the agent-side contract: a soft trigger at 85% of
context (or ≤8192 tokens free), a durable task-state summary, a verbatim tail of
up to 10% of context capped at 50k tokens, then step 6 — *"Re-sync the live KV
state to the rebuilt transcript."*

That re-sync is the part with a cost. Compaction rewrites the **head** of the
prompt, which invalidates the prefix — so a compaction at 850k tokens throws away
the 131 s (or 9 min) of prefill you paid for and starts again from the rebuilt,
shorter transcript.

The architectural consequence for a repo-in-context workflow: **put the stable,
expensive material first and never let compaction touch it.** System prompt and
repo context as an immutable prefix; conversation after it; compact only the
conversational tail. Get the order wrong and every compaction re-prefills your
entire repository.

Also worth knowing, from the engine's CHANGELOG: Codex **never auto-compacts
against custom providers**, so long sessions die at the capacity wall rather than
degrading — *"For long agent sessions today, Claude Code remains the recommended
harness."* The server exposes `/v1/messages` (native Anthropic) alongside
`/v1/chat/completions`, `/v1/responses` and `/v1/batch`.

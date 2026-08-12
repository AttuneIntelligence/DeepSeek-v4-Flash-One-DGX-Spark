# Recorded benchmark runs

Raw output from `tools/bench.py`. The analysis, method and the reasoning behind
the serving defaults live in [`../../docs/performance.md`](../../docs/performance.md).

Each `.jsonl` file is one row per measurement, appended, never rewritten. Every
row carries its own metadata (`model`, `host`, `timestamp`, `thinking`) so rows
from different runs can be concatenated safely.

| file | suite | conditions |
|---|---|---|
| `abliterated-depth.jsonl` | `depth` | 1k → 771k tokens, `CTX=1000000`, 128 tokens generated, local NVMe |
| `abliterated-concurrency.jsonl` | `concurrency` | 32k prompts, 1/2/4 streams — **prefill-contaminated, kept as the counter-example** |
| `abliterated-concurrency-ctx1m.jsonl` | `concurrency` | 512-token prompts, 256 tokens generated — the clean decode-batching read |
| `abliterated-needle.jsonl` | `needle` | 32k and 128k, three placements each, thinking disabled |

Fields worth knowing:

- `prompt_tokens` is what the **server** counted, not what the harness asked for.
- `decode_tok_s`, `prefill_tok_s`, `ttft_ms` come from the server's `timings`
  block when `timing_source` is `server`; `wall` means they were derived from
  wall clock instead.
- `aggregate_tok_s` divides total output tokens by wall clock **including
  prefill**; `aggregate_decode_tok_s` sums the per-stream decode rates. Use the
  latter to reason about batching.
- `spread_tok_s` is the best-minus-worst across `--repeat` samples. It is often
  large at shallow depth because CUDA graphs get invalidated and recaptured;
  treat single-sample differences under ~15% as noise.
- `thinking` records whether ds4's default thinking mode was left on. It is off
  for these runs — see [`../../docs/inference-controls.md`](../../docs/inference-controls.md).

Regenerate:

```bash
make bench MODEL=abliterated       # everything, into this directory
make bench-depth                   # just the depth curve, to stdout
```

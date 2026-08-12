# Performance: what we measured, and why the defaults are what they are

Every number here was produced by `tools/bench.py` against this repo's launcher on
one DGX Spark (GB10 / SM121, 20 CPU cores, 121.69 GiB unified LPDDR5X), serving
the `abliterated` set from local NVMe with the DSpark drafter armed. Raw rows are
in [`bench/results/`](../bench/results).

The short version: **the box is GPU-bandwidth-bound, the 1M window is nearly free
to *hold* and expensive only to *fill cold*, and prefix caching is what makes deep
context practical.** The defaults optimise for a long-lived deep session, not for
a cold one-shot.

## Method, and how much to trust it

- Every prompt is prefixed with a random nonce, so "cold" rows really are cold —
  otherwise the prefix cache serves them and the number is meaningless.
- Depths are reported as the **measured** `prompt_tokens` from the server's own
  usage block, never the requested filler size.
- Rates come from the server's `timings` block (`ttft_ms`, `prefill_tok_s`,
  `decode_tok_s`, `spec_accept_rate`), not from wall clock, except where a row
  says `wall`.
- `--repeat N` reports best-of-N with the spread. **Spread matters here.**
  Repeated identical samples vary by up to 14 tok/s at shallow depth, largely
  because CUDA graphs get invalidated and recaptured (`invalidated 616 captured
  graph(s): f16 gemm activations`). Treat any single-sample difference under
  ~15% as noise. Several conclusions below were revised after adding repeats.

Reproduce with:

```bash
make bench                 # full sweep, writes bench/results/
make bench-depth           # just the depth curve
```

## Decode and prefill versus context depth

`abliterated`, `CTX=1000000`, 128 tokens generated per point, one stream.

| prompt tokens | TTFT | prefill tok/s | decode tok/s |
|---:|---:|---:|---:|
| 1,031 | 1.1 s | 947 | 16.5 |
| 8,058 | 8.8 s | 916 | 26.9 |
| 32,156 | 32.8 s | 982 | **27.2** |
| 128,530 | 132 s | 975 | 21.2 |
| 257,035 | 281 s | 914 | 21.6 |
| 514,041 | 631 s | 815 | 16.9 |
| 771,048 | 1,056 s | 730 | 13.2 |

Two things to read off this.

**Decode degrades gracefully, it does not fall off a cliff.** Peak is 27.2 tok/s
around 32k. At 771k tokens — 77% of the window — it still runs 13.2 tok/s, **49%
of peak while carrying 24x the context**. The 1,031-token row is *slower* than
the 8k row because it is cold: first request after boot pays graph capture.

**Prefill is the real cost of depth, and it is roughly linear.** ~950 tok/s
holding to ~730 tok/s at 771k. That linearity is the whole story: a cold 771k
prompt costs **17.6 minutes** before the first token. Which brings us to the
number that justifies the entire configuration.

## Prefix caching: why 1M context is practical

Same 128k prompt sent twice, three independent runs at `CTX=1000000`. Each run
uses a fresh nonce, so every cold pass really is cold:

| run | pass | served from cache | TTFT | decode tok/s |
|---:|---|---:|---:|---:|
| 1 | cold | 0 | 134,950 ms | 26.8 |
| 1 | warm | 128,512 | **462 ms** | 32.6 |
| 2 | cold | 0 | 133,399 ms | 24.3 |
| 2 | warm | 128,512 | **462 ms** | 31.5 |
| 3 | cold | 0 | 134,429 ms | 26.4 |
| 3 | warm | 128,512 | **426 ms** | 35.1 |

**Median 292x faster to first token** (range 289x-316x), with 99.99% of the
prompt reused. Both halves are stable across runs -- cold within 1%, warm within
8% -- so the spread in the ratio is small. Earlier single-sample measurements of
this gave 242x and 349x; neither was wrong, both were undersampled, which is why
the suite now takes `--repeat`.

Warm decode is also consistently faster than cold (31.5-35.1 vs 24.3-26.8
tok/s). A request admitted against a warm bank is not merely skipping prefill,
it starts generating in a better state than one that just finished pushing 128k
tokens through the prefill path.

In the server log this is the `warm admit` path:

```
ds4-server: warm admit bank=4 cached=277259 suffix=581
```

277,259 tokens already resident, 581 new tokens to prefill.

This is the case that matters for coding work. An agent editing a repository does
not send 300k fresh tokens per turn; it sends the same context plus a diff. You
pay the deep prefill **once per session**, then every subsequent turn costs
roughly half a second of TTFT regardless of how deep the window is. The linear
prefill cost above is a one-time entry fee, not a per-turn tax.

That is the rationale for keeping the window large: **context depth is cheap to
hold and cheap to extend; it is only expensive to establish.** A smaller window
would not make warm turns faster — they are already sub-second — it would only
force eviction and turn cheap warm turns back into expensive cold ones -- a 134
second penalty, every time it happens.

## Retrieval: the window is used, not just held

Throughput at depth is worthless if the model cannot see the middle of the
window. Three needle placements (10%, 50%, 90% depth), thinking disabled, answer
format pinned by a system prompt:

| depth | 10% | 50% | 90% |
|---:|---|---|---|
| 32,198 | yes | yes | yes |
| 128,578 | yes | yes | yes |

**6/6.** Note this suite is sensitive to how you ask: with thinking mode left on
(ds4's default) and a 32-token cap, the same prompts score 2/6 — not because
retrieval fails but because the cap truncates the reasoning block before it
closes and the server emits raw deliberation as the answer. See
[`inference-controls.md`](inference-controls.md).

## Concurrency

Shallow prompt (512 tokens), 512 tokens generated, so decode is measured without
prefill contention. `aggregate decode` is the sum of per-stream decode rates.

`CTX=1000000`, `COALESCE_MAX=4`:

| streams | aggregate decode tok/s | per stream | TTFT p50 |
|---:|---:|---:|---:|
| 1 | 24.4 | 24.40 | 0.7 s |
| 2 | 28.1 | 14.05 | 1.3 s |
| 4 | **45.6** | 11.40 | 2.5 s |

`CTX=262144`, `COALESCE_MAX=8`:

| streams | aggregate decode tok/s | per stream | TTFT p50 |
|---:|---:|---:|---:|
| 1 | 30.8 | 30.80 | 0.8 s |
| 2 | 30.7 | 15.35 | 1.9 s |
| 4 | 42.6 | 10.65 | 3.5 s |
| 8 | **61.9** | 7.74 | 4.5 s |

Concurrency is real headroom: **1.9x aggregate at 4 streams, 2.0x at 8**. Batching
amortises the expert-weight traffic that dominates decode. It is not free — per
stream you go from 24.4 to 11.4 tok/s at 4 streams — so this is throughput for a
fleet, not latency for one user.

**Measure concurrency at a shallow depth.** Our first attempt used 32k prompts and
showed aggregate *falling* with concurrency. That was an artifact: one stream's
decode was competing with three others' prefill. The server was genuinely
co-scheduling the whole time (`path=cont served=4`).

## Why not a smaller context

Best-of-3, 512 tokens generated, one stream:

| depth | `CTX=1000000` | `CTX=262144` | delta |
|---:|---:|---:|---:|
| 532 | 32.8 (spread 14.7) | 35.3 (spread 2.1) | +7.6% |
| 8,056 | 30.5 (spread 5.3) | 34.3 (spread 7.1) | +12.5% |

Dropping to 262k does buy something. At that size ds4 stops using the managed KV
path entirely — the line

```
ds4: CUDA using managed KV cache for ctx=1000000 (kv cache 13.17 GiB, context
     buffers 20.80 GiB); this may degrade performance but is needed for very
     large contexts
```

disappears, context buffers fall from 20.80 GiB to 5.86 GiB, and the freed ~26
GiB lets the batch lane take 8 banks instead of 4.

**We keep 1M anyway.** The gain is ~10% on single-stream decode, with spreads
wide enough that a single sample can show either sign. Against that, 262k means a
coding session that exceeds 262k tokens stops being a warm session and starts
paying the ~134 s cold-prefill penalty on every eviction. Ten percent of decode rate
is a poor trade for the ability to keep a large repository resident. If your
workload is many short concurrent chats rather than one deep session, invert this
choice: `make serve CTX=262144` with `COALESCE_MAX=8` is the throughput
configuration and it is 61.9 vs 45.6 aggregate tok/s.

## What did not help

**The 19 idle CPU cores.** The obvious suspicion on a 20-core box running at 5%
CPU is wasted parallelism. It is not:

| signal | reading |
|---|---|
| `utilization.gpu` | 96% |
| `clocks.current.sm` | 2379 MHz (max boost, sustained) |
| SW/HW power cap, thermal slowdown | Not Active |
| power draw | 82 W |
| ds4-server CPU | 97.7% of one core, 5 threads |
| graph capture | `per-layer graph capture armed (default ON)` |

The failure mode that *would* make idle cores matter is launch-bound decode — a
pegged host thread feeding a starved GPU. ds4 already closes it with per-layer
CUDA graph capture, on by default, and the GPU sits at 96% regardless. Nothing is
throttling. More fundamentally this is a **unified memory** part: the CPU cores
and the GPU share one LPDDR5X bus, and batch-1 MoE decode is bandwidth-bound.
Recruiting CPU cores would contend for the same bus, not add throughput. 93% idle
CPU next to a 96%-busy GPU is the correct shape for this workload.

**Bigger prefill chunks.** `DS4_CONT_PREFILL_CHUNK=16384` is clamped — the server
still reports `prefill_chunk=4096` and prefill throughput was unchanged within
noise.

**More speculation depth.** Already near its ceiling: 93.6% accept rate and 4.4
tokens per step against a drafter whose `block_size` is 5. Typical steady state:

```
CONT_MTP_ACCEPT(DSpark) D=4 steps=265 emit=988 drafts=856 hits=723 accept=84.5% tok/step=3.73
```

There may be ~10% left in the adaptive-depth controllers
(`DS4_DSPARK_ADAPT_GATE`), not a multiple.

## What did help, a lot

**Serving from local NVMe rather than the NAS.** This was the single largest
effect measured, and it is not subtle:

| | from NFS | from local NVMe |
|---|---|---|
| boot to listening | **9 m 40 s** | **60 s** |
| aligned artifact build | 268.7 s | 13.9 s |
| q2k repack (28.22 GiB) | 105.3 s | 4.9 s |
| deep-prompt decode | stalls, ~30 MB/s of NFS read RPCs | steady |

The base GGUF is mmapped unpinned. Over NFS, deep prefill demand-pages expert
weights across the wire, and once memory fills at 1M context the page cache
starts evicting model pages that then have to be re-fetched. `make localize
MODEL=abliterated` copies a set down at ~345 MB/s (92.79 GB in 4m16s). Do this
before benchmarking anything, or you will benchmark your network.

## Disk KV: a restart no longer costs a session

`ds4-server` can checkpoint KV state to disk and restore it into a bank instead
of re-prefilling. It is **off unless `--kv-disk-dir` is passed**, and we were not
passing it. Measured here, same 125,517-token prompt, across a **full server
restart** (new process, cold page cache):

| pass | served from cache | TTFT |
|---|---:|---:|
| cold, empty disk cache | 0 | 126,433 ms |
| after restart, restored from disk | 125,440 | **4,219 ms** |

**30x**, of which the actual checkpoint load was 659 ms:

```
kv cache bank restore hit tokens=125548 bank=0 file=...kv load=659.4 ms
bank restore admit bank=0 bytes=691490 lcp=691340 partial=1
```

The checkpoint was 549 MiB for 125k tokens — about **4.6 KiB/token on disk**
against ~35.6 KiB/token resident, because the on-disk form is packed-native.

This changes the economics of everything above. The warm-path argument for a 1M
window previously had a hole in it: warm turns cost 0.46 s, but *any* restart —
to switch models, retune the memory floor, or clear a wedge — dropped you back to
134 s. With disk KV that penalty becomes ~4 s. It is now on by default in
`start.sh` (`--no-kv-disk` opts out).

Two defaults had to be overridden. ds4's budget when enabled is 4096 MiB, which
holds less than one full-window session; we use 65536. And its cold-save ceiling
is 30000 tokens, which would skip *every* prompt this box exists to serve; we set
it to the full context.

## Further tuning: what we checked and rejected

The engine has a very large tunable surface (roughly 400 `DS4_*` variables). Most
of what looks promising is either already on or measurably wrong for this box.

**`DS4_SERVER_DEFAULT_TEMP=0` — the documentation is stale.** `misc/cuda-env-vars.md`
claims that requests omitting `temperature` get the 1.0 default and are therefore
"silently routed down the serial path — no continuous batching, no DSpark". That
was true when written (2026-07-13); it is not true on v0.5.6.1. `job_is_batchable`
(`ds4_server.c:13374`) does not test temperature at all — only the older *static*
batch path does (`job_is_static_batchable`, `ds4_server.c:13402`). Verified
empirically: requests with no temperature, with `temperature: 0`, and with tools
all took `path=cont` with speculation active. There is still a real but
second-order reason to prefer greedy — a deterministic target makes the drafter
agree more often, so the speculative accept rate rises — but it is not the lane
cliff the docs describe.

**`DS4_SERVER_COALESCE_MAX`.** ds4's own default is 32 with fit-or-reduce, and the
engine's shipped deep-context gate (`speed-bench/deep_ctx_gate.sh`) uses 8 at
`-c 524288`. We use 4 because that is what fits at 1M; the fitter confirms it
(`persistent batch ctx ready (max_seq=4 ...)`). Worth revisiting only alongside a
smaller window.

**`DS4_CONT_PREFILL_CHUNK`** is clamped to the session cap; setting 16384 leaves
the server still reporting `prefill_chunk=4096`.

**Already on, verified in our boot log** — packed FP8 KV primary, packed FP4
indexer primary, per-layer CUDA graph capture, aligned Q2_K/IQ2_XXS/Q8_0 repack
artifacts built in-process, fused MoE gate/up/swiglu. These have silent-failure
modes (the FP8/FP4 primaries print `REFUSED` if VMM demand mapping is
unavailable; a boot that misses the derived artifacts runs ~30% slower with no
log tell), so they are worth grepping for after any config change.

**Profiling, if you want to chase the 27 → 13 tok/s slope at depth**: the engine
ships `DS4_CUDA_STAGE_PROF_LITE=1` for per-stage GPU spans (valid in production),
`DS4_DSPARK_PROFILE=1` for the verify/accept/inject split, and
`speed-bench/deep_decode_profile.sh`, which nsys-profiles *only* the warm decode
window after a deep cold admit. We have not run these yet.

## The serial-lane wedge has a root cause, and it is structural

This repo already documented the wedge — the serial session right-sizes upward
and never shrinks until `usable` hits zero and every admission is rejected. The
engine source shows why, and it is worse than "it does not shrink":

`serial_session_ensure_fit` (`ds4_server.c:13567`) re-creates the session only
when `cur_ctx < need_min` (`ds4_server.c:13589`). Nothing re-creates it smaller.
Worse, when it does grow it runs a binary search for **the largest ctx that
currently fits** plus 32768 tokens of headroom (`ds4_server.c:13641-13654`) — so
each step claims whatever the box happens to have free at that instant. That is
exactly the observed 2561 → 16115 → 78924 ladder. `ds4_session_free` is called in
only two places: inside that grow path, and at teardown.

Two further findings worth knowing:

- **The deep-serial guard does not cover the path that actually grows the graph.**
  `DS4_SERVER_SERIAL_MAX_TOKENS` is only consulted in the continuous-lane
  *fallback* sweep (`ds4_server.c:15542`). Requests routed serial *by need* —
  buffered tool turns, live-frontier continuations, prefill-only — dispatch at
  `ds4_server.c:15664` without consulting it, and grow the graph freely. This is
  probably how a box with a 65536 guard reached 78924.
- **Our `SERIAL_MAX_TOKENS=$CTX` may be backwards.** The engine's comment at
  `ds4_server.c:15513` argues the opposite of this repo's README: at deep context
  the serial lane takes the managed-KV fallback at "minutes of prefill, ~40x
  slower decode", and the reject that bounced the job there is usually transient,
  so a fast 503 the client can retry is better than a silent 40x degradation.
  Setting the guard to the full context disables it. We have not changed this
  yet — it deserves its own measurement, because "fail fast" and "always succeed
  eventually" are a product decision, not just a perf one.

Mitigations exist and all have real costs. `DS4_SERVER_SERIAL_RIGHTSIZE=0`
(`ds4_server.c:17569`) stops the ratchet completely by making the serial lane
never allocate — but that lane is not optional for requests routed to it by need,
so tool-heavy workloads may start failing. `DS4_SESSION_GRAPH_HEADROOM_MB`
(default 1024, `ds4.c:10188`) raises the bar the fit probe must clear and so caps
how far the ratchet can climb, at the cost of more 503s. **We ship neither**, on
the grounds that a documented restart beats an undocumented failure mode; disk KV
now makes that restart cost ~4 s instead of a session.

## The resulting defaults

| setting | value | why |
|---|---|---|
| `CTX` | 1000000 | Warm turns cost ~0.46 s at any depth; eviction costs 134 s. Worth ~10% decode. |
| `COALESCE_MAX` | 4 | What fits at 1M. 8 banks needs the memory that 1M context is using. |
| `MAX_OUT` | 32768 | Every admission credits `prompt + max_out` of KV growth; ds4's 393216 default reserves 461k tokens for a 68k prompt. |
| `SERIAL_MAX_TOKENS` | `$CTX` | ds4's hardcoded 65536 is a cliff at 6.5% of a 1M window. |
| `MEM_FLOOR_GB` | 4 (ds4 default) | The only thing between this box and the OOM killer at 99.6% occupancy. |
| weights location | local NVMe | 9.7x faster boot, and deep prefill does not stall. |
| `--kv-disk-dir` | `~/ds4-kv`, 64 GiB | A restart costs ~4 s instead of 134 s. Off in ds4 by default. |
| `--kv-cache-cold-max-tokens` | `$CTX` | ds4's 30000 default would skip every prompt this box exists to serve. |

For a throughput-first deployment instead of a depth-first one:

```bash
COALESCE_MAX=8 make serve CTX=262144      # 61.9 vs 45.6 aggregate decode tok/s
```

# DSpark drafter repair

The abliterated weight set will not serve with speculative decode as published.
`ds4-server` aborts during load:

```
ds4-serve: DSpark speculative decode, MTP head dropped (--no-dspark / --no-spec to downgrade)
ds4: required tensor is missing: dspark.main_proj.weight
```

This is a metadata problem, not a weights problem. `tools/gguf_dspark_remap.py`
fixes it, and `make weights MODEL=abliterated` (or the first `./start.sh --model
abliterated`) runs it for you. This document is the why.

## What is wrong

`DeepSeek-V4-Flash-0731-Abliterated-DS4-Headroom128-DSpark-support.gguf` is a
correct three-stage DSpark support model exported under upstream's **legacy
`mtp.*` naming**. ds4's DSpark loader looks for `dspark.*`. The two files are
structurally identical — 81 tensors, same shapes, three stages distilled against
target layers 40/41/42 — so nothing needs to be recomputed.

Three things differ.

**1. Per-stage tensor names.** `mtp.N.<tensor>` where ds4 wants `dspark.N.<tensor>`.

**2. The shared head is folded into stages.** ds4 keeps the nine shared-head
tensors un-indexed; the legacy export hangs them off whichever stage produced
them (`main_*` on stage 0, everything else on the last stage) and nests two of
them under sub-module names:

| legacy | ds4 |
|---|---|
| `mtp.0.main_proj.weight` | `dspark.main_proj.weight` |
| `mtp.0.main_norm.weight` | `dspark.main_norm.weight` |
| `mtp.2.norm.weight` | `dspark.norm.weight` |
| `mtp.2.hc_head_{fn,base,scale}.weight` | `dspark.hc_head_{fn,base,scale}.weight` |
| `mtp.2.markov_head.markov_w{1,2}.weight` | `dspark.markov_w{1,2}.weight` |
| `mtp.2.confidence_head.proj.weight` | `dspark.conf_proj.weight` |

Note that `mtp.2.norm.weight` maps to the shared `dspark.norm.weight` while
`mtp.2.attn_norm.weight` maps to the per-stage `dspark.2.attn_norm.weight`. The
tool matches the whole remainder of the name, never a suffix, so those cannot
collide.

**3. Metadata keys.** ds4 reads `deepseek4.dspark.*`:

| legacy | ds4 |
|---|---|
| `dspark.block_size` | `deepseek4.dspark.block_size` |
| `dspark.markov_rank` | `deepseek4.dspark.markov_rank` |
| `dspark.noise_token_id` | `deepseek4.dspark.noise_token_id` |
| `dspark.target_layer_ids` (`u32[]`) | `deepseek4.dspark.target_layers` (`i32[]`) |
| `dspark.stage_count`, `dspark.n_layers` | `deepseek4.dspark.layer_count` |
| — | `deepseek4.dspark.expert_count` |

`expert_count` has no legacy counterpart. It is derived from the router width
(`ffn_gate_inp` is `[4096, 256]`, so 256 experts) rather than hardcoded.

## Why the fallback does not save you

`ds4-server` does know the `mtp.*` names — but that is the **legacy MTP module**,
a different thing. It expects `mtp.0.enorm.weight` and `mtp.0.e_proj.weight`,
which this file does not contain, and it is a single block, not three stages.
`ds4-serve` also refuses to pair any legacy MTP GGUF with an 0731-generation base
(the MTP head was replaced upstream by the DSpark stages), so `--no-dspark` on
this model drops straight to plain decode with no speculation at all — about a
3.5x loss in tokens per step.

## The dtype pass

Renaming alone is not enough. ds4 type-checks every tensor outside the routed
expert bank:

```
ds4: tensor dspark.markov_w1.weight has type q8_0, expected f16
```

The Headroom128 profile is more aggressive than the reference drafter in a few
places, so 13 tensors need widening:

| tensor | published | ds4 wants |
|---|---|---|
| `markov_w1`, `markov_w2` | Q8_0 | F16 |
| `conf_proj` | Q8_0 | F32 |
| `ffn_gate_inp` (×3 stages) | Q8_0 | F32 |
| `hc_attn_fn`, `hc_ffn_fn` (×3), `hc_head_fn` | F16 | F32 |

Every one of these is an **upcast**: Q8_0 → F32 and F16 → F32 are exact, and
Q8_0 → F16 is exact up to half-precision rounding of the dequantized product
(measured worst case 1.95e-3 absolute on `markov_w1`, well inside the Q8_0
quantization error already baked into the file). The tool refuses to go the
other way. If a future build needs a *downcast* or a requantization — say
IQ2_XXS experts where ds4 demands Q2_K — it errors out and tells you the drafter
has to be rebuilt from its FP8 parent, because inventing those weights would
silently change the model.

The routed experts are the one family ds4 type-checks loosely ("expected a
routed expert quant type"), which is what lets this drafter's IQ2_XXS gate/up
tensors ride the same kernels as the reference's Q2_K.

Cost of the widening: **+72 MiB**, 5.58 GiB → 5.65 GiB.

## Using it

```bash
make repair MODEL=abliterated     # rebuild the drafter from the published file
make verify                       # check every drafter against ds4's schema
```

Directly:

```bash
tools/gguf_dspark_remap.py IN.gguf OUT.gguf --plan     # show the map, write nothing
tools/gguf_dspark_remap.py IN.gguf OUT.gguf            # repair
tools/gguf_dspark_remap.py --verify OUT.gguf --reference STOCK-DRAFTER.gguf
```

The repaired file lands beside the published one as
`...-DSpark-support-ds4.gguf`; the original is never modified, and the weight
table points the launcher at the repaired name. `--plan` is worth running first
on any new build: it prints every rename, every KV change and every upcast
before a byte is written.

The tool is stdlib-only Python 3 — no numpy, no gguf package, nothing to install.
A repair takes about 15 seconds and is idempotent; re-running `make repair`
overwrites the output with `--force` semantics only when you ask for it, and
`ensure_drafter` skips the work entirely if the repaired file already exists.

## Verifying the result

`--verify` checks the tensor set against ds4's schema, the dtypes against what
the loader accepts, and the data section for alignment, overlap and truncation.
With `--reference` it also diffs names and shapes against a known-good drafter:

```
$ make verify
== abliterated
arch        : deepseek4-dspark
stages      : 3
target      : [40, 41, 42]
experts     : 256
tensors     : 81
experts     : 9 tensors, IQ2_XXS/Q2_K (ds4 accepts any routed expert quant)
verify      : OK
```

The payload was also checked tensor-by-tensor against the source on the first
build of this file: 81/81 SHA-256 matches for the copied tensors, and the
upcast tensors compared against an independent dequantization.

The real proof is in the log:

```
ds4: DSpark drafter loaded: ...-DSpark-support-ds4.gguf (3 layers)
ds4: CONT_MTP_ACCEPT(DSpark) D=4 steps=265 emit=988 drafts=856 hits=723 accept=84.5% tok/step=3.73
```

`accept=` in the 80% range and `tok/step` near 3.5–3.8 means speculation is
working. If you see neither line, the drafter did not arm.

## Reporting it upstream

The correct long-term fix is in the exporter, not here: the DS4-Headroom128
profile in `deepseek-model-tools` writes DSpark stages under the legacy `mtp.*`
prefix and the legacy KV keys, so every rebuild of that profile will need this
repair. Anyone republishing that model card should ship the `dspark.*` layout
directly.

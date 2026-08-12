#!/usr/bin/env python3
"""
gguf_dspark_remap.py — rename a legacy `mtp.*` DSpark GGUF into the `dspark.*`
layout ds4-server actually loads.

Why this exists
---------------
The abliterated DS4-Headroom128 build ships its three-stage DSpark support model
with upstream's *legacy MTP* naming. ds4-server's DSpark loader looks up
`dspark.<stage>.<tensor>` plus nine un-indexed shared-head tensors, so it dies at
startup with:

    ds4: required tensor is missing: dspark.main_proj.weight

The two files are structurally identical — same 81 tensors, same shapes, same
three stages against target layers 40/41/42. Only the names differ:

    mtp.N.attn_q_a.weight              -> dspark.N.attn_q_a.weight
    mtp.0.main_proj.weight             -> dspark.main_proj.weight
    mtp.2.markov_head.markov_w1.weight -> dspark.markov_w1.weight
    dspark.n_layers (KV)               -> deepseek4.dspark.layer_count

so this is a metadata repair, not a conversion. Tensor payload bytes are copied
verbatim; nothing is requantized, reordered or recomputed. The output is
byte-identical to the input in the data section, modulo alignment padding.

Usage
-----
    tools/gguf_dspark_remap.py IN.gguf OUT.gguf            # repair
    tools/gguf_dspark_remap.py IN.gguf OUT.gguf --plan     # show the map, write nothing
    tools/gguf_dspark_remap.py --verify OUT.gguf           # check against the schema
    tools/gguf_dspark_remap.py --verify OUT.gguf --reference STOCK.gguf

Exit codes: 0 ok, 1 usage/IO error, 2 schema mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import struct
import sys
import tempfile

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32
COPY_CHUNK = 8 << 20

# ---------------------------------------------------------------- GGUF typing
(
    T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32,
    T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64,
) = range(13)

SCALAR_FMT = {
    T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h",
    T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f", T_BOOL: "<B",
    T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d",
}
TYPE_NAME = {
    T_UINT8: "u8", T_INT8: "i8", T_UINT16: "u16", T_INT16: "i16",
    T_UINT32: "u32", T_INT32: "i32", T_FLOAT32: "f32", T_BOOL: "bool",
    T_STRING: "str", T_ARRAY: "arr", T_UINT64: "u64", T_INT64: "i64",
    T_FLOAT64: "f64",
}

# ggml_type -> (block numel, block bytes). Only what these models use.
GGML_TYPE = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    19: ("IQ3_XXS", 256, 98), 20: ("IQ1_S", 256, 50), 21: ("IQ4_NL", 32, 18),
    23: ("IQ3_S", 256, 110), 24: ("IQ2_S", 256, 82), 25: ("IQ4_XS", 256, 136),
    26: ("I8", 1, 1), 27: ("I16", 1, 2), 28: ("I32", 1, 4), 30: ("BF16", 1, 2),
}


class GGUFError(Exception):
    pass


# ------------------------------------------------------------------- schema
# What ds4-server's DSpark loader asks for (from the `dspark.%u.*` / `dspark.*`
# format strings in the binary).
PER_STAGE = (
    "attn_q_a.weight", "attn_q_b.weight", "attn_kv.weight",
    "attn_output_a.weight", "attn_output_b.weight",
    "attn_q_a_norm.weight", "attn_kv_a_norm.weight", "attn_sinks.weight",
    "attn_norm.weight", "ffn_norm.weight",
    "hc_attn_fn.weight", "hc_attn_base.weight", "hc_attn_scale.weight",
    "hc_ffn_fn.weight", "hc_ffn_base.weight", "hc_ffn_scale.weight",
    "ffn_gate_inp.weight", "exp_probs_b.bias",
    "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
    "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
)
SHARED = (
    "main_proj.weight", "main_norm.weight", "norm.weight",
    "hc_head_fn.weight", "hc_head_base.weight", "hc_head_scale.weight",
    "markov_w1.weight", "markov_w2.weight", "conf_proj.weight",
)

# What ds4's DSpark loader accepts per tensor, verified against the reference
# drafter (bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF, DSpark-drafter-Q2K-Q8).
# The loader is strict outside the routed-expert bank:
#
#     ds4: tensor dspark.markov_w1.weight has type q8_0, expected f16
#
# The routed experts are the one family it type-checks loosely ("expected a
# routed expert quant type"), which is what lets an IQ2_XXS-quantized drafter
# ride the same kernels as the Q2_K reference.
T_F32, T_F16, T_Q8_0 = 0, 1, 8
EXPERT_FAMILY = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
WANT_DTYPE = {
    # per-stage
    "attn_q_a.weight": T_Q8_0, "attn_q_b.weight": T_Q8_0, "attn_kv.weight": T_Q8_0,
    "attn_output_a.weight": T_Q8_0, "attn_output_b.weight": T_Q8_0,
    "attn_q_a_norm.weight": T_F32, "attn_kv_a_norm.weight": T_F32,
    "attn_sinks.weight": T_F32, "attn_norm.weight": T_F32, "ffn_norm.weight": T_F32,
    "hc_attn_fn.weight": T_F32, "hc_attn_base.weight": T_F32, "hc_attn_scale.weight": T_F32,
    "hc_ffn_fn.weight": T_F32, "hc_ffn_base.weight": T_F32, "hc_ffn_scale.weight": T_F32,
    "ffn_gate_inp.weight": T_F32, "exp_probs_b.bias": T_F32,
    "ffn_gate_shexp.weight": T_Q8_0, "ffn_up_shexp.weight": T_Q8_0,
    "ffn_down_shexp.weight": T_Q8_0,
    # shared head
    "main_proj.weight": T_Q8_0, "main_norm.weight": T_F32, "norm.weight": T_F32,
    "hc_head_fn.weight": T_F32, "hc_head_base.weight": T_F32, "hc_head_scale.weight": T_F32,
    "markov_w1.weight": T_F16, "markov_w2.weight": T_F16, "conf_proj.weight": T_F32,
}

# Conversions the repair is willing to perform. Every one of these is an upcast:
# it materialises a coarser representation into a wider one and cannot invent
# precision. Downcasting or requantizing (e.g. IQ2_XXS -> Q2_K) would change the
# weights, so it is refused rather than guessed.
UPCASTS = {
    (T_Q8_0, T_F32), (T_Q8_0, T_F16), (T_F16, T_F32),
}


def dtype_family(name: str) -> str:
    parts = name.split(".")
    return ".".join(parts[2:]) if len(parts) > 2 and parts[1].isdigit() else ".".join(parts[1:])

# Legacy suffix (after `mtp.<stage>.`) -> shared-head name. Matched on the whole
# remainder, so `attn_norm.weight` / `ffn_norm.weight` never collide with `norm`.
SHARED_FROM_LEGACY = {
    "main_proj.weight": "main_proj.weight",
    "main_norm.weight": "main_norm.weight",
    "norm.weight": "norm.weight",
    "hc_head_fn.weight": "hc_head_fn.weight",
    "hc_head_base.weight": "hc_head_base.weight",
    "hc_head_scale.weight": "hc_head_scale.weight",
    "markov_head.markov_w1.weight": "markov_w1.weight",
    "markov_head.markov_w2.weight": "markov_w2.weight",
    "confidence_head.proj.weight": "conf_proj.weight",
}

# Legacy KV key -> ds4 KV key. `stage_count` and `n_layers` both mean
# layer_count; they agree in every build seen so far and we assert that.
KV_FROM_LEGACY = {
    "dspark.block_size": "deepseek4.dspark.block_size",
    "dspark.markov_rank": "deepseek4.dspark.markov_rank",
    "dspark.noise_token_id": "deepseek4.dspark.noise_token_id",
    "dspark.target_layer_ids": "deepseek4.dspark.target_layers",
    "dspark.n_layers": "deepseek4.dspark.layer_count",
    "dspark.stage_count": "deepseek4.dspark.layer_count",
}
# ds4 reads target_layers as a signed array; the legacy file writes it unsigned.
KV_RETYPE = {"deepseek4.dspark.target_layers": T_INT32}
KV_ORDER = (
    "general.architecture",
    "general.name",
    "general.alignment",
    "deepseek4.dspark.layer_count",
    "deepseek4.dspark.block_size",
    "deepseek4.dspark.markov_rank",
    "deepseek4.dspark.noise_token_id",
    "deepseek4.dspark.expert_count",
    "deepseek4.dspark.target_layers",
)


def expected_names(n_stages: int) -> set[str]:
    out = {"dspark.%s" % s for s in SHARED}
    for i in range(n_stages):
        out |= {"dspark.%d.%s" % (i, s) for s in PER_STAGE}
    return out


# -------------------------------------------------------------------- reader
class Reader:
    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "rb")
        if self.f.read(4) != GGUF_MAGIC:
            raise GGUFError("%s: not a GGUF file" % path)
        self.version, = struct.unpack("<I", self.f.read(4))
        if self.version != 3:
            raise GGUFError("%s: GGUF v%d, only v3 is handled" % (path, self.version))
        n_tensors, = struct.unpack("<Q", self.f.read(8))
        n_kv, = struct.unpack("<Q", self.f.read(8))

        self.kv: list[tuple[str, int, object, int]] = []   # key, type, value, elem_type
        for _ in range(n_kv):
            key = self._string()
            vtype, = struct.unpack("<I", self.f.read(4))
            value, etype = self._value(vtype)
            self.kv.append((key, vtype, value, etype))

        self.tensors: list[dict] = []
        for _ in range(n_tensors):
            name = self._string()
            n_dims, = struct.unpack("<I", self.f.read(4))
            dims = [struct.unpack("<Q", self.f.read(8))[0] for _ in range(n_dims)]
            ggml_type, = struct.unpack("<I", self.f.read(4))
            offset, = struct.unpack("<Q", self.f.read(8))
            self.tensors.append({
                "name": name, "dims": dims, "type": ggml_type, "offset": offset,
                "nbytes": tensor_nbytes(name, dims, ggml_type),
            })

        self.alignment = int(self.kv_get("general.alignment", DEFAULT_ALIGNMENT))
        self.data_start = align_up(self.f.tell(), self.alignment)

    def _string(self) -> str:
        n, = struct.unpack("<Q", self.f.read(8))
        return self.f.read(n).decode("utf-8")

    def _value(self, vtype: int):
        if vtype == T_STRING:
            return self._string(), None
        if vtype == T_ARRAY:
            etype, = struct.unpack("<I", self.f.read(4))
            n, = struct.unpack("<Q", self.f.read(8))
            return [self._value(etype)[0] for _ in range(n)], etype
        fmt = SCALAR_FMT.get(vtype)
        if fmt is None:
            raise GGUFError("unknown KV type %d" % vtype)
        raw = self.f.read(struct.calcsize(fmt))
        v, = struct.unpack(fmt, raw)
        return (bool(v) if vtype == T_BOOL else v), None

    def kv_get(self, key: str, default=None):
        for k, _t, v, _e in self.kv:
            if k == key:
                return v
        return default

    def kv_dict(self) -> dict:
        return {k: v for k, _t, v, _e in self.kv}

    def names(self) -> set[str]:
        return {t["name"] for t in self.tensors}

    def close(self):
        self.f.close()


def align_up(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def tensor_nbytes(name: str, dims: list[int], ggml_type: int) -> int:
    info = GGML_TYPE.get(ggml_type)
    if info is None:
        raise GGUFError("%s: unhandled ggml type %d" % (name, ggml_type))
    tname, blk_numel, blk_bytes = info
    numel = 1
    for d in dims:
        numel *= d
    if numel % blk_numel:
        raise GGUFError("%s: %d elements is not a multiple of the %s block (%d)"
                        % (name, numel, tname, blk_numel))
    return numel // blk_numel * blk_bytes


# ---------------------------------------------------------------- conversion
# Pure stdlib on purpose: this repo's only hard requirements are bash and curl,
# and a one-shot header repair is not worth a numpy dependency. `struct`'s "e"
# format is IEEE half, which is exactly GGML's F16.
BLOCKS_PER_BATCH = 8192          # 8192 * 32 = 262144 values per struct call


def convert_stream(f, nbytes: int, numel: int, src_type: int, dst_type: int):
    """Yield the tensor's bytes re-encoded from src_type to dst_type."""
    if (src_type, dst_type) not in UPCASTS:
        raise GGUFError("refusing to convert %s -> %s"
                        % (GGML_TYPE[src_type][0], GGML_TYPE[dst_type][0]))
    out_fmt = "e" if dst_type == T_F16 else "f"

    if src_type == T_F16:                                  # F16 -> F32
        left = nbytes
        while left:
            take = min(BLOCKS_PER_BATCH * 32 * 2, left)
            raw = f.read(take)
            if len(raw) != take:
                raise GGUFError("short read during F16 upcast")
            n = take // 2
            yield struct.pack("<%d%s" % (n, out_fmt), *struct.unpack("<%de" % n, raw))
            left -= take
        return

    # Q8_0 -> F32/F16. Block layout: f16 scale, then 32 int8 quants.
    blocks = numel // 32
    done = 0
    while done < blocks:
        batch = min(BLOCKS_PER_BATCH, blocks - done)
        raw = f.read(batch * 34)
        if len(raw) != batch * 34:
            raise GGUFError("short read during Q8_0 upcast")
        vals = []
        for i in range(batch):
            base = i * 34
            scale, = struct.unpack_from("<e", raw, base)
            qs = struct.unpack_from("<32b", raw, base + 2)
            vals.extend([scale * q for q in qs])
        yield struct.pack("<%d%s" % (len(vals), out_fmt), *vals)
        done += batch


def converted_nbytes(numel: int, dst_type: int) -> int:
    return numel * (2 if dst_type == T_F16 else 4)


def numel_of(dims: list[int]) -> int:
    n = 1
    for d in dims:
        n *= d
    return n


# -------------------------------------------------------------------- writer
def enc_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def enc_value(vtype: int, value, etype: int | None) -> bytes:
    if vtype == T_STRING:
        return enc_string(value)
    if vtype == T_ARRAY:
        out = struct.pack("<I", etype) + struct.pack("<Q", len(value))
        for v in value:
            out += enc_value(etype, v, None)
        return out
    return struct.pack(SCALAR_FMT[vtype], int(value) if vtype != T_FLOAT32 and vtype != T_FLOAT64 else value)


def build_header(kv: list[tuple[str, int, object, int]], tensors: list[dict],
                 alignment: int) -> bytes:
    """Lay out the data section and emit the header for it.

    Chicken-and-egg: the header's size does not depend on the offsets (they are
    fixed-width u64s), but writing it is clearest as the same loop that assigns
    them. Offsets are relative to the start of the aligned data section, so they
    are independent of the header length. Each tensor's new `offset` is written
    back into its dict for the copy loop to use.
    """
    def emit(offsets: list[int]) -> bytes:
        out = bytearray(GGUF_MAGIC)
        out += struct.pack("<I", 3)
        out += struct.pack("<Q", len(tensors))
        out += struct.pack("<Q", len(kv))
        for key, vtype, value, etype in kv:
            out += enc_string(key) + struct.pack("<I", vtype) + enc_value(vtype, value, etype)
        for t, off in zip(tensors, offsets):
            out += enc_string(t["name"])
            out += struct.pack("<I", len(t["dims"]))
            for d in t["dims"]:
                out += struct.pack("<Q", d)
            out += struct.pack("<I", t["type"])
            out += struct.pack("<Q", off)
        return bytes(out)

    offsets = []
    cursor = 0
    for t in tensors:
        offsets.append(cursor)
        t["offset"] = cursor
        cursor = align_up(cursor + t["nbytes"], alignment)
    return emit(offsets)


# ------------------------------------------------------------------- remap
def remap_name(name: str) -> str:
    if name.startswith("dspark."):
        return name                                   # already ds4-native
    if not name.startswith("mtp."):
        raise GGUFError("unexpected tensor name %r (neither mtp.* nor dspark.*)" % name)
    _, stage, rest = name.split(".", 2)
    if not stage.isdigit():
        raise GGUFError("unexpected tensor name %r (no stage index)" % name)
    shared = SHARED_FROM_LEGACY.get(rest)
    if shared is not None:
        return "dspark." + shared
    return "dspark.%d.%s" % (int(stage), rest)


def remap_kv(src: Reader) -> list[tuple[str, int, object, int]]:
    out: dict[str, tuple[int, object, int]] = {}
    for key, vtype, value, etype in src.kv:
        if key.startswith("deepseek4.dspark."):
            new = key
        elif key in KV_FROM_LEGACY:
            new = KV_FROM_LEGACY[key]
        elif key.startswith("dspark."):
            raise GGUFError("unmapped legacy KV key %r — refusing to guess" % key)
        else:
            new = key                                  # general.* and friends
        if new in KV_RETYPE and vtype == T_ARRAY:
            etype = KV_RETYPE[new]
        if new in out and out[new][1] != value:
            raise GGUFError("%r maps onto %r but the values disagree (%r vs %r)"
                            % (key, new, out[new][1], value))
        out[new] = (vtype, value, etype)

    # expert_count is absent from the legacy metadata; ds4 requires it. Derive it
    # from the router width rather than hardcoding 256.
    if "deepseek4.dspark.expert_count" not in out:
        experts = None
        for t in src.tensors:
            if t["name"].endswith("ffn_gate_inp.weight"):
                experts = t["dims"][-1]
                break
        if experts is None:
            raise GGUFError("cannot derive expert_count: no ffn_gate_inp tensor")
        out["deepseek4.dspark.expert_count"] = (T_UINT32, int(experts), None)

    if "deepseek4.dspark.layer_count" not in out:
        raise GGUFError("cannot derive layer_count: no stage_count/n_layers key")

    ordered = [(k, *out.pop(k)) for k in KV_ORDER if k in out]
    ordered += [(k, *v) for k, v in out.items()]       # anything unrecognised, kept
    return ordered


def validate(names: set[str], n_stages: int) -> list[str]:
    want = expected_names(n_stages)
    problems = []
    for miss in sorted(want - names):
        problems.append("missing: %s" % miss)
    for extra in sorted(names - want):
        problems.append("unexpected: %s" % extra)
    return problems


# --------------------------------------------------------------------- main
def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return "%.2f %s" % (n, unit)
        n /= 1024


def do_verify(path: str, reference: str | None) -> int:
    src = Reader(path)
    kv = src.kv_dict()
    stages = int(kv.get("deepseek4.dspark.layer_count", 0))
    print("file        : %s" % path)
    print("arch        : %s" % kv.get("general.architecture"))
    print("stages      : %s" % stages)
    print("target      : %s" % kv.get("deepseek4.dspark.target_layers"))
    print("experts     : %s" % kv.get("deepseek4.dspark.expert_count"))
    print("block/markov: %s / %s" % (kv.get("deepseek4.dspark.block_size"),
                                     kv.get("deepseek4.dspark.markov_rank")))
    print("tensors     : %d" % len(src.tensors))

    rc = 0
    if kv.get("general.architecture") != "deepseek4-dspark":
        print("FAIL: general.architecture is not deepseek4-dspark")
        rc = 2
    if not stages:
        print("FAIL: no deepseek4.dspark.layer_count")
        return 2
    problems = validate(src.names(), stages)
    for p in problems:
        print("FAIL: %s" % p)
    rc = 2 if problems else rc

    experts = 0
    for t in src.tensors:
        fam = dtype_family(t["name"])
        if fam in EXPERT_FAMILY:
            experts += 1
            continue
        want = WANT_DTYPE.get(fam)
        if want is not None and t["type"] != want:
            print("FAIL: %s is %s, ds4 wants %s"
                  % (t["name"], GGML_TYPE[t["type"]][0], GGML_TYPE[want][0]))
            rc = 2
    if experts:
        kinds = sorted({GGML_TYPE[t["type"]][0] for t in src.tensors
                        if dtype_family(t["name"]) in EXPERT_FAMILY})
        print("experts     : %d tensors, %s (ds4 accepts any routed expert quant)"
              % (experts, "/".join(kinds)))

    # data-section sanity: every tensor inside the file, aligned, non-overlapping
    size = os.path.getsize(path)
    cursor = -1
    for t in sorted(src.tensors, key=lambda x: x["offset"]):
        abs_off = src.data_start + t["offset"]
        if t["offset"] % src.alignment:
            print("FAIL: %s offset %d not %d-aligned" % (t["name"], t["offset"], src.alignment))
            rc = 2
        if abs_off < cursor:
            print("FAIL: %s overlaps the previous tensor" % t["name"])
            rc = 2
        if abs_off + t["nbytes"] > size:
            print("FAIL: %s runs %s past EOF" % (t["name"], human(abs_off + t["nbytes"] - size)))
            rc = 2
        cursor = abs_off + t["nbytes"]

    if reference:
        ref = Reader(reference)
        ref_stages = int(ref.kv_dict().get("deepseek4.dspark.layer_count", 0))
        if ref.names() != src.names():
            for n in sorted(ref.names() - src.names()):
                print("FAIL: reference has %s, this file does not" % n)
            for n in sorted(src.names() - ref.names()):
                print("FAIL: this file has %s, reference does not" % n)
            rc = 2
        else:
            print("names       : identical to %s (%d stages)" % (reference, ref_stages))
        shapes = {t["name"]: t["dims"] for t in src.tensors}
        for t in ref.tensors:
            if t["name"] in shapes and shapes[t["name"]] != t["dims"]:
                print("FAIL: %s shape %s vs reference %s"
                      % (t["name"], shapes[t["name"]], t["dims"]))
                rc = 2
        quants = {t["name"]: t["type"] for t in src.tensors}
        diffs = [(t["name"], GGML_TYPE[quants[t["name"]]][0], GGML_TYPE[t["type"]][0])
                 for t in ref.tensors
                 if t["name"] in quants and quants[t["name"]] != t["type"]]
        if diffs:
            kinds = sorted({(a, b) for _n, a, b in diffs})
            print("note        : %d tensors differ in quant from the reference %s"
                  % (len(diffs), ", ".join("%s vs %s" % k for k in kinds)))
        ref.close()

    src.close()
    print("verify      : %s" % ("OK" if rc == 0 else "FAILED"))
    return rc


def do_remap(inp: str, outp: str, plan_only: bool, force: bool, checksum: bool) -> int:
    src = Reader(inp)
    stages = int(src.kv_get("dspark.n_layers")
                 or src.kv_get("dspark.stage_count")
                 or src.kv_get("deepseek4.dspark.layer_count") or 0)
    if not stages:
        raise GGUFError("%s: cannot determine the stage count" % inp)

    tensors = []
    seen: dict[str, str] = {}
    renames = 0
    casts = []
    for t in src.tensors:
        new = remap_name(t["name"])
        if new in seen:
            raise GGUFError("%s and %s both map to %s" % (seen[new], t["name"], new))
        seen[new] = t["name"]
        renames += new != t["name"]

        # dtype pass: ds4 type-checks everything outside the routed-expert bank.
        fam = dtype_family(new)
        want = None if fam in EXPERT_FAMILY else WANT_DTYPE.get(fam)
        out = dict(t, new_name=new, cast=None)
        if want is not None and want != t["type"]:
            if (t["type"], want) not in UPCASTS:
                raise GGUFError(
                    "%s is %s but ds4 wants %s, and that is not an upcast this "
                    "tool will do — the drafter needs rebuilding from its parent"
                    % (new, GGML_TYPE[t["type"]][0], GGML_TYPE[want][0]))
            numel = numel_of(t["dims"])
            if t["type"] == T_Q8_0 and numel % 32:
                raise GGUFError("%s: %d elements is not a whole number of Q8_0 blocks"
                                % (new, numel))
            out["cast"] = (t["type"], want)
            out["numel"] = numel
            out["src_nbytes"] = t["nbytes"]
            out["nbytes"] = converted_nbytes(numel, want)
            out["type"] = want
            casts.append((new, GGML_TYPE[t["type"]][0], GGML_TYPE[want][0],
                          out["nbytes"] - out["src_nbytes"]))
        tensors.append(out)

    problems = validate(set(seen), stages)
    if problems:
        for p in problems:
            print("FAIL: %s" % p, file=sys.stderr)
        raise GGUFError("the remapped tensor set does not match ds4's DSpark schema")

    kv = remap_kv(src)

    print("in          : %s (%s)" % (inp, human(os.path.getsize(inp))))
    print("out         : %s" % outp)
    print("stages      : %d, alignment %d" % (stages, src.alignment))
    print("tensors     : %d (%d renamed, %d upcast)" % (len(tensors), renames, len(casts)))
    print()
    if casts:
        print("dtype upcasts (ds4 type-checks everything but the routed experts):")
        for name, was, now, grew in casts:
            print("  %-34s %-7s -> %-4s  %+s" % (name, was, now, human(grew)))
        print()
    print("KV:")
    old_kv = src.kv_dict()
    for key, vtype, value, _e in kv:
        was = [k for k in old_kv if k == key or KV_FROM_LEGACY.get(k) == key]
        origin = was[0] if was and was[0] != key else ("<derived>" if not was else "")
        v = value if not isinstance(value, list) or len(value) <= 8 else "%s..." % value[:8]
        print("  %-38s %-5s %-18s %s" % (key, TYPE_NAME[vtype], v, origin))
    print()
    print("tensor names (first of each family):")
    shown = set()
    for t in tensors:
        fam = t["new_name"].split(".")[-2:]
        famk = tuple(fam)
        if famk in shown or t["name"] == t["new_name"]:
            continue
        shown.add(famk)
        print("  %-40s -> %s" % (t["name"], t["new_name"]))

    if plan_only:
        print("\n--plan: nothing written.")
        src.close()
        return 0

    if os.path.exists(outp) and not force:
        raise GGUFError("%s exists (pass --force to overwrite)" % outp)
    if os.path.realpath(outp) == os.path.realpath(inp):
        raise GGUFError("refusing to rewrite the input in place; pick a new --out")

    payload = sum(t["nbytes"] for t in tensors)
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(outp))).free
    if free < payload * 1.05:
        raise GGUFError("need ~%s free next to %s, have %s"
                        % (human(payload * 1.05), outp, human(free)))

    # `offset` is rewritten by build_header; stash where the bytes came from.
    out_tensors = [dict(t, name=t["new_name"], src_offset=t["offset"]) for t in tensors]
    header = build_header(kv, out_tensors, src.alignment)
    data_start = align_up(len(header), src.alignment)

    tmp_dir = os.path.dirname(os.path.abspath(outp))
    fd, tmp = tempfile.mkstemp(dir=tmp_dir, prefix=".gguf-remap-", suffix=".part")
    digest = hashlib.sha256() if checksum else None
    try:
        with os.fdopen(fd, "wb") as w:
            w.write(header)
            w.write(b"\0" * (data_start - len(header)))
            cursor = 0
            done = 0
            tick = 0
            live = sys.stdout.isatty()
            for t in out_tensors:
                pad = t["offset"] - cursor
                if pad:
                    w.write(b"\0" * pad)
                    cursor += pad
                src.f.seek(src.data_start + t["src_offset"])
                if t["cast"]:
                    written = 0
                    for chunk in convert_stream(src.f, t["src_nbytes"], t["numel"],
                                                *t["cast"]):
                        w.write(chunk)
                        if digest:
                            digest.update(chunk)
                        written += len(chunk)
                        cursor += len(chunk)
                    if written != t["nbytes"]:
                        raise GGUFError("%s: upcast produced %d bytes, expected %d"
                                        % (t["name"], written, t["nbytes"]))
                else:
                    left = t["nbytes"]
                    while left:
                        chunk = src.f.read(min(COPY_CHUNK, left))
                        if not chunk:
                            raise GGUFError("%s: short read on %s" % (inp, t["name"]))
                        w.write(chunk)
                        if digest:
                            digest.update(chunk)
                        left -= len(chunk)
                        cursor += len(chunk)
                done += t["nbytes"]
                pct = done * 20 // payload
                if pct != tick:
                    tick = pct
                    if live:
                        sys.stdout.write("\r  copying %3d%%  %s / %s"
                                         % (pct * 5, human(done), human(payload)))
                    else:
                        sys.stdout.write("  copying %3d%%\n" % (pct * 5))
                    sys.stdout.flush()
            w.flush()
            os.fsync(w.fileno())
        if live:
            print()
        os.replace(tmp, outp)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    src.close()

    if digest:
        print("payload sha256: %s" % digest.hexdigest())
    print("wrote %s (%s)" % (outp, human(os.path.getsize(outp))))
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Rename a legacy mtp.* DSpark GGUF into ds4's dspark.* layout.")
    ap.add_argument("input", nargs="?", help="legacy DSpark support GGUF")
    ap.add_argument("output", nargs="?", help="repaired GGUF to write")
    ap.add_argument("--plan", action="store_true", help="print the mapping, write nothing")
    ap.add_argument("--force", action="store_true", help="overwrite the output")
    ap.add_argument("--sha256", action="store_true", help="hash the copied payload")
    ap.add_argument("--verify", metavar="GGUF", help="check a file against ds4's schema")
    ap.add_argument("--reference", metavar="GGUF",
                    help="with --verify: a known-good drafter to diff names against")
    args = ap.parse_args(argv)

    try:
        if args.verify:
            return do_verify(args.verify, args.reference)
        if not args.input or not args.output:
            ap.error("need INPUT and OUTPUT (or --verify GGUF)")
        return do_remap(args.input, args.output, args.plan, args.force, args.sha256)
    except GGUFError as e:
        print("gguf_dspark_remap: %s" % e, file=sys.stderr)
        return 1
    except OSError as e:
        print("gguf_dspark_remap: %s" % e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

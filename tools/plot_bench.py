#!/usr/bin/env python3
"""
plot_bench.py — render the benchmark figure from recorded runs.

Reads the JSONL rows tools/bench.py writes and draws the depth curve: decode
tok/s and prefill tok/s against measured context depth, plus the cold-vs-warm
prefill comparison that justifies serving a 1M window. Nothing is hand-entered;
if the numbers in the image disagree with bench/results/ then this script was not
re-run.

Usage:
  tools/plot_bench.py                                   # defaults below
  tools/plot_bench.py --results bench/results --out bench.png
  tools/plot_bench.py --model abliterated --title "..."

Requires Pillow only.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("plot_bench: needs Pillow (python3 -m pip install --user pillow)")

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
INK = (24, 28, 34)
MUTED = (120, 130, 142)
GRID = (228, 232, 238)
BG = (255, 255, 255)
PANEL = (250, 251, 252)
DECODE = (37, 99, 235)
PREFILL = (217, 119, 6)
WARM = (16, 152, 106)
COLD = (203, 63, 63)


def font(name: str, size: int):
    try:
        return ImageFont.truetype(os.path.join(FONT_DIR, name), size)
    except OSError:
        return ImageFont.load_default()


F_TITLE = font("DejaVuSans-Bold.ttf", 30)
F_SUB = font("DejaVuSans.ttf", 17)
F_AXIS = font("DejaVuSans-Bold.ttf", 15)
F_TICK = font("DejaVuSansMono.ttf", 13)
F_NOTE = font("DejaVuSans.ttf", 13)
F_VAL = font("DejaVuSansMono.ttf", 13)


def load(results_dir: str, model: str | None) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if model and model not in str(r.get("model", "")):
                    continue
                rows.append(r)
    return rows


def human_tokens(n: float) -> str:
    if n >= 1000000:
        return "%.1fM" % (n / 1000000.0)
    if n >= 1000:
        return "%dk" % round(n / 1000.0)
    return "%d" % n


def nice_step(rough: float) -> float:
    """Round a rough tick interval up to a human one (1, 2, 2.5 or 5 x 10^n)."""
    if rough <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(rough))
    for mult in (1, 2, 2.5, 5, 10):
        if rough <= mult * mag:
            return mult * mag
    return 10 * mag


class Axes:
    """Minimal log-x / linear-y plotting frame."""

    def __init__(self, draw, box, xs, ys, ylabel, colour, headroom=1.25):
        self.d = draw
        self.x0, self.y0, self.x1, self.y1 = box
        self.lx0 = math.log10(min(xs))
        self.lx1 = math.log10(max(xs))
        steps = 4
        step = nice_step(max(ys) * headroom / steps)
        self.ymax = step * steps
        self.colour = colour
        draw.rectangle(box, fill=PANEL)
        for i in range(steps + 1):
            v = step * i
            y = self.py(v)
            draw.line([(self.x0, y), (self.x1, y)], fill=GRID, width=1)
            label = "%d" % round(v) if v >= 10 or v == 0 else "%.1f" % v
            w = draw.textlength(label, font=F_TICK)
            draw.text((self.x0 - w - 10, y - 8), label, font=F_TICK, fill=MUTED)
        draw.line([(self.x0, self.y0), (self.x0, self.y1)], fill=MUTED, width=1)
        draw.line([(self.x0, self.y1), (self.x1, self.y1)], fill=MUTED, width=1)
        # rotated-ish y label, drawn horizontally above the axis for legibility
        draw.text((self.x0 - 62, self.y0 - 30), ylabel, font=F_AXIS, fill=colour)

    def px(self, x):
        t = (math.log10(x) - self.lx0) / max(1e-9, self.lx1 - self.lx0)
        return self.x0 + t * (self.x1 - self.x0)

    def py(self, y):
        t = y / self.ymax
        return self.y1 - t * (self.y1 - self.y0)

    def series(self, xs, ys, colour, annotate=True):
        pts = [(self.px(x), self.py(y)) for x, y in zip(xs, ys)]
        for i in range(len(pts) - 1):
            self.d.line([pts[i], pts[i + 1]], fill=colour, width=4)
        for (x, y), val in zip(pts, ys):
            self.d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=BG, outline=colour, width=3)
            if annotate:
                label = "%.1f" % val if val < 100 else "%d" % round(val)
                w = self.d.textlength(label, font=F_VAL)
                self.d.text((x - w / 2, y - 26), label, font=F_VAL, fill=colour)

    def xticks(self, xs):
        for x in xs:
            px = self.px(x)
            self.d.line([(px, self.y1), (px, self.y1 + 5)], fill=MUTED, width=1)
            label = human_tokens(x)
            w = self.d.textlength(label, font=F_TICK)
            self.d.text((px - w / 2, self.y1 + 9), label, font=F_TICK, fill=MUTED)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="bench/results")
    ap.add_argument("--out", default="bench.png")
    ap.add_argument("--model", default="abliterated")
    ap.add_argument("--title", default="DeepSeek-V4-Flash on one DGX Spark")
    ap.add_argument("--subtitle", default=None)
    args = ap.parse_args(argv)

    rows = load(args.results, args.model)
    depth = sorted([r for r in rows if r.get("suite") == "depth" and r.get("decode_tok_s")],
                   key=lambda r: r["prompt_tokens"])
    # One point per measured depth; later runs win.
    by_depth = {}
    for r in depth:
        by_depth[round(r["prompt_tokens"], -2)] = r
    depth = sorted(by_depth.values(), key=lambda r: r["prompt_tokens"])
    if not depth:
        sys.exit("plot_bench: no depth rows in %s for model %r" % (args.results, args.model))

    cache = [r for r in rows if r.get("suite") == "cache"]
    cold = next((r for r in cache if r.get("pass_") == "cold"), None)
    warm = next((r for r in cache if r.get("pass_") == "warm"), None)
    # Prefer the median over repeats when the suite recorded one; a single
    # cold/warm pair moves by tens of percent run to run.
    speedup_median = warm.get("speedup_median") if warm else None

    xs = [r["prompt_tokens"] for r in depth]
    dec = [r["decode_tok_s"] for r in depth]
    pre = [r.get("prefill_tok_s") or 0 for r in depth]

    W, H = 1400, 1010
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.text((70, 44), args.title, font=F_TITLE, fill=INK)
    sub = args.subtitle or (
        "%s weights, 1M context, DSpark speculative decode, one stream, weights on local NVMe"
        % (args.model,))
    d.text((70, 86), sub, font=F_SUB, fill=MUTED)

    a1 = Axes(d, (140, 150, W - 70, 470), xs, dec, "decode tok/s", DECODE)
    a1.series(xs, dec, DECODE)
    a1.xticks(xs)

    a2 = Axes(d, (140, 560, W - 70, 830), xs, pre, "prefill tok/s", PREFILL,
              headroom=1.15)
    a2.series(xs, pre, PREFILL)
    a2.xticks(xs)
    d.text((140, 862), "measured prompt tokens (log scale)", font=F_AXIS, fill=MUTED)

    peak = max(dec)
    deepest = depth[-1]
    note_y = 906
    d.text((70, note_y),
           "Decode holds %.0f%% of peak (%.1f tok/s) at %s tokens — %.0f%% of the window."
           % (100.0 * deepest["decode_tok_s"] / peak, peak,
              human_tokens(deepest["prompt_tokens"]),
              100.0 * deepest["prompt_tokens"] / 1000000),
           font=F_NOTE, fill=INK)

    if cold and warm and warm.get("ttft_ms") and cold.get("ttft_ms"):
        speedup = speedup_median or (cold["ttft_ms"] / warm["ttft_ms"])
        spread = ""
        if warm.get("speedup_min") and warm.get("speedup_max"):
            spread = " (median of 3, %.0fx-%.0fx)" % (warm["speedup_min"],
                                                      warm["speedup_max"])
        d.text((70, note_y + 24),
               "Prefill is a one-time cost: re-sending the same %s-token prompt "
               "reaches first token %.0fx sooner%s, %s of %s tokens reused from cache."
               % (human_tokens(warm["prompt_tokens"]), speedup, spread,
                  human_tokens(warm.get("cached_tokens", 0)),
                  human_tokens(warm["prompt_tokens"])),
               font=F_NOTE, fill=WARM)
        note_y += 24

    d.text((70, note_y + 24),
           "Generated by tools/plot_bench.py from bench/results/. "
           "Reproduce with: make bench && make plot",
           font=F_NOTE, fill=MUTED)

    img.save(args.out)
    print("wrote %s (%dx%d) from %d depth points" % (args.out, W, H, len(depth)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

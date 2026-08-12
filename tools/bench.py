#!/usr/bin/env python3
"""
bench.py — measure what actually matters on a 1M-context DGX Spark box.

The question this repo has to answer is not "how fast is a 200-token chat" but
"does the box still work at 900k tokens of context, and what does it cost".
So the suites here sweep depth rather than reporting a single number:

  depth        decode tok/s, prefill tok/s and TTFT at increasing prompt depth.
               The headline curve: flat decode = healthy, collapsing decode =
               the KV budget or the serial lane is in trouble.
  concurrency  N simultaneous streams at one depth. Shows whether the batch
               lane is funded (aggregate scales) or starved (it does not).
  needle       retrieval accuracy at depth. Throughput at 1M is meaningless if
               the model cannot see the middle of the window.
  cache        the same deep prompt twice: cold prefill vs warm-admit reuse.
  smoke        one short request, to prove the server answers at all.

Numbers come from the server's own `timings` block when it provides one
(ttft_ms, prefill/decode tok/s, cached token counts) and from wall clock
otherwise; every row records which. Depths are reported as the *measured*
prompt_tokens, never as the requested filler size.

Prefix caching is defeated per request with a random nonce at the head of the
prompt, so "cold" rows really are cold. `cache` deliberately does the opposite.

Usage:
  tools/bench.py smoke
  tools/bench.py depth --depths 1024,16384,131072,524288
  tools/bench.py concurrency --depth 32768 --streams 1,2,4
  tools/bench.py needle --depths 65536,262144
  tools/bench.py all --out bench/results/spark.json
  tools/bench.py all --quick

Options:
  --url URL         server base (default http://127.0.0.1:8888)
  --model NAME      label recorded in the results (default: from /v1/models)
  --tokens N        tokens to generate per decode sample (default 128)
  --repeat N        samples per point, best-of reported with spread (default 1)
  --out FILE        append a JSON record per row
  --markdown FILE   write a markdown report
  --timeout SEC     per-request timeout (default 1800)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8888"

# Filler that tokenizes densely and predictably, and that the model has no
# reason to continue in an interesting way. Deliberately not lorem ipsum: real
# English exercises the routed experts more like a real workload does.
CORPUS = (
    "The maintenance log records that the pump was inspected on a Tuesday and "
    "the seal was replaced without incident. Inventory shows twelve spare "
    "gaskets remaining in the east storage room. The night shift reported "
    "nominal pressure across all four lines and no alarms were raised. "
    "Calibration of the flow meter is scheduled for the following quarter. "
)


class Bench:
    def __init__(self, url: str, timeout: float, thinking: bool = False):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.thinking = thinking

    def get(self, path: str):
        req = urllib.request.Request(self.url + path)
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())

    def complete(self, prompt: str, max_tokens: int, system: str | None = None):
        """One non-streaming completion. Returns (wall_seconds, response).

        Thinking mode is disabled by default. ds4 defaults DeepSeek-compatible
        chat requests to thinking mode, which (a) spends tokens on deliberation
        that a throughput measurement should not be paying for, (b) ignores
        client sampling knobs entirely, so temperature=0 is silently discarded
        and runs are not reproducible, and (c) truncates into garbage when a
        short max_tokens cuts the reasoning before it closes -- the server then
        logs "thinking not closed, ignoring DSML in reasoning" and emits the
        raw deliberation as content. Pass --thinking to measure that path
        deliberately.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": "ds4",
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.thinking:
            payload["temperature"] = 1.0
        else:
            payload["thinking"] = {"type": "disabled"}
            payload["temperature"] = 0.0
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.url + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            payload = json.loads(r.read().decode())
        return time.perf_counter() - t0, payload


# Measured against this tokenizer on CORPUS: 5.4 characters per token, within a
# few percent from 1k to 500k. Only used to size the filler -- every row reports
# the prompt_tokens the server actually counted, never this estimate.
CHARS_PER_TOKEN = 5.4


def filler(target_tokens: int, nonce: str) -> str:
    need = max(0, int(target_tokens * CHARS_PER_TOKEN))
    reps = need // len(CORPUS) + 1
    return "%s\n%s" % (nonce, (CORPUS * reps)[:need])


def nonce() -> str:
    return "session %08x-%08x" % (random.getrandbits(32), random.getrandbits(32))


def sample(b: Bench, prompt: str, max_tokens: int, system: str | None = None) -> dict:
    """One measurement. Prefers server-side timings, falls back to wall clock."""
    wall, resp = b.complete(prompt, max_tokens, system=system)
    usage = resp.get("usage", {}) or {}
    timings = resp.get("timings", {}) or {}
    details = usage.get("prompt_tokens_details", {}) or {}
    out_tokens = usage.get("completion_tokens") or 0
    prompt_tokens = usage.get("prompt_tokens") or 0

    ttft = timings.get("ttft_ms")
    decode_tps = timings.get("decode_tok_s")
    prefill_tps = timings.get("prefill_tok_s")
    source = "server"
    if decode_tps is None:
        source = "wall"
        # Without a TTFT split, wall clock covers prefill+decode; attribute it
        # all to decode and say so, rather than inventing a split.
        decode_tps = out_tokens / wall if wall > 0 else 0.0
    if ttft is not None and prefill_tps is None and ttft > 0:
        prefill_tps = prompt_tokens / (ttft / 1000.0)

    text = ""
    reasoning = ""
    try:
        message = resp["choices"][0]["message"]
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
    except (KeyError, IndexError, TypeError):
        pass

    return {
        "wall_s": round(wall, 3),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": details.get("cached_tokens", 0),
        "output_tokens": out_tokens,
        "ttft_ms": round(ttft, 1) if ttft is not None else None,
        "prefill_tok_s": round(prefill_tps, 1) if prefill_tps else None,
        "decode_tok_s": round(decode_tps, 2),
        "timing_source": source,
        "text": text,
        "reasoning_chars": len(reasoning),
    }


def best_of(samples: list[dict], key: str):
    vals = [s[key] for s in samples if s.get(key) is not None]
    if not vals:
        return None, None
    spread = round(max(vals) - min(vals), 2) if len(vals) > 1 else 0.0
    return max(vals), spread


# ------------------------------------------------------------------- suites
def suite_smoke(b: Bench, args) -> list[dict]:
    s = sample(b, "Reply with exactly: ok", 16)
    print("smoke: %r  (%.2f tok/s, %s timings)"
          % (s["text"].strip()[:40], s["decode_tok_s"], s["timing_source"]))
    return [dict(suite="smoke", **{k: v for k, v in s.items() if k != "text"})]


def suite_depth(b: Bench, args) -> list[dict]:
    rows = []
    print("%9s %9s %9s %11s %11s %9s" %
          ("depth", "measured", "ttft_ms", "prefill t/s", "decode t/s", "spread"))
    for depth in args.depths:
        samples = []
        for _ in range(args.repeat):
            prompt = (filler(depth, nonce()) +
                      "\n\nSummarise the log above in two sentences.")
            samples.append(sample(b, prompt, args.tokens))
        dec, spread = best_of(samples, "decode_tok_s")
        pre, _ = best_of(samples, "prefill_tok_s")
        ttft = min([s["ttft_ms"] for s in samples if s["ttft_ms"] is not None]
                   or [None])
        measured = samples[0]["prompt_tokens"]
        print("%9d %9d %9s %11s %11.2f %9s" %
              (depth, measured,
               "%.0f" % ttft if ttft else "-",
               "%.0f" % pre if pre else "-", dec,
               "%.2f" % spread if spread is not None else "-"))
        rows.append({
            "suite": "depth", "requested_depth": depth, "prompt_tokens": measured,
            "output_tokens": args.tokens, "ttft_ms": ttft, "prefill_tok_s": pre,
            "decode_tok_s": dec, "spread_tok_s": spread,
            "timing_source": samples[0]["timing_source"],
            "cached_tokens": samples[0]["cached_tokens"],
        })
    if len(rows) > 1:
        first, last = rows[0]["decode_tok_s"], rows[-1]["decode_tok_s"]
        if first:
            print("decode retention %d -> %d tokens: %.0f%%"
                  % (rows[0]["prompt_tokens"], rows[-1]["prompt_tokens"],
                     100.0 * last / first))
    return rows


def suite_concurrency(b: Bench, args) -> list[dict]:
    # Two aggregates, because they answer different questions:
    #   wall     total output tokens / wall clock, prefill included. What a
    #            client experiences, but at deep prompts it mostly measures
    #            prefill and says little about batching.
    #   decode   sum of the per-stream decode rates. Isolates whether the batch
    #            lane is actually amortising weight traffic across sequences.
    # Run this suite at a shallow depth with a long generation to read `decode`
    # cleanly; a deep prompt makes one stream's decode compete with another's
    # prefill and understates the batching win.
    rows = []
    print("%8s %11s %12s %12s %11s %8s" %
          ("streams", "wall tok/s", "decode agg", "per-stream", "ttft_ms p50", "errors"))
    for n in args.streams:
        results: list[dict] = []
        errors: list[str] = []
        lock = threading.Lock()

        def worker():
            prompt = (filler(args.depth, nonce()) +
                      "\n\nSummarise the log above in two sentences.")
            try:
                s = sample(b, prompt, args.tokens)
            except Exception as e:                      # noqa: BLE001
                with lock:
                    errors.append(str(e))
                return
            with lock:
                results.append(s)

        threads = [threading.Thread(target=worker) for _ in range(n)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - t0

        produced = sum(r["output_tokens"] for r in results)
        agg = produced / elapsed if elapsed else 0.0
        decode_agg = sum(r["decode_tok_s"] for r in results)
        per = statistics.mean([r["decode_tok_s"] for r in results]) if results else 0.0
        ttfts = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
        p50 = statistics.median(ttfts) if ttfts else None
        print("%8d %11.2f %12.2f %12.2f %11s %8d"
              % (n, agg, decode_agg, per, "%.0f" % p50 if p50 else "-", len(errors)))
        if errors:
            print("    first error: %s" % errors[0][:120])
        rows.append({
            "suite": "concurrency", "streams": n, "depth": args.depth,
            "aggregate_tok_s": round(agg, 2),
            "aggregate_decode_tok_s": round(decode_agg, 2),
            "per_stream_tok_s": round(per, 2),
            "ttft_ms_p50": p50, "errors": len(errors),
            "error_sample": errors[0][:200] if errors else None,
        })
    return rows


NEEDLE = "The vault combination for the Nakamura cabinet is %d-%d-%d."
NEEDLE_Q = ("\n\nQuestion: what is the vault combination for the Nakamura "
            "cabinet? Answer with only the three numbers.")
NEEDLE_SYSTEM = ("You answer retrieval questions from the provided text. Reply "
                 "with only the requested value and nothing else.")


def suite_needle(b: Bench, args) -> list[dict]:
    rows = []
    print("%9s %9s %8s %10s %s" % ("depth", "measured", "place", "found", "answer"))
    for depth in args.depths:
        for place in (0.1, 0.5, 0.9):
            a, c, d = (random.randint(10, 99) for _ in range(3))
            secret = NEEDLE % (a, c, d)
            body = filler(depth, nonce())
            cut = int(len(body) * place)
            prompt = body[:cut] + "\n" + secret + "\n" + body[cut:] + NEEDLE_Q
            s = sample(b, prompt, 48, system=NEEDLE_SYSTEM)
            answer = " ".join(s["text"].split())[:48]
            found = all(str(x) in s["text"] for x in (a, c, d))
            print("%9d %9d %8.0f%% %10s %s"
                  % (depth, s["prompt_tokens"], place * 100,
                     "yes" if found else "NO", answer))
            rows.append({
                "suite": "needle", "requested_depth": depth,
                "prompt_tokens": s["prompt_tokens"], "placement": place,
                "found": found, "answer": answer,
                "decode_tok_s": s["decode_tok_s"],
            })
    hits = sum(1 for r in rows if r["found"])
    print("needle: %d/%d recovered" % (hits, len(rows)))
    return rows


def suite_cache(b: Bench, args) -> list[dict]:
    """Cold prefill vs warm-admit reuse of the same prompt.

    Each repeat gets its own nonce. Reusing one prompt across repeats would make
    every "cold" pass after the first a cache hit, and the suite would report a
    speedup of roughly 1x while looking perfectly healthy.
    """
    rows = []
    speedups = []
    print("%5s %6s %9s %9s %10s %11s"
          % ("run", "pass", "prompt", "cached", "ttft_ms", "decode t/s"))
    for i in range(max(1, args.repeat)):
        prompt = (filler(args.depth, nonce()) +
                  "\n\nSummarise the log above in two sentences.")
        cold = sample(b, prompt, args.tokens)
        warm = sample(b, prompt, args.tokens)
        for label, s in (("cold", cold), ("warm", warm)):
            print("%5d %6s %9d %9d %10s %11.2f"
                  % (i + 1, label, s["prompt_tokens"], s["cached_tokens"],
                     "%.0f" % s["ttft_ms"] if s["ttft_ms"] else "-",
                     s["decode_tok_s"]))
            rows.append(dict(suite="cache", pass_=label, run=i + 1,
                             **{k: v for k, v in s.items() if k != "text"}))
        if cold["ttft_ms"] and warm["ttft_ms"]:
            speedups.append(cold["ttft_ms"] / warm["ttft_ms"])

    if speedups:
        lo, hi = min(speedups), max(speedups)
        med = statistics.median(speedups)
        print("warm TTFT speedup: median %.0fx over %d run(s)%s"
              % (med, len(speedups),
                 "" if len(speedups) < 2 else " (range %.0fx-%.0fx)" % (lo, hi)))
        for r in rows:
            r["speedup_median"] = round(med, 1)
            r["speedup_min"] = round(lo, 1)
            r["speedup_max"] = round(hi, 1)
    return rows


def suite_restart(b: Bench, args) -> list[dict]:
    """Does a deep prompt survive a server restart?

    This is the disk KV tier (`--kv-disk-dir`). Without it a restart drops every
    resident bank and the second pass re-prefills from scratch; with it the bank
    is restored from a checkpoint. The suite proves which one you have.

    Both passes send a byte-identical prompt. The nonce is drawn once and reused,
    so the cold pass is genuinely cold even if earlier runs left checkpoints on
    disk, while the warm pass can still hit the one this run just wrote.

    This suite restarts your server. It is deliberately excluded from `all`.
    """
    fixed = nonce()
    prompt = (filler(args.depth, fixed) +
              "\n\nSummarise the log above in two sentences.")

    print("cold pass (this is a full prefill; expect minutes at depth) ...")
    cold = sample(b, prompt, args.tokens)

    print("restarting: %s" % args.restart_cmd)
    t0 = time.perf_counter()
    proc = subprocess.run(args.restart_cmd, shell=True, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError("restart command failed (%d): %s"
                           % (proc.returncode, " / ".join(tail)))
    for _ in range(int(args.restart_timeout)):
        try:
            b.get("/v1/models")
            break
        except Exception:                                  # noqa: BLE001
            time.sleep(1)
    else:
        raise RuntimeError("server did not come back within %ss" % args.restart_timeout)
    boot_s = time.perf_counter() - t0
    print("back up after %.0fs" % boot_s)

    warm = sample(b, prompt, args.tokens)

    print("%22s %9s %9s %10s %11s"
          % ("pass", "prompt", "cached", "ttft_ms", "decode t/s"))
    rows = []
    for label, s_ in (("cold", cold), ("after restart", warm)):
        print("%22s %9d %9d %10s %11.2f"
              % (label, s_["prompt_tokens"], s_["cached_tokens"],
                 "%.0f" % s_["ttft_ms"] if s_["ttft_ms"] else "-",
                 s_["decode_tok_s"]))
        rows.append(dict(suite="restart", pass_=label.replace(" ", "_"),
                         restart_boot_s=round(boot_s, 1),
                         **{k: v for k, v in s_.items() if k != "text"}))

    if warm["cached_tokens"] == 0:
        print("NOT RESTORED: the warm pass re-prefilled from scratch. Either disk KV")
        print("  is off (start.sh --kv-disk-dir / KV_DISK_DIR) or the checkpoint was")
        print("  never written -- ds4's --kv-cache-cold-max-tokens defaults to 30000.")
    elif cold["ttft_ms"] and warm["ttft_ms"]:
        speedup = cold["ttft_ms"] / warm["ttft_ms"]
        print("RESTORED %d of %d tokens: TTFT %.0f ms vs %.0f ms (%.0fx) across a restart"
              % (warm["cached_tokens"], warm["prompt_tokens"],
                 warm["ttft_ms"], cold["ttft_ms"], speedup))
        for r in rows:
            r["restart_speedup"] = round(speedup, 1)
    return rows


SUITES = {
    "smoke": suite_smoke,
    "depth": suite_depth,
    "concurrency": suite_concurrency,
    "needle": suite_needle,
    "cache": suite_cache,
    "restart": suite_restart,
}


def parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.replace(" ", "").split(",") if x]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", choices=list(SUITES) + ["all"], nargs="?", default="all")
    ap.add_argument("--url", default=os.environ.get("BENCH_URL", DEFAULT_URL))
    ap.add_argument("--model", default=None)
    ap.add_argument("--depths", default="1024,8192,32768,131072,524288")
    ap.add_argument("--depth", type=int, default=32768)
    ap.add_argument("--streams", default="1,2,4")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--restart-cmd", default="./start.sh --restart",
                    help="how the `restart` suite bounces the server")
    ap.add_argument("--restart-timeout", type=float, default=900,
                    help="seconds to wait for the server to answer after a restart")
    ap.add_argument("--out", default=None)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--thinking", action="store_true",
                    help="measure the thinking path (default: thinking disabled, "
                         "which is deterministic and does not burn tokens on "
                         "deliberation)")
    ap.add_argument("--quick", action="store_true",
                    help="shallower sweep, one stream set, fewer tokens")
    args = ap.parse_args(argv)

    args.depths = parse_ints(args.depths)
    args.streams = parse_ints(args.streams)
    if args.quick:
        args.depths = [1024, 8192, 32768]
        args.streams = [1, 2]
        args.tokens = min(args.tokens, 64)

    b = Bench(args.url, args.timeout, thinking=args.thinking)
    try:
        models = b.get("/v1/models")
        served = models["data"][0]["id"]
    except Exception as e:                                # noqa: BLE001
        print("bench: %s is not answering (%s)" % (args.url, e), file=sys.stderr)
        return 1
    label = args.model or served
    print("server : %s" % args.url)
    print("model  : %s" % label)
    print("thinking: %s" % ("on (sampling knobs ignored by the server)"
                            if args.thinking else "disabled"))
    print()

    # `restart` bounces the server, so it is opt-in only and never part of `all`.
    names = [args.suite]
    if args.suite == "all":
        names = ["smoke", "depth", "cache", "concurrency", "needle"]

    rows = []
    for name in names:
        print("== %s" % name)
        try:
            rows += SUITES[name](b, args)
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:300]
            print("   HTTP %s: %s" % (e.code, detail))
            rows.append({"suite": name, "error": "HTTP %s" % e.code, "detail": detail})
        except Exception as e:                            # noqa: BLE001
            print("   failed: %s" % e)
            rows.append({"suite": name, "error": str(e)})
        print()

    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta = {"model": label, "url": args.url, "timestamp": stamp,
            "host": os.uname().nodename, "tokens_per_sample": args.tokens,
            "thinking": args.thinking}

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "a") as f:
            for r in rows:
                f.write(json.dumps(dict(meta, **r)) + "\n")
        print("appended %d rows to %s" % (len(rows), args.out))

    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)), exist_ok=True)
        with open(args.markdown, "w") as f:
            f.write("# ds4 bench — %s\n\n%s on %s\n\n" % (label, stamp, meta["host"]))
            depth_rows = [r for r in rows if r.get("suite") == "depth"]
            if depth_rows:
                f.write("## Decode vs context depth\n\n")
                f.write("| prompt tokens | TTFT ms | prefill tok/s | decode tok/s |\n")
                f.write("|---:|---:|---:|---:|\n")
                for r in depth_rows:
                    f.write("| %d | %s | %s | %.2f |\n" % (
                        r["prompt_tokens"],
                        "%.0f" % r["ttft_ms"] if r.get("ttft_ms") else "-",
                        "%.0f" % r["prefill_tok_s"] if r.get("prefill_tok_s") else "-",
                        r["decode_tok_s"]))
                f.write("\n")
            conc = [r for r in rows if r.get("suite") == "concurrency"]
            if conc:
                f.write("## Concurrency at %d tokens\n\n" % args.depth)
                f.write("| streams | aggregate tok/s | per stream tok/s | errors |\n")
                f.write("|---:|---:|---:|---:|\n")
                for r in conc:
                    f.write("| %d | %.2f | %.2f | %d |\n" % (
                        r["streams"], r["aggregate_tok_s"],
                        r["per_stream_tok_s"], r["errors"]))
                f.write("\n")
            needles = [r for r in rows if r.get("suite") == "needle"]
            if needles:
                hits = sum(1 for r in needles if r["found"])
                f.write("## Retrieval\n\n%d/%d needles recovered.\n\n"
                        % (hits, len(needles)))
        print("wrote %s" % args.markdown)

    return 0 if not any("error" in r for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

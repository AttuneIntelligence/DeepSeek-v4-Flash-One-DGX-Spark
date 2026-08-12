#!/usr/bin/env bash
#
# repo-context.sh — put a whole repository in the context window, once.
#
# The point of a 1M-token window is not longer chats, it is skipping retrieval.
# Instead of grepping a repo into a prompt one chunk at a time and hoping the
# chunker picked the right files, you pack the entire thing in once, pay the
# prefill, and let ds4 checkpoint it. Every later turn against that same prefix
# is a warm admit: measured ~0.46s at 128k tokens, and ~2-4s even after a full
# server restart because the checkpoint is on disk.
#
# Measured on this box:
#   cold prefill of a 128k-token context      ~131 s
#   the same context, warm                    ~0.46 s      (~290x)
#   the same context, after a server restart  ~2.1 s       (~60x)
#   FOUR agents interrogating it concurrently ~5.3 s TTFT, 31 tok/s aggregate
#                                             vs ~524 s if each prefilled alone
#
# Usage:
#   tools/repo-context.sh pack .                  # print a packed context to stdout
#   tools/repo-context.sh pack . -o ctx.txt       # ...or to a file
#   tools/repo-context.sh warm ctx.txt            # prefill it and checkpoint it
#   tools/repo-context.sh ask ctx.txt "question"  # ask against the warm prefix
#   tools/repo-context.sh size .                  # estimate tokens before packing
#
# Options:
#   -o FILE          write the packed context here
#   --max-bytes N    stop packing at N bytes (default 4000000, ~740k tokens)
#   --include GLOB   extra path glob to include (repeatable)
#   --url URL        server (default http://127.0.0.1:8888)
#
# IMPORTANT: the prefix must be byte-identical between `warm` and `ask`, or you
# get a partial match and re-prefill the divergent tail. This script always
# sends the packed file verbatim followed by a separator and the question, so
# the prefix is stable as long as the file is.
set -euo pipefail

URL="${URL:-http://127.0.0.1:8888}"
MAX_BYTES="${MAX_BYTES:-4000000}"
OUT=""
EXTRA_GLOBS=()

die()  { echo "repo-context: $*" >&2; exit 1; }
note() { echo "==> $*" >&2; }

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; }

cmd="${1:-}"; shift || true
case "$cmd" in
    pack|warm|ask|size) ;;
    ""|--help|-h) usage; exit 0 ;;
    *) die "unknown command '$cmd' (pack|warm|ask|size)" ;;
esac

args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o)           OUT="$2"; shift 2 ;;
        --max-bytes)  MAX_BYTES="$2"; shift 2 ;;
        --include)    EXTRA_GLOBS+=("$2"); shift 2 ;;
        --url)        URL="$2"; shift 2 ;;
        --help|-h)    usage; exit 0 ;;
        *)            args+=("$1"); shift ;;
    esac
done

# Roughly 5.4 characters per token on this tokenizer for English prose; source
# code is denser, closer to 3.5-4. We use 4.0 so `size` errs toward over-warning.
CHARS_PER_TOKEN=4.0

# ---------------------------------------------------------------- packing
# Text files only, skipping the things that waste the window: VCS internals,
# build output, dependencies, lockfiles and binaries. Files are emitted with a
# stable path header so the model can cite locations, and in sorted order so the
# byte-identical-prefix property holds across runs.
pack_repo() {
    local root="$1" total=0
    local -a files=()
    while IFS= read -r f; do files+=("$f"); done < <(
        cd "$root" && git ls-files 2>/dev/null | sort || \
        find . -type f | sed 's|^\./||' | sort
    )
    [[ ${#files[@]} -gt 0 ]] || die "no files found under $root"

    printf '# Repository context: %s\n' "$(cd "$root" && basename "$PWD")"
    printf '# Packed %s, %d candidate files.\n' "$(date -u +%FT%TZ)" "${#files[@]}"
    printf '# Each file is delimited by ===== FILE: <path> =====\n\n'

    local f sz
    for f in "${files[@]}"; do
        case "$f" in
            .git/*|*/.git/*|node_modules/*|*/node_modules/*|\
            *.png|*.jpg|*.jpeg|*.gif|*.webp|*.ico|*.pdf|*.zip|*.gz|*.xz|*.zst|\
            *.tar|*.o|*.a|*.so|*.dylib|*.dll|*.exe|*.bin|*.gguf|*.kv|*.safetensors|\
            *.pyc|*.class|*.wasm|*.mp4|*.mov|*.mp3|*.wav|\
            *.lock|package-lock.json|yarn.lock|poetry.lock|Cargo.lock|\
            dist/*|build/*|target/*|__pycache__/*|*/__pycache__/*|.venv/*|venv/*)
                continue ;;
        esac
        [ -f "$root/$f" ] || continue
        # Skip anything that is not text.
        if ! file -b --mime-encoding "$root/$f" 2>/dev/null | grep -qvE 'binary'; then
            continue
        fi
        sz=$(stat -c%s "$root/$f")
        if (( total + sz > MAX_BYTES )); then
            printf '\n# TRUNCATED: budget of %d bytes reached before %s\n' "$MAX_BYTES" "$f"
            note "hit --max-bytes $MAX_BYTES; stopped before $f"
            break
        fi
        printf '===== FILE: %s =====\n' "$f"
        cat "$root/$f"
        printf '\n'
        total=$(( total + sz + 32 ))
    done
    note "packed $(awk -v b="$total" 'BEGIN{printf "%.2f", b/1048576}') MiB \
(~$(awk -v b="$total" -v c="$CHARS_PER_TOKEN" 'BEGIN{printf "%d", b/c}') tokens)"
}

# ---------------------------------------------------------------- requests
# Prefix first, then a stable separator, then the turn. Keeping this identical
# between warm and ask is the whole trick.
SEPARATOR=$'\n\n===== END OF REPOSITORY CONTEXT =====\n\n'

post() { # <context-file> <question> <max-tokens>
    python3 - "$1" "$2" "$3" "$URL" <<'PY'
import json, sys, time, urllib.request
ctx_path, question, max_tokens, url = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
with open(ctx_path, encoding="utf-8", errors="replace") as f:
    ctx = f.read()
prompt = ctx + "\n\n===== END OF REPOSITORY CONTEXT =====\n\n" + question
body = json.dumps({
    "model": "ds4",
    "thinking": {"type": "disabled"},
    "temperature": 0,
    "max_tokens": max_tokens,
    "messages": [{"role": "user", "content": prompt}],
}).encode()
req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=3600) as r:
    d = json.loads(r.read().decode())
wall = time.perf_counter() - t0
u, t = d.get("usage", {}), d.get("timings", {})
cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
total = u.get("prompt_tokens", 0)
sys.stderr.write(
    "==> %d prompt tokens, %d served from cache (%.1f%%), ttft %.0f ms, %.1f tok/s, %.1fs wall\n"
    % (total, cached, 100.0 * cached / total if total else 0.0,
       t.get("ttft_ms") or 0.0, t.get("decode_tok_s") or 0.0, wall))
if total and cached == 0:
    sys.stderr.write("==> NOTE: nothing reused. First run is expected to be cold; if a\n"
                     "==> second identical run is also cold, the prefix is not stable.\n")
print(d["choices"][0]["message"]["content"])
PY
}

root="${args[0]:-.}"
case "$cmd" in
    size)
        b=$(pack_repo "$root" 2>/dev/null | wc -c)
        awk -v b="$b" -v c="$CHARS_PER_TOKEN" 'BEGIN{
            printf "packed bytes : %.2f MiB\nest. tokens  : %d\n", b/1048576, b/c
            printf "est. cold prefill at 950 tok/s : %.1f min\n", (b/c)/950/60
            printf "est. disk checkpoint at 4.6 KiB/token : %.2f GiB\n", (b/c)*4.6/1048576 }'
        ;;
    pack)
        if [ -n "$OUT" ]; then pack_repo "$root" > "$OUT"; note "wrote $OUT";
        else pack_repo "$root"; fi
        ;;
    warm)
        [ -f "$root" ] || die "warm wants a packed context FILE (see: repo-context.sh pack)"
        note "prefilling and checkpointing $root -- this is the slow one"
        post "$root" "Reply with the single word: ready." 8 >/dev/null
        note "warm. later turns against this file reuse the prefix."
        ;;
    ask)
        [ -f "$root" ] || die "ask wants a packed context FILE"
        q="${args[1]:-}"
        [ -n "$q" ] || die "ask wants a question: repo-context.sh ask ctx.txt 'question'"
        post "$root" "$q" 1024
        ;;
esac

#!/usr/bin/env bash
#
# start.sh — serve a DeepSeek-V4-Flash variant on :8888 via ds4-server.
#
# Models now live on the Empress NAS (NFSv4, autofs):
#     /home/reed/Empress -> /mnt/empress  ->  empress:/mnt/tank/models
# Both supported builds are present there. A local copy under ~/gguf is used
# automatically when it matches, because pulling 86 GiB over NFS costs ~4 min.
#
# ds4-on-spark's install.sh has no --host option, so it can never bind anything
# but 127.0.0.1. This wrapper therefore calls the ds4-serve launcher directly
# (it does support --host) and only falls back to the upstream installer when
# you explicitly ask for it with --install.
#
# Usage:
#   ./start.sh                         # stock model, 1M ctx, 0.0.0.0:8888
#   ./start.sh --model abliterated     # abliterated DS4-Headroom128 build
#   ./start.sh --list                  # show models and where they resolve
#   ./start.sh --restart               # stop whatever is serving, then start
#   ./start.sh --host 127.0.0.1        # loopback only
#   ./start.sh --source empress        # force NFS even if ~/gguf has a copy
#   ./start.sh --fg                    # run in foreground instead of nohup
#   ./start.sh --install               # legacy path: run upstream install.sh
#   ./start.sh --no-dspark             # unknown flags pass through to ds4-serve
#
# Memory governance (see "deep-context governance" below):
#   ./start.sh --max-out 65536         # per-request KV credit; default 32768
#   ./start.sh --coalesce-max 8        # concurrent batch banks; default 4
#   ./start.sh --serial-max-tokens 0   # 0 = fail fast instead of degrading
#
# Env equivalents: MODEL PORT CTX HOST SOURCE MODELS_ROOT LOCAL_GGUF_DIR LOG
#                  MAX_OUT COALESCE_MAX SERIAL_MAX_TOKENS MEM_FLOOR_GB
#
# KV budget is ~35.6 KiB/token at ctx=1000000 (13.17 GiB kv cache + 20.80 GiB
# context buffers, measured on this host) -- so 1M ctx ~= 34 GiB on top of the
# ~87 GiB of weights, which is 121 of this box's 121.69 GiB.
# The script prints the arithmetic and refuses to start if it will not fit.
#
# ---- deep-context governance (why the defaults below are not ds4's) --------
# At -c 1000000 this box is at ~99.6% before serving a token. It boots anyway
# because the serial lane's graph is allocated LAZILY (ds4 logs "session graph
# allocated lazily"), so boot sees ~15 GiB free. That memory then drains as the
# serial lane right-sizes upward on deep prompts -- and it never shrinks:
#
#     23:16:00  ctx 1000000 -> 2561      usable 162.1 MiB
#     23:16:25  ctx    2561 -> 16115     usable   0.0 MiB
#     23:21:38  ctx   16115 -> 78924     usable   0.0 MiB  (pinned for 9h)
#
# Once usable hits 0 the batch lane rejects every admission on the memory floor,
# and prompts above DS4_SERVER_SERIAL_MAX_TOKENS get a 503 from the deep-serial
# guard instead of a fallback. The server wedges into a reject/503 loop that
# only a restart clears. Three ds4 defaults are wrong for a 1M-ctx box:
#
#   -n 393216   Every admission credits (prompt + max_out) tokens of KV growth
#               and holds it for the row's lifetime. At the ds4 default a 68k
#               prompt reserves 461k tokens. MAX_OUT caps the default credit;
#               clients that send their own max_tokens are unaffected.
#   COALESCE 32 ds4 asks for 32 banks, the fit claws it back to ~9, and those
#               still commit ~7 of the 15 free GiB. This box cannot serve 32
#               concurrent 1M streams; 4 returns ~3.8 GiB to the usable pool.
#   SERIAL_MAX  Hardcoded 65536 in ds4_server.c, does NOT scale with -c (it was
#      = 65536  sized for a -c 131072 box where ctx/2 happened to equal 65536).
#               At 1M ctx that is a cliff at 6.5% of context.
#
# Tradeoff on the third: raising it turns a fast 503 into a slow success, and
# the serial lane is what drives the creep above. MAX_OUT and COALESCE_MAX are
# what actually keep the batch lane funded; SERIAL_MAX_TOKENS is the safety net
# for when it is not. Set --serial-max-tokens 0 to fail fast instead.
#
# MEM_FLOOR_GB is deliberately left at ds4's default of 4. With ~0.5 GiB of true
# slack it is the only thing between this box and the OOM killer, and ds4 caches
# it in a static on first read, so it cannot be retuned without a restart.
set -euo pipefail

MODEL="${MODEL:-stock}"
PORT="${PORT:-8888}"
CTX="${CTX:-1000000}"
HOST="${HOST:-0.0.0.0}"
SOURCE="${SOURCE:-auto}"                       # auto | empress | local
MODELS_ROOT="${MODELS_ROOT:-$HOME/Empress/models}"
LOCAL_GGUF_DIR="${LOCAL_GGUF_DIR:-$HOME/gguf}"
LOG="${LOG:-$HOME/ds4-server.log}"
# Measured on this host at ctx=1000000: kv cache 13.17 GiB + context buffers
# 20.80 GiB = 33.97 GiB, i.e. ~35.6 KiB/token -- not the 9.5 KiB/token quoted in
# the original script header, which undercounts the context buffers by ~3.7x.
KV_KIB_PER_TOKEN="${KV_KIB_PER_TOKEN:-35.6}"

# Deep-context governance. See the header block for why these override ds4.
MAX_OUT="${MAX_OUT:-32768}"                    # ds4 -n; ds4 default 393216
COALESCE_MAX="${COALESCE_MAX:-4}"              # banks; ds4 default 32
SERIAL_MAX_TOKENS="${SERIAL_MAX_TOKENS:-}"     # empty => CTX; 0 => fail fast
MEM_FLOOR_GB="${MEM_FLOOR_GB:-}"               # empty => ds4 default (4 GiB)

FOREGROUND=0; DO_LIST=0; DO_INSTALL=0; DO_RESTART=0; SKIP_MEMCHECK=0; DRY_RUN=0
extra=()

die()  { echo "start.sh: $*" >&2; exit 1; }
note() { echo "==> $*"; }

# Print the leading comment block, however long it grows.
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|-M)   MODEL="$2"; shift 2 ;;
        --host)       HOST="$2"; shift 2 ;;
        --port)       PORT="$2"; shift 2 ;;
        --ctx|-c)     CTX="$2"; shift 2 ;;
        --source)     SOURCE="$2"; shift 2 ;;
        --local)      SOURCE=local; shift ;;
        --empress)    SOURCE=empress; shift ;;
        --list)       DO_LIST=1; shift ;;
        --restart)    DO_RESTART=1; shift ;;
        --fg|--foreground) FOREGROUND=1; shift ;;
        --install)    DO_INSTALL=1; shift ;;
        --no-memcheck) SKIP_MEMCHECK=1; shift ;;
        --strict-mem) STRICT_MEM=1; shift ;;
        --max-out)            MAX_OUT="$2"; shift 2 ;;
        --coalesce-max)       COALESCE_MAX="$2"; shift 2 ;;
        --serial-max-tokens)  SERIAL_MAX_TOKENS="$2"; shift 2 ;;
        --mem-floor-gb)       MEM_FLOOR_GB="$2"; shift 2 ;;
        # -n is dry-run here for back-compat; ds4's -n is --max-out.
        --dry-run|-n) DRY_RUN=1; shift ;;
        --help|-h)    usage; exit 0 ;;
        *)            extra+=("$1"); shift ;;    # forwarded to ds4-serve
    esac
done

# ------------------------------------------------- governance knob resolution
is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }
is_uint "$CTX"          || die "--ctx wants a non-negative integer (got '$CTX')"
is_uint "$MAX_OUT"      || die "--max-out wants a non-negative integer (got '$MAX_OUT')"
is_uint "$COALESCE_MAX" || die "--coalesce-max wants a non-negative integer (got '$COALESCE_MAX')"
(( COALESCE_MAX >= 1 )) || die "--coalesce-max must be >= 1 (got '$COALESCE_MAX')"

# The deep-serial guard is a flat token count in ds4 and does not scale with -c,
# so track CTX unless the operator pins it. 0 disables the fallback (fail fast).
[ -n "$SERIAL_MAX_TOKENS" ] || SERIAL_MAX_TOKENS="$CTX"
is_uint "$SERIAL_MAX_TOKENS" \
    || die "--serial-max-tokens wants a non-negative integer (got '$SERIAL_MAX_TOKENS')"

if [ -n "$MEM_FLOOR_GB" ]; then
    is_uint "$MEM_FLOOR_GB" \
        || die "--mem-floor-gb wants a non-negative integer in GiB (got '$MEM_FLOOR_GB')"
fi

# ds4 caps the batch path at 2 banks minimum for the continuous lane; below that
# it silently drops to the serial path, which is the lane we are trying to avoid.
if (( COALESCE_MAX < 2 )); then
    note "WARNING: --coalesce-max 1 disables the continuous batch path;"
    note "         every request will take the serial lane."
fi

# ---------------------------------------------------------------- model table
# Each entry: <subdir>|<base gguf>|<dspark gguf>
model_spec() {
    case "$1" in
        stock|base|antirez|iq2xxs)
            echo "DeepSeek-v4-Flash|DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf|DSpark-drafter-Q2K-Q8-0731.gguf" ;;
        abliterated|ablit|headroom128|hr128)
            echo "DeepSeek-V4-Flash-0731-Abliterated-DS4-Headroom128|DeepSeek-V4-Flash-0731-Abliterated-DS4-Headroom128.gguf|DeepSeek-V4-Flash-0731-Abliterated-DS4-Headroom128-DSpark-support.gguf" ;;
        *) return 1 ;;
    esac
}
ALL_MODELS="stock abliterated"

# Pick the directory that actually holds both files, honouring $SOURCE.
resolve_dir() {
    local sub="$1" base="$2" dspark="$3" nfs="$MODELS_ROOT/$1" loc="$LOCAL_GGUF_DIR"
    local have_nfs=0 have_loc=0
    [[ -f "$nfs/$base" && -f "$nfs/$dspark" ]] && have_nfs=1
    [[ -f "$loc/$base" && -f "$loc/$dspark" ]] && have_loc=1
    case "$SOURCE" in
        local)   (( have_loc )) && { echo "$loc"; return 0; }; return 1 ;;
        empress) (( have_nfs )) && { echo "$nfs"; return 0; }; return 1 ;;
        auto)    (( have_loc )) && { echo "$loc"; return 0; }
                 (( have_nfs )) && { echo "$nfs"; return 0; }; return 1 ;;
        *) die "--source must be auto, empress or local (got '$SOURCE')" ;;
    esac
}

if (( DO_LIST )); then
    printf '%-14s %-9s %s\n' MODEL SOURCE PATH
    for m in $ALL_MODELS; do
        IFS='|' read -r sub base dspark <<<"$(model_spec "$m")"
        if d="$(resolve_dir "$sub" "$base" "$dspark")"; then
            case "$d" in "$LOCAL_GGUF_DIR") s=local ;; *) s=empress ;; esac
            printf '%-14s %-9s %s\n' "$m" "$s" "$d"
        else
            printf '%-14s %-9s %s\n' "$m" MISSING "$MODELS_ROOT/$sub"
        fi
    done
    exit 0
fi

IFS='|' read -r SUB BASE DSPARK <<<"$(model_spec "$MODEL")" \
    || die "unknown --model '$MODEL' (known: $ALL_MODELS)"

# Touch the autofs path so the NFS mount is triggered before we probe it.
[ -d "$MODELS_ROOT" ] || die "$MODELS_ROOT is not available (is the Empress NFS mount up?)"

SRCDIR="$(resolve_dir "$SUB" "$BASE" "$DSPARK")" \
    || die "model '$MODEL' not found. Looked in $MODELS_ROOT/$SUB and $LOCAL_GGUF_DIR"

# ------------------------------------------------------------- legacy install
if (( DO_INSTALL )); then
    note "Running upstream installer (--start, port $PORT, ctx $CTX)"
    tmp="$(mktemp)"; trap 'rm -f "$tmp"' EXIT
    curl -fsSL https://github.com/Entrpi/ds4-on-spark/raw/main/install.sh -o "$tmp"
    bash "$tmp" --start --port "$PORT" --ctx "$CTX" ${extra[@]+"${extra[@]}"}
    exit $?
fi

command -v ds4-serve >/dev/null 2>&1 || {
    [ -x "$HOME/.local/bin/ds4-serve" ] || die "ds4-serve not installed; run: ./start.sh --install"
    PATH="$HOME/.local/bin:$PATH"
}

# ---------------------------------------------------------------- memory math
bytes_gib() { awk -v b="$1" 'BEGIN{printf "%.2f", b/1073741824}'; }
W_BASE=$(stat -c %s "$SRCDIR/$BASE")
W_DRAFT=$(stat -c %s "$SRCDIR/$DSPARK")
KV_GIB=$(awk -v c="$CTX" -v k="$KV_KIB_PER_TOKEN" 'BEGIN{printf "%.2f", c*k/1048576}')
NEED=$(awk -v a="$W_BASE" -v b="$W_DRAFT" -v kv="$KV_GIB" \
         'BEGIN{printf "%.2f", a/1073741824 + b/1073741824 + kv}')
TOTAL_GIB=$(awk '/MemTotal/{printf "%.2f", $2/1048576}' /proc/meminfo)

note "model    : $MODEL"
note "source   : $SRCDIR"
note "weights  : $(bytes_gib "$W_BASE") GiB base + $(bytes_gib "$W_DRAFT") GiB DSpark"
note "kv cache : $KV_GIB GiB  ($CTX tok x $KV_KIB_PER_TOKEN KiB/tok)"
note "required : $NEED GiB of $TOTAL_GIB GiB RAM"
note "bind     : $HOST:$PORT"
note "governor : max_out=$MAX_OUT banks=$COALESCE_MAX serial_max=$SERIAL_MAX_TOKENS\
 mem_floor=${MEM_FLOOR_GB:-4} GiB"
if (( SERIAL_MAX_TOKENS == 0 )); then
    note "           serial fallback OFF -- deep prompts get 503 rather than degrade"
fi

# This estimate is an UPPER bound: ds4-server mmaps the base GGUF unpinned and
# serves experts from in-process device artifacts, so weights are not fully
# resident on top of the runtime buffers. Treat overshoot as advice, not a
# verdict -- gate startup only when --strict-mem is given.
if (( ! SKIP_MEMCHECK )) && awk -v n="$NEED" -v t="$TOTAL_GIB" 'BEGIN{exit !(n > t*0.97)}'; then
    if [ "${STRICT_MEM:-0}" = 1 ]; then
        die "estimate ${NEED} GiB exceeds ${TOTAL_GIB} GiB. Lower --ctx or drop --strict-mem."
    fi
    note "WARNING: upper-bound estimate ${NEED} GiB vs ${TOTAL_GIB} GiB RAM."
    note "         weights are mmapped, so this usually still fits; watch for"
    note "         'reduced from requested to fit memory' in $LOG."
fi

if (( DRY_RUN )); then
    note "dry run — would exec:"
    echo "    DS4_GGUF_DIR=$SRCDIR GGUF_FILE=$BASE DSPARK_FILE=$DSPARK \\"
    echo "    DS4_SERVER_COALESCE_MAX=$COALESCE_MAX \\"
    echo "    DS4_SERVER_SERIAL_MAX_TOKENS=$SERIAL_MAX_TOKENS \\"
    if [ -n "$MEM_FLOOR_GB" ]; then echo "    DS4_MEM_FLOOR_GB=$MEM_FLOOR_GB \\"; fi
    echo "    ds4-serve -c $CTX -n $MAX_OUT --port $PORT --host $HOST ${extra[*]-}"
    exit 0
fi

# ------------------------------------------------------------------- (re)start
if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    if (( DO_RESTART )); then
        note "Stopping the server currently on :$PORT ..."
        pkill -x ds4-server || true
        # ds4-server drains requests and holds its single-instance lock well
        # after the listening socket closes, so waiting on the port alone races
        # the next launch into "another ds4 process is already running".
        # Wait for the process itself to disappear.
        for _ in $(seq 1 240); do
            pgrep -x ds4-server >/dev/null 2>&1 || break
            sleep 0.5
        done
        if pgrep -x ds4-server >/dev/null 2>&1; then
            note "still draining after 120s — sending SIGKILL"
            pkill -9 -x ds4-server || true
            for _ in $(seq 1 40); do
                pgrep -x ds4-server >/dev/null 2>&1 || break
                sleep 0.5
            done
        fi
        pgrep -x ds4-server >/dev/null 2>&1 && die "could not stop the running ds4-server"
        while ss -tln "sport = :$PORT" 2>/dev/null | grep -q ":$PORT"; do sleep 0.3; done
    else
        note "Already serving on :$PORT — use --restart to switch models."
        curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -m json.tool 2>/dev/null || true
        exit 0
    fi
fi

launch=(env "DS4_GGUF_DIR=$SRCDIR" "GGUF_FILE=$BASE" "DSPARK_FILE=$DSPARK"
        "DS4_SERVER_COALESCE_MAX=$COALESCE_MAX"
        "DS4_SERVER_SERIAL_MAX_TOKENS=$SERIAL_MAX_TOKENS"
        ${MEM_FLOOR_GB:+"DS4_MEM_FLOOR_GB=$MEM_FLOOR_GB"}
        ds4-serve -c "$CTX" -n "$MAX_OUT" --port "$PORT" --host "$HOST"
        ${extra[@]+"${extra[@]}"})

if (( FOREGROUND )); then
    exec "${launch[@]}"
fi

note "Launching in background; log -> $LOG"
nohup "${launch[@]}" >>"$LOG" 2>&1 &
pid=$!

for _ in $(seq 1 600); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo
        note "ds4-server is up on http://$HOST:$PORT (pid $pid)"
        curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -m json.tool 2>/dev/null || true
        if [ "$HOST" = "0.0.0.0" ]; then
            note "LAN clients: http://$(hostname -I | awk '{print $1}'):$PORT"
        fi
        exit 0
    fi
    kill -0 "$pid" 2>/dev/null || die "ds4-server exited during load; see $LOG"
    sleep 1
done

die "not reachable after 600s; see $LOG"

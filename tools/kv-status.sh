#!/usr/bin/env bash
#
# kv-status.sh — is the disk KV tier earning its keep, and is it about to fill?
#
# The disk tier is enabled by default (start.sh --kv-disk-dir) on the strength of
# a 30x restart-recovery measurement. That measurement was one clean test on an
# empty cache; the things it did NOT exercise are budget pressure and eviction
# behaviour over days of real sessions. This is the thing to watch.
#
# Usage:
#   tools/kv-status.sh              # one-shot report
#   tools/kv-status.sh --watch      # refresh every 30s
#
# Env: KV_DISK_DIR (default ~/ds4-kv), KV_DISK_MB (default 65536), PORT (8888)
set -euo pipefail

KV_DISK_DIR="${KV_DISK_DIR:-$HOME/ds4-kv}"
KV_DISK_MB="${KV_DISK_MB:-65536}"
PORT="${PORT:-8888}"
URL="http://127.0.0.1:$PORT"
WATCH=0
[ "${1:-}" = "--watch" ] && WATCH=1

report() {
    echo "== disk KV tier  $(date '+%F %T')"
    if [ ! -d "$KV_DISK_DIR" ]; then
        echo "  $KV_DISK_DIR does not exist -- disk KV is off (KV_DISK_DIR= to confirm)"
        return
    fi

    local used_kb used_mb files pct
    used_kb=$(du -sk "$KV_DISK_DIR" 2>/dev/null | awk '{print $1}')
    used_mb=$(( used_kb / 1024 ))
    files=$(find "$KV_DISK_DIR" -name '*.kv' 2>/dev/null | wc -l)
    pct=$(awk -v u="$used_mb" -v b="$KV_DISK_MB" \
              'BEGIN{ if (b > 0) printf "%.0f", 100*u/b; else print 0 }')

    printf '  dir      : %s\n' "$KV_DISK_DIR"
    printf '  used     : %s MiB of %s MiB budget (%s%%), %s checkpoint(s)\n' \
        "$used_mb" "$KV_DISK_MB" "$pct" "$files"
    if [ "$files" -gt 0 ]; then
        printf '  oldest   : %s\n' \
            "$(find "$KV_DISK_DIR" -name '*.kv' -printf '%T+ %f\n' 2>/dev/null | sort | head -1)"
        printf '  largest  : %s\n' \
            "$(find "$KV_DISK_DIR" -name '*.kv' -printf '%s %f\n' 2>/dev/null \
               | sort -rn | head -1 | awk '{printf "%.0f MiB  %s", $1/1048576, $2}')"
    fi
    # Free space matters independently of the budget: ds4 trims to its budget,
    # not to what the filesystem has left.
    printf '  fs free  : %s\n' "$(df -h "$KV_DISK_DIR" | awk 'NR==2{print $4" on "$6}')"
    if [ "$pct" -ge 90 ]; then
        echo "  NOTE: at/over budget -- ds4 evicts by hit-decayed LRU, so the deep"
        echo "        trunk you care about can be dropped if it has gone cold."
    fi

    echo "== restores and admissions (since boot)"
    if ! curl -sf --max-time 5 "$URL/metrics" -o /tmp/.kvstat.$$ 2>/dev/null; then
        echo "  server not answering on $URL"
        rm -f /tmp/.kvstat.$$
        return
    fi
    awk '
        /^ds4_admits_total\{kind="cold"\}/          {cold=$2}
        /^ds4_admits_total\{kind="warm"\}/          {warm=$2}
        /^ds4_admits_total\{kind="fork"\}/          {fork=$2}
        /^ds4_admits_total\{kind="partial_fork"\}/  {pfork=$2}
        /^ds4_cont_admit_rejects_total/             {rej=$2}
        /^ds4_banks_live/                           {live=$2}
        /^ds4_banks_total/                          {tot=$2}
        /^ds4_kv_pages_resident/                    {pages=$2}
        /^ds4_spec_accept_ratio/                    {acc=$2}
        /^ds4_decode_tok_s/                         {dec=$2}
        END {
            printf "  admits   : cold=%s warm=%s fork=%s partial_fork=%s\n",
                   cold+0, warm+0, fork+0, pfork+0
            printf "  rejects  : %s on the memory floor / comp budget\n", rej+0
            printf "  banks    : %s live of %s\n", live+0, tot+0
            printf "  kv pages : %s resident\n", pages+0
            printf "  decode   : %.1f tok/s, spec accept %.3f\n", dec+0, acc+0
        }' /tmp/.kvstat.$$
    rm -f /tmp/.kvstat.$$

    # Restores are logged, not counted in /metrics.
    local log="${LOG:-$HOME/ds4-server.log}"
    if [ -f "$log" ]; then
        local hits
        hits=$(grep -c "kv cache bank restore hit" "$log" 2>/dev/null || true)
        printf '  restores : %s "bank restore hit" line(s) in %s\n' "${hits:-0}" "$log"
        grep "kv cache bank restore hit" "$log" 2>/dev/null | tail -2 | sed 's/^/    /' || true
    fi
    echo "  a warm admit that re-prefilled shows up as cold with cached=0 in the"
    echo "  response usage block -- that is the signal the tier missed."
}

if (( WATCH )); then
    while true; do clear; report; sleep 30; done
else
    report
fi

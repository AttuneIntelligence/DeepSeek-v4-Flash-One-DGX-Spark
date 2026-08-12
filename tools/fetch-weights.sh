#!/usr/bin/env bash
#
# fetch-weights.sh — download a weight set and leave it in a servable state.
#
# For each model this fetches the target GGUF and its DSpark drafter from
# Hugging Face, then runs whatever repair that drafter needs (see
# tools/gguf_dspark_remap.py). When it exits 0, `./start.sh --model <name>`
# will boot with speculative decode armed.
#
# Usage:
#   tools/fetch-weights.sh                     # every model, into $LOCAL_GGUF_DIR
#   tools/fetch-weights.sh abliterated         # one model
#   tools/fetch-weights.sh --dest empress all  # into $MODELS_ROOT/<subdir>
#   tools/fetch-weights.sh --check abliterated # report only, download nothing
#   tools/fetch-weights.sh --repair-only ablit # skip the download, just repair
#
# Options:
#   --dest local|empress   where to land the files (default: local)
#   --check                print what is present/missing and exit
#   --repair-only          do not download; run the DSpark repair if it applies
#   --no-repair            download only; leave the drafter as published
#   --force                re-download even if the size already matches
#
# Env: LOCAL_GGUF_DIR MODELS_ROOT HF_TOKEN
#
# Downloads go through `hf download` when the Hugging Face CLI is installed
# (Xet-accelerated, resumable, integrity-checked) and fall back to resumable
# curl otherwise. Both are safe to re-run: a file whose size already matches
# the remote is left alone.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"

die()  { echo "fetch-weights: $*" >&2; exit 1; }
note() { echo "==> $*"; }

# shellcheck source=tools/models.sh
. "$REPO_DIR/tools/models.sh"

DEST=local
CHECK=0
REPAIR=1
REPAIR_ONLY=0
FORCE=0
want=()

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "$0"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest)        DEST="$2"; shift 2 ;;
        --local)       DEST=local; shift ;;
        --empress)     DEST=empress; shift ;;
        --check)       CHECK=1; shift ;;
        --repair-only) REPAIR_ONLY=1; shift ;;
        --no-repair)   REPAIR=0; shift ;;
        --force)       FORCE=1; shift ;;
        --help|-h)     usage; exit 0 ;;
        -*)            die "unknown option '$1'" ;;
        all)           want=($ALL_MODELS); shift ;;
        *)             want+=("$1"); shift ;;
    esac
done
[[ ${#want[@]} -gt 0 ]] || want=($ALL_MODELS)

case "$DEST" in
    local)   ;;
    empress) [ -d "$MODELS_ROOT" ] || die "$MODELS_ROOT is not available (NFS mount down?)" ;;
    *) die "--dest must be local or empress (got '$DEST')" ;;
esac

dest_dir() {
    case "$DEST" in
        local)   echo "$LOCAL_GGUF_DIR" ;;
        empress) echo "$MODELS_ROOT/$(model_field "$1" sub)" ;;
    esac
}

human() { awk -v b="$1" 'BEGIN{printf "%.2f GiB", b/1073741824}'; }

# Remote size via a HEAD against the resolve URL. Empty on failure, and callers
# treat that as "cannot verify" rather than "mismatch" — a flaky HEAD must never
# trigger an 80 GiB re-download.
remote_size() {
    curl -sIL "https://huggingface.co/$1/resolve/main/$2" \
        | awk -F': ' 'tolower($1)=="content-length"{n=$2+0} END{if(n)print n}'
}

have_file() {   # <dir> <repo> <file> -> 0 if present and the right size
    local path="$1/$3" remote local_size
    [ -f "$path" ] || return 1
    (( FORCE )) && return 1
    local_size=$(stat -c%s "$path")
    remote=$(remote_size "$2" "$3")
    if [ -z "$remote" ]; then
        note "  $3 present ($(human "$local_size")); could not reach HF to confirm the size"
        return 0
    fi
    [ "$remote" = "$local_size" ] && return 0
    note "  $3 is $(human "$local_size"), remote is $(human "$remote") — resuming"
    return 1
}

download() {    # <dir> <repo> <file>
    local dir="$1" repo="$2" file="$3"
    mkdir -p "$dir"
    if command -v hf >/dev/null 2>&1; then
        note "  hf download $repo $file"
        hf download "$repo" "$file" --local-dir "$dir" >/dev/null \
            || die "hf download failed for $repo/$file"
    else
        note "  curl $repo/$file"
        curl -L --fail --progress-bar -C - \
            ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
            -o "$dir/$file" \
            "https://huggingface.co/$repo/resolve/main/$file" \
            || die "download failed for $repo/$file"
    fi
}

status_line() { # <model>
    local m="$1" dir base drafter served
    dir="$(dest_dir "$m")"
    base="$(model_field "$m" base)"
    drafter="$(model_field "$m" dspark_src)"
    served="$(model_field "$m" dspark)"
    local b=missing d=missing r=n/a
    [ -f "$dir/$base" ] && b=present
    [ -f "$dir/$drafter" ] && d=present
    if [ "$served" != "$drafter" ]; then
        if [ -f "$dir/$served" ]; then r=repaired; else r=needed; fi
    fi
    printf '%-13s base=%-8s drafter=%-8s repair=%-8s %s\n' "$m" "$b" "$d" "$r" "$dir"
}

rc=0
for m in "${want[@]}"; do
    model_spec "$m" >/dev/null || die "unknown model '$m' (known: $ALL_MODELS all)"
    m="$(model_canonical "$m")"

    if (( CHECK )); then
        status_line "$m"
        continue
    fi

    dir="$(dest_dir "$m")"
    base="$(model_field "$m" base)"
    base_repo="$(model_field "$m" base_repo)"
    drafter="$(model_field "$m" dspark_src)"
    drafter_repo="$(model_field "$m" dspark_repo)"

    note "$m -> $dir"
    if (( ! REPAIR_ONLY )); then
        mkdir -p "$dir" || die "cannot create $dir"
        if have_file "$dir" "$base_repo" "$base"; then
            note "  base already present"
        else
            download "$dir" "$base_repo" "$base"
        fi
        if have_file "$dir" "$drafter_repo" "$drafter"; then
            note "  drafter already present"
        else
            download "$dir" "$drafter_repo" "$drafter"
        fi
    fi

    if (( REPAIR )); then
        if ensure_drafter "$dir" "$m"; then
            :
        else
            note "  no DSpark repair applies (drafter is already ds4-native)"
        fi
    fi

    if [ -f "$dir/$base" ] && [ -f "$dir/$(model_field "$m" dspark)" ]; then
        note "  ready: ./start.sh --model $m --restart"
    else
        note "  INCOMPLETE — see above"
        rc=1
    fi
done

exit $rc

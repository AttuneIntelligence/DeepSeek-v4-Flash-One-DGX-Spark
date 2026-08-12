#!/usr/bin/env bash
#
# tools/models.sh — the weight table, shared by start.sh and fetch-weights.sh.
#
# Sourced, never executed. Defines:
#
#   model_spec <name>      -> sub|base|base_repo|dspark|dspark_src|dspark_repo
#   model_field <name> <k> -> one field by name (sub, base, base_repo, ...)
#   resolve_dir <name>     -> the directory holding a usable pair, or nonzero
#   have_drafter <dir> <n> -> true if the repaired drafter is there or buildable
#   ensure_drafter <dir> <n> -> build the repaired drafter from the shipped one
#
# Callers may pre-set MODELS_ROOT, LOCAL_GGUF_DIR and SOURCE; defaults match
# start.sh. `die` and `note` are used if the caller defines them.

# shellcheck disable=SC2034
MODELS_ROOT="${MODELS_ROOT:-$HOME/Empress/models}"
LOCAL_GGUF_DIR="${LOCAL_GGUF_DIR:-$HOME/gguf}"
SOURCE="${SOURCE:-auto}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPAIR_TOOL="${REPAIR_TOOL:-$REPO_ROOT/tools/gguf_dspark_remap.py}"

if ! declare -F die >/dev/null; then
    die() { echo "models.sh: $*" >&2; exit 1; }
fi
if ! declare -F note >/dev/null; then
    note() { echo "==> $*"; }
fi

# ---------------------------------------------------------------- model table
# Fields:
#   sub          subdirectory under $MODELS_ROOT
#   base         target GGUF, as served
#   base_repo    Hugging Face repo the target comes from
#   dspark       DSpark drafter as SERVED (may be a repaired name)
#   dspark_src   DSpark drafter as PUBLISHED (what we download)
#   dspark_repo  Hugging Face repo the drafter comes from
#
# stock's drafter is published in ds4-native `dspark.*` form, so dspark ==
# dspark_src and no repair runs. The abliterated build ships its drafter under
# the legacy `mtp.*` names, so it is downloaded as `-DSpark-support.gguf` and
# repaired into `-DSpark-support-ds4.gguf` before it is ever served. See
# tools/gguf_dspark_remap.py and docs/dspark-drafter-repair.md.
model_spec() {
    case "$1" in
        stock|base|antirez|iq2xxs) cat <<'SPEC'
DeepSeek-v4-Flash|DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf|antirez/deepseek-v4-gguf|DSpark-drafter-Q2K-Q8-0731.gguf|DSpark-drafter-Q2K-Q8-0731.gguf|bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF
SPEC
            ;;
        abliterated|ablit|headroom128|hr128) cat <<'SPEC'
DeepSeek-V4-Flash-0731-Abliterated-DS4-Headroom128|DeepSeek-V4-Flash-0731-Abliterated-DS4-Headroom128.gguf|apetersson/DeepSeek-V4-Flash-0731-Abliterated-DS4-Headroom128|DeepSeek-V4-Flash-0731-Abliterated-DS4-Headroom128-DSpark-support-ds4.gguf|DeepSeek-V4-Flash-0731-Abliterated-DS4-Headroom128-DSpark-support.gguf|apetersson/DeepSeek-V4-Flash-0731-Abliterated-DS4-Headroom128
SPEC
            ;;
        *) return 1 ;;
    esac
}
ALL_MODELS="stock abliterated"

# Canonical name for a model alias ("hr128" -> "abliterated"), for messages.
model_canonical() {
    local want; want="$(model_spec "$1")" || return 1
    local m
    for m in $ALL_MODELS; do
        [ "$(model_spec "$m")" = "$want" ] && { echo "$m"; return 0; }
    done
    echo "$1"
}

model_field() {
    local spec sub base base_repo dspark dspark_src dspark_repo
    spec="$(model_spec "$1")" || return 1
    IFS='|' read -r sub base base_repo dspark dspark_src dspark_repo <<<"$spec"
    case "$2" in
        sub)         echo "$sub" ;;
        base)        echo "$base" ;;
        base_repo)   echo "$base_repo" ;;
        dspark)      echo "$dspark" ;;
        dspark_src)  echo "$dspark_src" ;;
        dspark_repo) echo "$dspark_repo" ;;
        *) die "model_field: unknown field '$2'" ;;
    esac
}

# ------------------------------------------------------- DSpark drafter repair
# The abliterated build ships its three-stage DSpark support model under
# upstream's legacy `mtp.*` tensor names. ds4-server's DSpark loader wants
# `dspark.*` and aborts at startup with
#
#     ds4: required tensor is missing: dspark.main_proj.weight
#
# The files are structurally identical (81 tensors, same shapes, three stages
# against target layers 40/41/42), so the repair is a header rewrite with the
# payload copied verbatim -- no requantization, no reordering.

# True if <dir> can serve this model's drafter, possibly after a repair.
have_drafter() {
    local dir="$1" model="$2"
    [ -f "$dir/$(model_field "$model" dspark)" ] && return 0
    [ -f "$dir/$(model_field "$model" dspark_src)" ] && return 0
    return 1
}

# Build the repaired drafter in <dir> if only the published one is present.
# Returns 0 if the served drafter exists afterwards.
ensure_drafter() {
    local dir="$1" model="$2" dspark src
    dspark="$(model_field "$model" dspark)"
    src="$(model_field "$model" dspark_src)"
    [ -f "$dir/$dspark" ] && return 0
    [ "$dspark" = "$src" ] && return 1          # nothing to repair from
    [ -f "$dir/$src" ] || return 1
    [ -x "$REPAIR_TOOL" ] || die "need $REPAIR_TOOL to repair $src"
    note "repairing DSpark drafter (legacy mtp.* -> ds4 dspark.*)"
    note "  in  $dir/$src"
    note "  out $dir/$dspark"
    "$REPAIR_TOOL" "$dir/$src" "$dir/$dspark" >/dev/null \
        || die "DSpark repair failed; run $REPAIR_TOOL by hand to see why"
    "$REPAIR_TOOL" --verify "$dir/$dspark" >/dev/null \
        || die "repaired drafter did not verify against ds4's DSpark schema"
    note "repaired drafter written and verified"
}

# Directory that holds a usable pair for <model>, honouring $SOURCE.
resolve_dir() {
    local model="$1" nfs loc have_nfs=0 have_loc=0 base
    base="$(model_field "$model" base)" || return 1
    nfs="$MODELS_ROOT/$(model_field "$model" sub)"
    loc="$LOCAL_GGUF_DIR"
    [ -f "$nfs/$base" ] && have_drafter "$nfs" "$model" && have_nfs=1
    [ -f "$loc/$base" ] && have_drafter "$loc" "$model" && have_loc=1
    case "$SOURCE" in
        local)   (( have_loc )) && { echo "$loc"; return 0; }; return 1 ;;
        empress) (( have_nfs )) && { echo "$nfs"; return 0; }; return 1 ;;
        auto)    (( have_loc )) && { echo "$loc"; return 0; }
                 (( have_nfs )) && { echo "$nfs"; return 0; }; return 1 ;;
        *) die "SOURCE must be auto, empress or local (got '$SOURCE')" ;;
    esac
}

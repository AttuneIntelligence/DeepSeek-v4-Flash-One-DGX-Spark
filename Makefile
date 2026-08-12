# DeepSeek-V4-Flash on one DGX Spark
#
# Everything in this repo has a target here. The scripts remain usable on their
# own -- the Makefile is a table of contents, not a wrapper layer.
#
#   make bootstrap                 engine + both weight sets + repair + verify
#   make weights MODEL=abliterated fetch one set and make it servable
#   make serve MODEL=abliterated   restart the server on that set
#   make bench                     full benchmark sweep against the live server
#
# MODEL defaults to `abliterated`, matching start.sh. Pass MODEL=stock for the
# standard 0731 weights.

SHELL      := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

MODEL      ?= abliterated
EVAL_MODELS ?= abliterated stock
PORT       ?= 8888
CTX        ?= 1000000
HOST       ?= 0.0.0.0
DEST       ?= local
URL        ?= http://127.0.0.1:$(PORT)
LOG        ?= $(HOME)/ds4-server.log
RESULTS    ?= bench/results
RESTART_DEPTH ?= 131072
FANOUT_DEPTH  ?= 131072
FANOUT_AGENTS ?= 1,2,4
EVAL_QUESTIONS ?= 16
EVAL_TOKENS    ?= 1536
EVAL_SEED      ?= 1
DS4_SRC       ?= $(HOME)/code/ds4
KV_DISK_DIR ?= $(HOME)/ds4-kv
KV_DISK_MB  ?= 65536
STAMP      := $(shell date +%Y%m%d-%H%M%S)

START      := ./start.sh
STOP       := ./stop.sh
FETCH      := tools/fetch-weights.sh
REMAP      := tools/gguf_dspark_remap.py
BENCH      := tools/bench.py
KVSTATUS   := tools/kv-status.sh
PLOT       := tools/plot_bench.py
INSTALLER  := https://github.com/Entrpi/ds4-on-spark/raw/main/install.sh

# Resolve model paths through the same table the scripts use.
model_field = $(shell . tools/models.sh >/dev/null 2>&1; . tools/models.sh; model_field $(MODEL) $(1))

.PHONY: help bootstrap engine weights weights-all weights-check localize \
        repair verify serve start stop restart status logs list dry-run \
        bench bench-quick bench-depth bench-concurrency bench-needle bench-cache \
        bench-restart bench-fanout eval kv-status kv-watch plot results lint clean-results

## ---------------------------------------------------------------- meta

help: ## Show this help
	@echo "DeepSeek-V4-Flash on one DGX Spark"
	@echo
	@echo "Targets:"
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-20s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Variables: MODEL=$(MODEL) PORT=$(PORT) CTX=$(CTX) DEST=$(DEST)"
	@echo "Models:    abliterated (default) stock   (aliases: ablit hr128 base ...)"

## ---------------------------------------------------------------- setup

bootstrap: engine weights-all verify ## Engine, both weight sets, repair and verify
	@echo
	@echo "Bootstrap complete. Serve with:"
	@echo "    make serve MODEL=stock"
	@echo "    make serve MODEL=abliterated"

engine: ## Build and install ds4-serve (upstream installer, no weight download)
	@if command -v ds4-serve >/dev/null 2>&1 || [ -x "$$HOME/.local/bin/ds4-serve" ]; then \
	    echo "==> ds4-serve already installed ($$(command -v ds4-serve || echo $$HOME/.local/bin/ds4-serve))"; \
	else \
	    echo "==> fetching and running the ds4-on-spark installer (no weights)"; \
	    tmp=$$(mktemp); curl -fsSL $(INSTALLER) -o $$tmp; \
	    bash $$tmp --no-download --no-smoke; rm -f $$tmp; \
	fi

weights: ## Fetch MODEL's weights and make them servable (DEST=local|empress)
	$(FETCH) --dest $(DEST) $(MODEL)

weights-all: ## Fetch every weight set
	$(FETCH) --dest $(DEST) all

weights-check: ## Report which weights are present, without downloading
	@$(FETCH) --check --dest $(DEST) all

localize: ## Copy MODEL's weights from the NAS to the local disk (faster boots)
	@src=$$(MODELS_ROOT=$${MODELS_ROOT:-$$HOME/Empress/models} SOURCE=empress \
	        bash -c '. tools/models.sh; resolve_dir $(MODEL)') || \
	    { echo "make: $(MODEL) is not on the NAS"; exit 1; }; \
	dst=$${LOCAL_GGUF_DIR:-$$HOME/gguf}; mkdir -p "$$dst"; \
	echo "==> rsync $$src -> $$dst"; \
	rsync -ah --info=progress2 --inplace \
	    "$$src/$(call model_field,base)" \
	    "$$src/$(call model_field,dspark)" "$$dst/"

## ---------------------------------------------------------------- drafter

repair: ## Rebuild MODEL's ds4-native DSpark drafter from the published one
	$(FETCH) --repair-only --dest $(DEST) $(MODEL)

verify: ## Check every present DSpark drafter against ds4's schema
	@. tools/models.sh; ok=1; \
	for m in $$ALL_MODELS; do \
	    if d=$$(resolve_dir $$m); then \
	        f="$$d/$$(model_field $$m dspark)"; \
	        if [ -f "$$f" ]; then \
	            echo "== $$m"; \
	            $(REMAP) --verify "$$f" || ok=0; \
	        else \
	            echo "== $$m: drafter not built yet (make repair MODEL=$$m)"; ok=0; \
	        fi; \
	    else \
	        echo "== $$m: weights missing (make weights MODEL=$$m)"; \
	    fi; \
	done; [ $$ok = 1 ]

## ---------------------------------------------------------------- serving

serve: start ## Alias for start

start: ## Serve MODEL on :$(PORT) at CTX tokens (restarts if something is up)
	$(START) --model $(MODEL) --port $(PORT) --ctx $(CTX) --host $(HOST) --restart

stop: ## Stop the server
	PORT=$(PORT) $(STOP)

restart: start ## Alias for start

dry-run: ## Print the exact exec line start.sh would use
	@$(START) --model $(MODEL) --port $(PORT) --ctx $(CTX) --dry-run

status: ## Show what is serving right now
	@curl -sf $(URL)/v1/models >/dev/null 2>&1 || { echo "nothing serving on $(URL)"; exit 1; }
	@# /v1/stats content-negotiates: without this accept header it answers in a
	@# plain-text format, which is why an earlier version of this target printed
	@# nothing at all. artifact_source must read "built" -- a boot that misses the
	@# derived artifacts runs ~30% slower with no other log tell.
	@curl -sf -H 'accept: application/json' $(URL)/v1/stats | python3 -c '\
import json,sys; d=json.load(sys.stdin); \
sv=d.get("server",{}); s2=d.get("serving",{}); c=d.get("cache",{}); sp=d.get("speculation",{}); rd=d.get("route_decisions",{}); \
print("model    : %s  ctx=%s  max_seq=%s" % (sv.get("model"), sv.get("context"), sv.get("max_seq"))); \
print("uptime   : %ss  inflight=%s" % (sv.get("uptime_seconds"), sv.get("requests_inflight"))); \
print("artifacts: %s (%s derived)%s" % (sv.get("artifact_source"), sv.get("derived_artifacts"), "" if sv.get("artifact_source")=="built" else "   <-- EXPECTED built")); \
print("banks    : %s live of %s   kv_pages=%s" % (c.get("banks_live"), c.get("banks_total"), c.get("kv_pages_resident"))); \
print("admits   : cold=%s warm=%s fork=%s  rejects=%s" % (c.get("admits_cold"), c.get("admits_warm"), c.get("admits_fork"), c.get("cont_admit_rejects"))); \
print("prefill  : %s tok/s (60s)   decode: %s tok/s (60s)" % (c.get("prefill_tok_s_60s"), s2.get("decode_tok_s_60s"))); \
print("spec     : accept=%s drafts=%s hits=%s quench=%s  tok/step=%s" % (sp.get("spec_accept_ratio"), sp.get("spec_drafts"), sp.get("spec_hits"), sp.get("spec_quench_events"), s2.get("tok_per_step_60s"))); \
print("routing  : cont=%s serial=%s deep_serial_503=%s  (serial should stay near 0)" % (rd.get("continuous"), s2.get("requests_serial"), s2.get("requests_refused_deep_serial")))' 2>/dev/null || \
	  curl -sf -H 'accept: application/json' $(URL)/v1/stats | python3 -m json.tool

list: ## Show the weight table and where each set resolves
	@$(START) --list

logs: ## Tail the server log
	@tail -f $(LOG)

## ---------------------------------------------------------------- benchmarks

bench: ## Full sweep against the live server; writes bench/results/
	@mkdir -p $(RESULTS)
	$(BENCH) all --url $(URL) --model $(MODEL) \
	    --out $(RESULTS)/$(MODEL)-$(STAMP).jsonl \
	    --markdown $(RESULTS)/$(MODEL)-$(STAMP).md

bench-quick: ## Short sweep (shallow depths, fewer tokens)
	@mkdir -p $(RESULTS)
	$(BENCH) all --quick --url $(URL) --model $(MODEL) \
	    --out $(RESULTS)/$(MODEL)-quick-$(STAMP).jsonl

bench-depth: ## Decode/prefill/TTFT vs context depth
	$(BENCH) depth --url $(URL) --model $(MODEL)

bench-concurrency: ## Aggregate throughput vs concurrent streams
	$(BENCH) concurrency --url $(URL) --model $(MODEL)

bench-needle: ## Retrieval accuracy at depth
	$(BENCH) needle --url $(URL) --model $(MODEL)

bench-cache: ## Cold vs warm prefill on the same deep prompt
	$(BENCH) cache --url $(URL) --model $(MODEL)

bench-restart: ## Does a deep prompt survive a server restart? (RESTARTS the server)
	@mkdir -p $(RESULTS)
	$(BENCH) restart --depth $(RESTART_DEPTH) --url $(URL) --model $(MODEL) \
	    --restart-cmd "$(START) --restart --model $(MODEL) --port $(PORT) --ctx $(CTX)" \
	    --out $(RESULTS)/$(MODEL)-restart.jsonl

bench-fanout: ## N agents against ONE shared deep context (the coding-agent shape)
	@mkdir -p $(RESULTS)
	$(BENCH) fanout --depth $(FANOUT_DEPTH) --streams $(FANOUT_AGENTS) \
	    --url $(URL) --model $(MODEL) --out $(RESULTS)/$(MODEL)-fanout.jsonl

eval: ## Capability eval (GPQA/SuperGPQA/AIME) -- STOPS the server, needs the GPU
	@mkdir -p $(RESULTS)/eval
	@command -v $(DS4_SRC)/ds4-eval >/dev/null 2>&1 || [ -x $(DS4_SRC)/ds4-eval ] || \
	    { echo "make: $(DS4_SRC)/ds4-eval not found (set DS4_SRC)"; exit 1; }
	@echo "==> stopping the server; ds4-eval needs exclusive GPU + memory"
	@-PORT=$(PORT) $(STOP) >/dev/null 2>&1
	@for i in $$(seq 1 60); do pgrep -x ds4-server >/dev/null || break; sleep 2; done
	@. tools/models.sh; for m in $(EVAL_MODELS); do \
	    d=$$(resolve_dir $$m) || { echo "== $$m: weights missing"; continue; }; \
	    echo "== $$m ($(EVAL_QUESTIONS) questions, -n $(EVAL_TOKENS), seed $(EVAL_SEED))"; \
	    $(DS4_SRC)/ds4-eval -m "$$d/$$(model_field $$m base)" --cuda \
	        --questions $(EVAL_QUESTIONS) -n $(EVAL_TOKENS) --temp 0 --seed $(EVAL_SEED) \
	        --trace $(RESULTS)/eval/$$m-trace.txt 2>&1 | tail -$$(( $(EVAL_QUESTIONS) + 2 )); \
	done
	@echo "==> traces in $(RESULTS)/eval/ -- restart with: make serve"

kv-status: ## Disk KV tier: usage, budget pressure, restores, admissions
	@KV_DISK_DIR=$(KV_DISK_DIR) KV_DISK_MB=$(KV_DISK_MB) PORT=$(PORT) $(KVSTATUS)

kv-watch: ## kv-status, refreshing every 30s
	@KV_DISK_DIR=$(KV_DISK_DIR) KV_DISK_MB=$(KV_DISK_MB) PORT=$(PORT) $(KVSTATUS) --watch

plot: ## Redraw bench.png from the recorded results
	$(PLOT) --results $(RESULTS) --model $(MODEL) --out bench.png

results: ## List recorded benchmark runs
	@ls -1t $(RESULTS)/*.md $(RESULTS)/*.jsonl 2>/dev/null || echo "no results yet"

clean-results: ## Delete recorded benchmark runs
	rm -f $(RESULTS)/*.jsonl $(RESULTS)/*.md

## ---------------------------------------------------------------- checks

lint: ## Syntax-check the shell and python in this repo
	@for f in start.sh stop.sh tools/*.sh; do bash -n "$$f" && echo "ok   $$f"; done
	@for f in tools/*.py; do python3 -m py_compile "$$f" && echo "ok   $$f"; done
	@command -v shellcheck >/dev/null 2>&1 && shellcheck -S warning start.sh stop.sh tools/*.sh || true

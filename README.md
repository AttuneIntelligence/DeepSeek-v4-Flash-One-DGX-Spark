# ds4f

Thin, idempotent launcher scripts for running the **DeepSeek-V4-Flash** server built for the NVIDIA DGX Spark (GB10 / SM121) — the DwarfStar 4 (C/CUDA) engine that serves an OpenAI-compatible `/v1` API on `:8888`.

This is based on **antirez/ds4** (DwarfStar 4) and its DGX Spark fork:

- [antirez/ds4](https://github.com/antirez/ds4) — upstream DwarfStar 4 engine (MIT-licensed, C/CUDA)
- [Entrpi/ds4-on-spark](https://github.com/Entrpi/ds4-on-spark) — DGX Spark one-command install, benchmarks, and roofline analysis (what `start.sh` pulls)
- [Entrpi/ds4 (batched-serving)](https://github.com/Entrpi/ds4/tree/batched-serving) — the DGX-Spark-optimized CUDA perf fork used here

> **Note:** this repo does **not** use vLLM. `ds4-server` exposes the same `/v1` API that `vllm serve` does, but vLLM cannot read this repo's asymmetric GGUF, so the repo ships its own server. These scripts are just thin wrappers over that server's official installer.

## Requirements

- A NVIDIA DGX Spark (GB10 / SM121)
- Bash
- `curl`
- Disk space for the ~110 GiB GGUF weight set
- `ss` (for `stop.sh`)

## Background

**DwarfStar 4** is a small, self-contained native inference engine optimized for DeepSeek V4 Flash, written by Salvatore Sanfilippo ([antirez](https://x.com/antirez)) — deliberately narrow, not a generic GGUF runner. [Entrpi](https://github.com/Entrpi) maintains a DGX-Spark-optimized CUDA perf fork of it, plus the [ds4-on-spark](https://github.com/Entrpi/ds4-on-spark) installer this repo wraps, which serves DeepSeek-V4-Flash entirely on-device on a GB10 / SM121 DGX Spark (RTX PRO 6000 / 5090-class `sm_120` also builds).

Thanks to Bleys Goodson ([@bleysg on X](https://x.com/bleysg)).

## Quick start

```bash
./start.sh    # full DSpark stack on :8888
```

First run does the heavy lifting: clones and builds the pinned fork, downloads the ~110 GiB GGUF set, smoke-tests, installs `ds4-serve`, and starts the server on `:8888`. Later runs fast-forward the clone to the pinned tag, skip GGUFs already on disk, and just start the server.

## Usage

```bash
PORT=8889 ./start.sh            # different port
CTX=262144 ./start.sh           # smaller context budget  (KV ≈ 9.5 KiB/token)
./start.sh --no-dspark          # plain continuous decode (passes through)
```

Environment variables (all optional):

| Variable       | Default    | Meaning                                   |
|----------------|------------|-------------------------------------------|
| `PORT`         | `8888`     | Server port                               |
| `CTX`          | `262144`   | Context budget (KV ≈ 9.5 KiB/token)       |
| `DS4_SRC_DIR`  | `~/code/ds4` | Source directory for the pinned clone   |
| `DS4_GGUF_DIR` | `~/gguf`     | Weights directory                       |

## Stopping and restarting

```bash
./stop.sh                     # stop server on :8888 (wait until port is freed)
PORT=8889 ./stop.sh           # stop server on a different port

pkill -x ds4-server; ./start.sh   # force a clean restart
```

`start.sh` is idempotent: if the server is already answering on the port, it reports the running model and exits without touching anything.

## Checking status

```bash
curl http://127.0.0.1:8888/v1/models
```

## Logs

Server logs go to `~/ds4-server.log`. Check there if the server doesn't come up:

```
!!! Not reachable yet — check $HOME/ds4-server.log
```

## Files

| File         | Purpose                                     |
|--------------|---------------------------------------------|
| `start.sh`   | Fetch installer, build, download weights, serve on `:8888` |
| `stop.sh`    | Stop the `ds4-server` process              |

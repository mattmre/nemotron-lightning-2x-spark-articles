# NVIDIA Nemotron 3.5 Lightning 30B-A3B NVFP4 + DSpark Speculative Decoding

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia's AI Lab</a></sub>
  <br><br>
  <a href="https://ko-fi.com/Z8Z3SPLOD" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://storage.ko-fi.com/cdn/kofi6.png?v=6" alt="Buy Me a Coffee at ko-fi.com" height="28" style="height:28px;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;vertical-align:middle;border:0;" /></a>
</p>

Run **NVIDIA Nemotron 3.5 Lightning 30B-A3B NVFP4** — a 30B-parameter, ~3B-active MoE reasoning model with a native **1M-token context** — on a **DGX Spark (GB10)**, **RTX 5090**, or **RTX 6000 PRO** using a single Docker command. This repo wires up [SGLang](https://docs.sglang.io/) with **DSpark speculative decoding** (NVIDIA's own DSpark draft model) and serves it behind an OpenAI-compatible API.

No source builds, no pip installs on the host: everything runs in a prebuilt container. The scripts handle model caching, GPU/container-runtime checks, and memory sizing automatically.

---

## Table of Contents

- [Features](#features)
- [Hardware & Prerequisites](#hardware--prerequisites)
- [Quick Start](#quick-start)
- [Server Management](#server-management)
- [Configuration](#configuration)
- [KV Cache, Concurrency & Tuning](#kv-cache-concurrency--tuning)
- [GPU Memory Quick Reference](#gpu-memory-quick-reference)
- [Usage (OpenAI-compatible API)](#usage-openai-compatible-api)
- [Troubleshooting](#troubleshooting)
- [Repository Structure](#repository-structure)
- [Technical Details](#technical-details)
- [Credits & License](#credits--license)

---

## Features

- **NVFP4 quantization (modelopt)** — 30B parameters with a small memory footprint; ~21 GiB total including the draft model
- **DSpark speculative decoding** — NVIDIA's DSpark draft model accelerates token generation vs. standard decoding
- **1M-token context** — `max_position_embeddings = 1,048,576`, made affordable by the hybrid architecture (see [KV Cache](#kv-cache-concurrency--tuning))
- **Hybrid architecture** — 52 layers: 23 Mamba-2 SSM layers + 23 MoE layers + 6 attention layers; only attention pays a growing K/V cost
- **OpenAI-compatible API** — drop-in client compatibility (OpenAI SDK, curl, or any OpenAI-compatible tool)
- **Docker-based deployment** — no source build; prebuilt SGLang image
- **Self-managing setup** — prerequisite checks, model caching, and memory sizing in one script

## Hardware & Prerequisites

| Requirement | Notes |
|---|---|
| GPU | NVIDIA **GB10 / SM121 (DGX Spark)**, **RTX 5090**, or **RTX 6000 PRO** (Ada or Blackwell) |
| Docker | installed and running |
| NVIDIA driver | `nvidia-smi` must work on the host |
| NVIDIA container runtime | `nvidia-container-toolkit` (needed for `docker run --gpus all`) |
| Host RAM | ≥ 80 GiB required; **~119 GiB recommended** (DGX Spark class) for the default `MEM_FRACTION_STATIC=0.78` |
| Free disk | ≥ 30 GiB for the Docker image + two weight sets (NVFP4 + DSpark draft) |

The script hard-fails below 80 GiB of host RAM and warns below 110 GiB, guiding you to lower `MEM_FRACTION_STATIC` (see [KV Cache & Tuning](#kv-cache-concurrency--tuning)).

> **Consumer GPUs (5090 / 6000 PRO):** the default memory tuning targets the DGX Spark's 128 GiB unified memory. On smaller cards you must override `MEM_FRACTION_STATIC` — see the [GPU Memory Quick Reference](#gpu-memory-quick-reference).

## Quick Start

```bash
git clone https://github.com/MiaAI-Lab/Nemotron3.5-Lightning-DGX-Spark-RTX-5090-6000-PRO.git
cd Nemotron3.5-Lightning-DGX-Spark-RTX-5090-6000-PRO

chmod +x start.sh
./start.sh
```

`start.sh` will:
1. Verify prerequisites (Docker, NVIDIA driver, container runtime, RAM, disk, free port)
2. Download the model + DSpark draft into `~/.cache/huggingface` (skipped if already cached)
3. Pull the prebuilt SGLang Docker image
4. Launch the container with DSpark speculative decoding and wait for readiness

Download models only (no server), useful for pre-staging on large files:

```bash
./start.sh --download-only
```

Once ready (`SGLang is ready!`), verify:

```bash
curl http://127.0.0.1:8888/v1/models
```

The OpenAI-compatible endpoint is `http://127.0.0.1:8888/v1` (bound on all interfaces by default).

## Server Management

```bash
# Show flags and all tunables
./start.sh --help

# Stop and remove the server + clean the PID file
./stop.sh
```

| File | Purpose |
|---|---|
| `start.sh` | Launch (idempotent: re-running replaces the old container) |
| `stop.sh` | Stop/remove the container and clean up `.sglang.pid` |
| `.sglang.log` | Server log (also streamed live during startup) |
| `.sglang.pid` | Container ID of the running instance |
| `.vllm-cache/` | Cached kernels (FlashInfer, Torch, vLLM) |

## Configuration

All settings are environment variables — set them **before** running `./start.sh`. Changing them requires a restart.

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8888` | Server port |
| `HOST` | `0.0.0.0` | Bind address |
| `MEM_FRACTION_STATIC` | `0.78` | SGLang static memory fraction (KV cache size + concurrency lever) |
| `CUDA_GRAPH_MAX_BS_DECODE` | `4` | CUDA-graph max batch size for decode |
| `DSPARK_BLOCK_SIZE` | `3` | DSpark draft block size |
| `HF_TOKEN` | *(empty)* | Hugging Face token for gated models |
| `HF_HOME` | `~/.cache/huggingface` | Model cache directory |
| `IMAGE` | `lmsysorg/sglang:dev-nemotron3-5-lightning` | Docker image |

## KV Cache, Concurrency & Tuning

Serving capacity is a **startup-time trade-off, not a fixed constant** — every number below is changeable. Values were measured on a running **DGX Spark (GB10)** with the stock settings (`MEM_FRACTION_STATIC=0.78`, `CUDA_GRAPH_MAX_BS_DECODE=4`, `DSPARK_BLOCK_SIZE=3`), so they reflect that machine, not a promise from any GPU:

| Value | Default (DGX Spark / GB10, mem-fraction 0.78) |
|---|---|
| **KV / token pool** | **~4,926,000 tokens (~4.93M total)** |
| KV cache memory | ~14.1 GiB |
| Weights (NVFP4 + DSpark draft) | ~21.0 GiB |
| Mamba/SSM cache | 716 MiB (float16) |
| **Max running requests (concurrency)** | **48** |
| Max input length (context) | ~1,048,570 tokens (~1M per request) |

**Why a 30B model fits a 1M window in ~14 GiB:** the model is hybrid — of its 52 layers, only 6 are full attention layers (23 Mamba-2 SSM + 23 MoE + 6 attention). Attention layers pay a per-token K/V cost that grows with sequence length; the Mamba state is a fixed 716 MiB. The token pool counts the shared paged cache across all layers, so the ~4.93M budget covers the 6 attention layers' K/V plus the Mamba cache for **all concurrent requests combined** — you get roughly 4–5 simultaneous full 1M-token contexts, or far more shorter ones.

**You can change all of this — and on smaller GPUs you must.** Set env vars before `./start.sh` (restart required):

```bash
# Shrink the footprint on a smaller GPU (see the 5090 note below):
MEM_FRACTION_STATIC=0.40 CUDA_GRAPH_MAX_BS_DECODE=2 ./start.sh

# Grow cache/concurrency on a memory-rich machine:
MEM_FRACTION_STATIC=0.85 ./start.sh
```

- **`MEM_FRACTION_STATIC`** (default `0.78`) — the **GPU memory utilization knob**: the fraction of device memory reserved for the static pool (weights + KV cache + CUDA-graph scratch). **This is the main lever for both KV cache size and concurrency** — the token pool and `max_running_requests` scale roughly with it. `0.78` is tuned for the 128 GiB unified-memory GB10; lower it to reduce memory usage.
- **`CUDA_GRAPH_MAX_BS_DECODE`** (default `4`) — CUDA-graph batch-size cap for decode. Lowering it saves graph-scratch memory and is also a decode-concurrency cap.
- **`DSPARK_BLOCK_SIZE`** (default `3`) — DSpark draft block size (performance knob, not memory).

`max_running_requests` (48) is derived from the cache budget at startup, so it follows `MEM_FRACTION_STATIC` automatically. To cap it explicitly, add SGLang's `--max-running-requests N` to the `sglang serve` command in `start.sh`.

**Inspect what the running server actually allocated** (tokens + GiB + concurrency):

```bash
curl -s http://127.0.0.1:8888/get_server_info | python3 -c \
"import sys,json; d=json.load(sys.stdin); m=d['internal_states'][0]['memory_usage'];\
print('pool tokens:', m['token_capacity']); print('kv cache GiB:', m['kvcache']);\
print('weights GiB:', m['weight']); print('max running:', d['max_running_requests'])"
```

## GPU Memory Quick Reference

The same `MEM_FRACTION_STATIC` means a fraction of *that GPU's* memory, so the stock `0.78` does not transfer across cards. Use this as a starting point, then confirm with the report command above.

| GPU | Memory | Suggested `MEM_FRACTION_STATIC` | Expected result |
|---|---|---|---|
| DGX Spark (GB10) | 128 GiB unified | **0.78** (stock) | ~4.93M token pool, ~48 concurrent — the reference configuration |
| RTX 5090 | 32 GiB GDDR7 | **0.30–0.45** | Weights (~16 GiB) + scaled cache; lower pool (roughly 100K–1M tokens), low single-digit concurrency |
| RTX 6000 PRO / RTX 6000 Ada | 48 GiB GDDR6 | **0.45–0.55** | Mid-sized pool; concurrency in mid-single digits — check live numbers |
| RTX PRO 6000 Blackwell | 96 GiB GDDR7 | **0.60–0.70** | Larger pool, modest concurrency |

Keep `CUDA_GRAPH_MAX_BS_DECODE` at `2` on the 5090/48-GiB cards to trim CUDA-graph scratch memory.

> **RTX 5090 — the default settings will NOT work as-is.** With `MEM_FRACTION_STATIC=0.78`, SGLang tries to hold the ~21 GiB of weights + CUDA-graph scratch *and* reserve 78% of a 32 GiB heap. It will OOM at startup or leave a near-zero KV cache. You **must** set a lower value — start with `MEM_FRACTION_STATIC=0.40 CUDA_GRAPH_MAX_BS_DECODE=2 ./start.sh`, watch the startup log, and use the report command to see the real numbers instead of guessing.

## Usage (OpenAI-compatible API)

### Python (OpenAI SDK)

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8888/v1", api_key="dummy")

response = client.chat.completions.create(
    model="nemotron-3.5-lightning",
    messages=[{"role": "user", "content": "Write a short haiku about GPUs."}],
    temperature=0.6,
    top_p=0.95,
    top_k=20,
)

print(response.choices[0].message.content)
```

`model` is free-form on a single-model server — any name is accepted.

### curl

```bash
curl http://127.0.0.1:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron-3.5-lightning",
    "messages": [{"role": "user", "content": "Hello! Who are you?"}],
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20
  }'
```

### Recommended client parameters

Tuned for strong reasoning quality on this model:

| Parameter | Value |
|---|---|
| `temperature` | `0.6` |
| `top_p` | `0.95` |
| `top_k` | `20` |
| `repetition_penalty` | `1.08` |
| `chat_template_kwargs` | `{"enable_thinking": true}` (reasoning/thinking traces) |

Set `"stream": true` in the request to enable token streaming (SGLang supports OpenAI-style SSE streaming).

**Tool calling:** the server runs SGLang's `qwen3_coder` tool-call parser — models may emit structured tool calls compatible with that format. For agentic use, prefer `enable_thinking` for the reasoning trace and feed the final answer back to your orchestrator.

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `FATAL: host has only X GiB RAM` | Start-fail below 80 GiB host RAM. On smaller hosts, you cannot run the stock config. Lower `MEM_FRACTION_STATIC` (but the model still needs ~21 GiB for weights — plan for a small cache). |
| **OOM / server exits on RTX 5090** | Default `MEM_FRACTION_STATIC=0.78` assumes 128 GiB unified memory. Run with `MEM_FRACTION_STATIC=0.40 CUDA_GRAPH_MAX_BS_DECODE=2`. See [GPU Memory Quick Reference](#gpu-memory-quick-reference). |
| `FATAL: port 8888 is already in use` | Another server is running. Use `PORT=8889 ./start.sh` or stop the existing one. |
| `FATAL: NVIDIA container runtime not found` | Install `nvidia-container-toolkit` and restart Docker. |
| Model download fails / 401 | The model is public (no auth required). A 401 usually means a stale/proxied `HF_HOME` or an outbound-auth environment; set `HF_TOKEN=... ./start.sh` for private mirrors or enterprise proxies. |
| Long prompts are slow to start generating | Expected: prompts prefill in 8K-token chunks (`chunked_prefill_size 8192`). First-token time grows with prompt size; generation after prefill is fast. |
| How do I see current memory/concurrency? | Use the `get_server_info` report shown in [KV Cache & Tuning](#kv-cache-concurrency--tuning). |

## Repository Structure

```
.
├── start.sh              # Launch script: checks, caching, Docker, wait-for-ready
├── stop.sh               # Stop and remove the SGLang container
├── README.md             # This file
├── LICENSE               # MIT License
├── .gitignore            # Ignores logs, PID, and .vllm-cache
├── .sglang.log           # Server log (gitignored)
├── .sglang.pid           # Container ID (gitignored)
└── .vllm-cache/          # Cached kernels: FlashInfer, Torch, vLLM (gitignored)
```

## Technical Details

- **Model**: [`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4)
  - 30B total / ~3B active MoE; 52 layers = 23 Mamba-2 + 23 MoE + 6 attention
  - `hidden_size` 2688, `max_position_embeddings` 1,048,576 (1M context)
  - quantized with NVIDIA ModelOpt **NVFP4** (`modelopt_mixed`)
- **Draft model**: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark`
- **Image**: `lmsysorg/sglang:dev-nemotron3-5-lightning`

**SGLang serve invocation** (what `start.sh` runs inside the container):

```text
--mamba-ssm-dtype float16
--mem-fraction-static 0.78
--cuda-graph-max-bs-decode 4
--reasoning-parser nemotron_3
--tool-call-parser qwen3_coder
--speculative-algorithm DSPARK
--speculative-draft-model-path .../NVFP4-DSpark/snapshots/<hash>
--speculative-dspark-block-size 3
--host 0.0.0.0
--port 8888
```

**Container**: runs with `--network host`, 32 GiB shared memory, `IPC_LOCK`, and `--gpus all`.

## Credits & License

- Recipe & guidance: [SGLang cookbook — Nemotron 3.5 Lightning](https://docs.sglang.io/cookbook/nemotron3-5-lightning)
- Model: NVIDIA Nemotron 3.5 Lightning 30B-A3B
- DSpark speculative decoding: SGLang project
- Setup & scripts: Mia's AI Lab ([@MiaAI_lab](https://x.com/MiaAI_lab))

Licensed under the [MIT License](LICENSE). The bundled model weights are subject to NVIDIA's original model license.

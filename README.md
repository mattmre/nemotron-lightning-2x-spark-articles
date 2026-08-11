# NVIDIA Nemotron 3.5 Lightning 30B-A3B NVFP4 + DSpark Speculative Decoding

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia's AI Lab</a></sub>
  <br><br>
  <a href="https://ko-fi.com/Z8Z3SPLOD" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://storage.ko-fi.com/cdn/kofi6.png?v=6" alt="Buy Me a Coffee at ko-fi.com" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

This repository provides a complete setup for running **NVIDIA Nemotron 3.5 Lightning 30B-A3B NVFP4** with **DSpark speculative decoding** on **DGX Spark** platforms (GB10 / SM121), and also on **RTX 5090** and **RTX 6000 PRO** (Ada or Blackwell).

## Features

- **NVFP4 quantization** — 30B parameter model with reduced memory footprint
- **DSpark speculative decoding** — Significant speedup over standard decoding
- **Docker-based deployment** — No source build required; uses prebuilt SGLang image
- **OpenAI-compatible API** — Easy integration with any OpenAI-compatible client
- **Comprehensive environment management** — Automatic model caching, memory configuration, and prerequisite checks
- **DGX Spark optimized** — Tuned for GB10/SM121 unified memory architecture

## Prerequisites

- **Docker** installed and running
- **NVIDIA driver** on the host (nvidia-smi must work)
- **NVIDIA container runtime** (nvidia-container-toolkit)
- **≥ 80 GiB host RAM** (119 GiB recommended for full configuration)
- **≥ 30 GiB free disk space** for model weights and Docker image
- **GPU**: NVIDIA GB10 / SM121 (DGX Spark), RTX 5090, RTX 6000 PRO (Ada or Blackwell)

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/nemotron-3.5-lightning-dspark.git
cd nemotron-3.5-lightning-dspark
```

### 2. Run the setup script

```bash
chmod +x start.sh
./start.sh
```

The script will:
- Verify all prerequisites
- Download the model (if not already cached)
- Pull the SGLang Docker image
- Start the server with DSpark speculative decoding

### 3. Access the API

Once the server is ready, the OpenAI-compatible endpoint will be available:

```
http://127.0.0.1:8888/v1
```

Or if bound to all interfaces:

```
http://0.0.0.0:8888/v1
```

## Usage

### Basic inference

```python
import openai

client = OpenAI(base_url="http://127.0.0.1:8888/v1", api_key="dummy")

response = client.chat.completions.create(
    model="nemotron-3.5-lightning",
    messages=[{"role": "user", "content": "Hello, how are you?"}],
    temperature=0.6,
    top_p=0.95,
    top_k=20,
)

print(response.choices[0].message.content)
```

### Recommended client parameters

Based on `~/.pi/agent/models.json` (`nemotron35-lightning-local` config):

- `temperature=0.6`
- `top_p=0.95`
- `top_k=20`
- `repetition_penalty=1.08`
- `chat_template_kwargs: {"enable_thinking": true}` (for reasoning models)

**Base URL**: `http://127.0.0.1:8888/v1`

## Help

```bash
./start.sh --help
```

### Download-only mode

Download models without starting the server:

```bash
./start.sh --download-only
```

## Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8888` | Server port |
| `HOST` | `0.0.0.0` | Bind address |
| `MEM_FRACTION_STATIC` | `0.78` | SGLang static memory fraction |
| `CUDA_GRAPH_MAX_BS_DECODE` | `4` | CUDA graph max batch size for decode |
| `DSPARK_BLOCK_SIZE` | `3` | DSpark draft block size |
| `HF_TOKEN` | `` | Hugging Face token for gated models |
| `IMAGE` | `lmsysorg/sglang:dev-nemotron3-5-lightning` | Docker image |

### Memory planning

The script auto-detects host RAM and warns if below DGX Spark class (~119 GiB). For systems with less memory, consider reducing `MEM_FRACTION_STATIC`.

### KV cache, concurrency & tuning

Serving capacity is a **startup-time trade-off, not a fixed constant** — and all the numbers below are *changeable*. They were measured on a running DGX Spark (GB10) with the stock settings (`MEM_FRACTION_STATIC=0.78`, `CUDA_GRAPH_MAX_BS_DECODE=4`, `DSPARK_BLOCK_SIZE=3`), so they reflect *this machine*, not a promise from the GPU:

| Value | Default (DGX Spark / GB10, mem-fraction 0.78) |
|---|---|
| **KV / token pool** | **~4,926,000 tokens (~4.93M total)** |
| KV cache memory | ~14.1 GiB |
| Weights (NVFP4 + DSpark draft) | ~21.0 GiB |
| Mamba/SSM cache | 716 MiB (float16) |
| **Max running requests (concurrency)** | **48** |
| Max input length (context) | ~1,048,570 tokens (~1M per request) |

A note on why this works: the model is a **Mamba-2 + attention hybrid** (23 Mamba layers, only 6 attention layers), so only the attention layers pay a per-token K/V cost — the Mamba state is a fixed 716 MiB. That is how a 30B model fits a 1M-token window in ~14 GiB of cache.

**You can change all of this — you often must on smaller GPUs.** Every knob is an environment variable set *before* `./start.sh` (a restart is required):

```bash
# Shrink the footprint on a GPU with less memory (see RTX 5090 note below):
MEM_FRACTION_STATIC=0.40 CUDA_GRAPH_MAX_BS_DECODE=2 ./start.sh

# Or grow cache/concurrency on a memory-rich machine:
MEM_FRACTION_STATIC=0.85 ./start.sh
```

- **`MEM_FRACTION_STATIC`** (default `0.78`) — the **GPU memory utilization knob**: the fraction of device memory reserved for the static pool (weights + KV cache + CUDA-graph scratch). **This is the main lever for both KV cache size and concurrency** — the token pool and `max_running_requests` scale roughly with it. `0.78` is tuned for the 128 GiB unified-memory GB10; lower it to cut memory usage.
- **`CUDA_GRAPH_MAX_BS_DECODE`** (default `4`) — CUDA-graph batch-size cap for decode. Lowering it saves graph-scratch memory and is also a decode-concurrency cap.
- **`DSPARK_BLOCK_SIZE`** (default `3`) — DSpark draft block size (performance knob, not memory).

`max_running_requests` (48) is derived from the cache budget at startup, so it follows `MEM_FRACTION_STATIC` automatically. To cap it explicitly, add SGLang's `--max-running-requests N` to the `sglang serve` command in `start.sh`. To see what the running server actually allocated, query its own report:

```bash
curl -s http://127.0.0.1:8888/get_server_info | python3 -c \
"import sys,json; d=json.load(sys.stdin); m=d['internal_states'][0]['memory_usage'];\
print('pool tokens:', m['token_capacity']); print('kv cache GiB:', m['kvcache']);\
print('weights GiB:', m['weight']); print('max running:', d['max_running_requests'])"
```

> **⚠️ RTX 5090 — the default settings will NOT work as-is.** The defaults assume a DGX Spark's 128 GiB unified memory. The RTX 5090 has only **32 GiB GDDR7**, and the RTX 6000 PRO cards have 48–96 GiB depending on variant. With `MEM_FRACTION_STATIC=0.78`, SGLang tries to hold the ~21 GiB of weights + CUDA-graph scratch *and* reserve 78% of a 32 GiB heap — it will either OOM at startup or leave a near-zero KV cache. **You must lower `MEM_FRACTION_STATIC`** (start around `0.30`–`0.45`) and keep `CUDA_GRAPH_MAX_BS_DECODE` low (e.g. `2`). Expect a far smaller KV pool and much lower concurrency than the DGX Spark table above — on a 32 GiB card the ~16 GiB of weights leaves only single-digit GiB for the cache, so plan for roughly ~100K–1M tokens of pool and low single-digit concurrency. Start small, watch the startup log, and use the report command above to see the real numbers instead of guessing. RTX 6000 PRO (48/96 GiB) lands in between: scale `MEM_FRACTION_STATIC` accordingly and check the report.

## Repository Structure

```
.
├── start.sh              # Main setup and launch script
├── .sglang.log           # SGLang server log file
├── .sglang.pid           # Container PID file
├── .vllm-cache/          # Cached dependencies (flashinfer, torch, vllm)
└── README.md             # This file
```

## Technical Details

- **Model**: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`
- **Draft model**: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark`
- **SGLang arguments**:
  - `--speculative-algorithm DSPARK`
  - `--speculative-draft-model-path` — path to DSpark draft model
  - `--speculative-dspark-block-size` — block size for DSpark (default: 3)
  - `--mem-fraction-static 0.78` — static memory fraction
  - `--cuda-graph-max-bs-decode 4` — max batch size for CUDA graphs
  - `--reasoning-parser nemotron_3` — reasoning parser
  - `--tool-call-parser qwen3_coder` — tool call parser

## Credits

- Recipe and guidance: [SGLang cookbook - Nemotron 3.5 Lightning](https://docs.sglang.io/cookbook/nemotron3-5-lightning)
- Model: NVIDIA Nemotron 3.5 Lightning 30B-A3B
- DSpark speculative decoding: SGLang project

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NVIDIA for the Nemotron model family
- SGLang team for the DSpark speculative decoding and cookbook recipes
- DGX Spark platform for the unified memory architecture

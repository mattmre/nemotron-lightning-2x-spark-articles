# NVIDIA Nemotron 3.5 Lightning 30B-A3B NVFP4 + DSpark Speculative Decoding

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-GPU-orange.svg)](https://nvidia.com)

This repository provides a complete setup for running **NVIDIA Nemotron 3.5 Lightning 30B-A3B NVFP4** with **DSpark speculative decoding** on **DGX Spark** platforms (GB10 / SM121), and also on **RTX 5090** and **RTX 6000 PRO** (Ada or Blackwell).

## Features

- **NVFP4 quantization** — 30B parameter model with reduced memory footprint
- **DSpark speculative decoding** — Significant speedup over standard decoding
- **Docker-based deployment** — No source build required; uses prebuilt SGLang image
- **OpenAI-compatible API** — Easy integration with any OpenAI-compatible client
- **Comprehensive environment management** — Automatic model caching, memory configuration, and prerequisite checks
- **DGX Spark optimized** — Tuned for GB10/SM121 unified memory architecture

## 📋 Prerequisites

- **Docker** installed and running
- **NVIDIA driver** on the host (nvidia-smi must work)
- **NVIDIA container runtime** (nvidia-container-toolkit)
- **≥ 80 GiB host RAM** (119 GiB recommended for full configuration)
- **≥ 30 GiB free disk space** for model weights and Docker image
- **GPU**: NVIDIA GB10 / SM121 (DGX Spark), RTX 5090, RTX 6000 PRO (Ada or Blackwell)

## 🛠️ Installation & Setup

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

## 💻 Usage

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

### Help

```bash
./start.sh --help
```

### Download-only mode

Download models without starting the server:

```bash
./start.sh --download-only
```

## ⚙️ Configuration

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

## 📁 Repository Structure

```
.
├── start.sh              # Main setup and launch script
├── .sglang.log           # SGLang server log file
├── .sglang.pid           # Container PID file
├── .vllm-cache/          # Cached dependencies (flashinfer, torch, vllm)
└── README.md             # This file
```

## 🏗️ Technical Details

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

## 🤝 Credits

- Recipe and guidance: [SGLang cookbook - Nemotron 3.5 Lightning](https://docs.sglang.io/cookbook/nemotron3-5-lightning)
- Model: NVIDIA Nemotron 3.5 Lightning 30B-A3B
- DSpark speculative decoding: SGLang project

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NVIDIA for the Nemotron model family
- SGLang team for the DSpark speculative decoding and cookbook recipes
- DGX Spark platform for the unified memory architecture
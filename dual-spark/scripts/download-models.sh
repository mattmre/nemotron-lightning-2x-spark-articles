#!/usr/bin/env bash
set -euo pipefail
: "${HF_HUB_CACHE:?Export the same absolute cache path used in dual-spark/.env}"
command -v hf >/dev/null || { echo 'Install the Hugging Face CLI first (pipx install huggingface-hub).' >&2; exit 1; }
hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --revision cc84af2fe71647d87f4486c064f320e1e7535243
hf download nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark --revision 8a0177116d138011e63103110f136ec0ca09ebbf

# Nemotron Lightning on two Sparks: short articles and comments at scale

A public snapshot of a working **two-replica vLLM deployment** for processing many small articles and comments. Each GB10 Spark runs the complete NVIDIA Nemotron 3.5 Lightning 30B-A3B NVFP4 model with the official DSpark drafter. nginx distributes requests between independent TP=1 replicas; this is not a tensor-parallel two-GPU model.

**For an AI agent deploying this recipe, start with [SETUP.md](SETUP.md).** It covers configuration, both nodes, health verification, and rollback. [AGENTS.md](AGENTS.md) gives repository-specific instructions.

## Saved operating point

| Setting | Head | Worker |
|---|---|---|
| Hardware | GB10, 128 GB nominal unified RAM | Same |
| Engine | `vllm/vllm-openai:v0.27.1` | Same |
| Target revision | `cc84af2fe71647d87f4486c064f320e1e7535243` | Same |
| DSpark revision | `8a0177116d138011e63103110f136ec0ca09ebbf` | Same |
| Speculative tokens | 3 | 3 |
| MoE / Mamba | Marlin / FlashInfer, aligned Mamba cache | Same |
| KV cache / prefix caching | FP8 / enabled | Same |
| GPU memory utilization budget | 0.45 | 0.45 |
| Maximum context | 32,768 tokens | Same |
| Maximum concurrent sequences | 10 | 12 |
| Maximum batched tokens | 8,192 | Same |
| Execution | Eager (`--enforce-eager`) | Same |
| Backend host port | 8801 | 8890 |

The memory fraction is an allocation budget, not a GPU compute-utilization target. This configuration shares machines with other workloads. It intentionally preserves the observed settings rather than claiming maximum hardware throughput. On GB10, NVIDIA's documented Marlin path uses W4A16 for the NVFP4 experts.

## Observed throughput, not a benchmark guarantee

A simultaneous 30-second passive sample on 2026-10-16 measured **212 output tokens/s on the head and 321 on the worker, about 533 combined**, during existing mixed traffic. At sample end there were 7/12 running requests, 0/2 waiting, and 4.8%/8.3% KV-cache occupancy; no cache preemptions occurred during the sample. Request lengths, output budgets, and workload mix were not controlled. These are aggregate output rates, not single-request speed or requests per second. No article/comment content, raw operational logs, or private addresses are included.

This is an engine recipe, not the original application pipeline. Use your own queue and bounded client concurrency. Short outputs, long article inputs, structured JSON, and reasoning budgets can produce very different rates. [The passive sampler](dual-spark/scripts/sample-metrics.py) measures both engines without sending inference requests.

## Attribution and lineage

- **NVIDIA** created the model, NVFP4 checkpoint, and DSpark drafter. The saved vLLM flags follow NVIDIA's [official model-card Spark recipe](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4#1x-dgx-spark-gb10) and [vLLM cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Lightning/vllm_cookbook.ipynb), with local memory/context/concurrency and eager-mode changes.
- **[MiaAI Lab](https://github.com/MiaAI-Lab)** maintains the [upstream Lightning recipe](https://github.com/MiaAI-Lab/Nemotron3.5-Lightning-DGX-Spark-RTX-5090-6000-PRO). This repository is a GitHub fork of that project at `38c16a403ea57f26f1828fa572270245ae51f01d`. Its original SGLang scripts and [README](upstream/README.md) are preserved. Local commissioning records identify NVIDIA's vLLM recipe as the deployed basis; they do **not** establish that MiaAI's SGLang launcher was used for this vLLM deployment. Fork ancestry and deployed-engine provenance are distinct.
- **[vLLM contributors](https://github.com/vllm-project/vllm)** provide the serving engine; **[FlashInfer contributors](https://github.com/flashinfer-ai/flashinfer)** and **[Marlin contributors](https://github.com/IST-DASLab/marlin)** provide supporting kernel work; **[nginx contributors](https://nginx.org/)** provide the load balancer.
- **[SGLang contributors](https://github.com/sgl-project/sglang)** provide the engine used by the preserved upstream recipe. It is not used by the `dual-spark/` deployment.
- **[mattmre](https://github.com/mattmre)** contributes the two-replica short-article/comment operating point, documentation, sanitized deployment files, and passive metric sampler, prepared with AI-agent assistance.

See [CREDITS.md](CREDITS.md). The upstream [MIT license](LICENSE) and copyright notice remain intact. Model weights have their own [OpenMDW-1.1 license](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4); engine and dependency licenses remain with their owners. No weights or container images are redistributed here.

## Files

- `SETUP.md`: agent-readable two-Spark deployment runbook.
- `dual-spark/`: pinned model downloads, replica Compose, nginx Compose/config generator, example environment, and passive metric sampler.
- `start.sh`, `stop.sh`, `upstream/README.md`: preserved upstream **single-node SGLang** workflow. Do not run it for the two-node vLLM setup.

Publication checks validate configuration and scripts without restarting the source deployment. The new packaging is not claimed to have undergone a fresh two-machine deployment test.

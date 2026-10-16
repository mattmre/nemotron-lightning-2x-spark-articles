# Credits and licensing

The root LICENSE is the unmodified MIT notice from MiaAI Lab's upstream repository, whose notice credits NVIDIA Corporation. Preserved upstream Git history records its contributors. The fork point is 38c16a403ea57f26f1828fa572270245ae51f01d.

MiaAI Lab: https://github.com/MiaAI-Lab/Nemotron3.5-Lightning-DGX-Spark-RTX-5090-6000-PRO

NVIDIA model and deployment recipe: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4

NVIDIA DSpark drafter: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark

NVIDIA cookbook: https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3.5-Lightning/vllm_cookbook.ipynb

vLLM: https://github.com/vllm-project/vllm

FlashInfer: https://github.com/flashinfer-ai/flashinfer

Marlin: https://github.com/IST-DASLab/marlin

nginx: https://nginx.org/

SGLang (preserved upstream single-node workflow): https://github.com/sgl-project/sglang

Two-Spark adaptation and documentation: https://github.com/mattmre

The running vLLM configuration was reconstructed from read-only process inspection on both owned machines. Local commissioning evidence names NVIDIA's official vLLM recipe as its basis. MiaAI's upstream uses SGLang; no unsupported claim of direct launcher derivation is made. Original single-node SGLang files remain available for attribution and reference. New two-node packaging was prepared with AI-agent assistance.

No model, engine, kernel, or upstream recipe authorship is claimed by the fork maintainer. Each dependency and model retains its own license. Fork additions are distributed under the repository's MIT terms.

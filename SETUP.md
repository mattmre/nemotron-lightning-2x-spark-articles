# Agent setup: two Sparks processing small articles and comments at scale

Give an AI agent this repository and say: “Deploy the saved two-Spark vLLM recipe in SETUP.md using these two SSH aliases.” The agent should execute this runbook, substitute local values, and report host acceptance results. Do not run root `start.sh`: it is the preserved single-node SGLang recipe.

## 1. Establish the two machines and inspect first

Obtain the head and worker SSH aliases from the operator or existing authorized SSH configuration. Discover the worker's private interconnect IPv4 address from those machines; do not guess addresses or publish them. Only request missing connection information if it cannot be discovered locally. Use existing SSH keys; never commit them or print credentials.

On both nodes inspect `hostname`, `nvidia-smi`, `free -h`, `df -h`, `docker ps`, `docker compose version`, and `ss -ltn`. Requirements: Linux ARM64 GB10 Spark-class hardware with approximately 128 GB unified memory, working NVIDIA Container Toolkit, Docker Compose with `gpus: all` support, Python 3, curl, Git, and the Hugging Face `hf` CLI. The source deployment used driver 580.173.02 with CUDA 13.0 compatibility. Reserve enough disk for both model snapshots and the image; inspect actual download sizes and leave operating headroom.

Check existing GPU workloads and memory pressure. Do not automatically stop them, kill processes, change firewall policy, or overwrite an existing deployment. If ports 8801/8890/8888 are occupied, identify their owners and plan a coordinated migration or an isolated trial with changed ports throughout the nginx configuration. A new Compose project does not isolate GPU memory or avoid occupied ports.

Use two independent TP=1 replicas. No Ray, NCCL, tensor parallelism, or shared model filesystem is required. Each node needs a complete local target and draft snapshot. The head must reach worker TCP 8890 over a private network. Restrict that backend to the head through the site's existing network policy. The published nginx endpoint binds loopback by default; use an authenticated private access layer for remote clients.

## 2. Clone on each Spark

```bash
git clone https://github.com/mattmre/nemotron-lightning-2x-spark-articles.git
cd nemotron-lightning-2x-spark-articles
cp dual-spark/.env.example dual-spark/.env
```

Edit the ignored `dual-spark/.env` separately on each node:

| Variable | Head | Worker |
|---|---|---|
| `HF_HUB_CACHE` | Absolute local path to HF hub cache | Absolute local path on worker |
| `BIND_IP` | `127.0.0.1` | Worker's private interconnect IPv4 |
| `BACKEND_PORT` | `8801` | `8890` |
| `MAX_NUM_SEQS` | `10` | `12` |
| `WORKER_HOST` | Worker private IPv4 or DNS name | Unused |

Use a writable cache for downloading; Compose mounts it read-only into the engine. Replace `/path/to/huggingface/hub` with the real absolute path, often `$HOME/.cache/huggingface/hub` expanded to an absolute path. Do not use an unexpanded shell variable inside the example value. Do not copy the head's cache path blindly to another user or node.

## 3. Download exact snapshots on both nodes

Export the same cache path chosen above in each node's shell. If authentication is needed, use the local HF login/token mechanism, never a committed token or command-line token literal.

```bash
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub" # match your local .env
bash dual-spark/scripts/download-models.sh
docker pull vllm/vllm-openai:v0.27.1
```

The downloader pins target revision `cc84af2fe71647d87f4486c064f320e1e7535243` and DSpark revision `8a0177116d138011e63103110f136ec0ca09ebbf`. Inspect CLI errors and confirm both downloads finish. A config file alone does not prove all weight shards exist. Containers run offline, so incomplete snapshots will fail startup. Image version is pinned by release tag, not immutable registry digest; record the pulled digest locally if stricter artifact reproducibility is needed.

## 4. Validate and start replicas

From the repository root on **each** node:

```bash
docker compose --env-file dual-spark/.env -f dual-spark/replica.compose.yaml config --quiet
docker compose --env-file dual-spark/.env -f dual-spark/replica.compose.yaml up -d --wait --wait-timeout 1200
```

Do not proceed on failed startup. Inspect local engine logs using:

```bash
docker compose --env-file dual-spark/.env -f dual-spark/replica.compose.yaml logs --tail 100 replica
```

Keep logs private. Check the resolved flags against the README table, especially eager mode, 0.45 memory budget, context 32768, sequence caps 10/12, and DSpark depth 3. The healthcheck calls `/health`; model loading may take several minutes. Do not increase memory allocation as a generic fix without checking other unified-memory users.

## 5. Render and start nginx on the head

The worker must be reachable from the head before starting nginx. Replace `YOUR_PRIVATE_WORKER_HOST` below with the same private address configured above. The generator accepts an explicit argument rather than sourcing a shell environment file. Generated topology is ignored by Git.

```bash
python3 dual-spark/scripts/render-nginx.py --worker-host YOUR_PRIVATE_WORKER_HOST
curl --fail --max-time 10 http://127.0.0.1:8801/health
curl --fail --max-time 10 http://YOUR_PRIVATE_WORKER_HOST:8890/health
docker compose -f dual-spark/lb.compose.yaml run --rm --no-deps lb nginx -t
docker compose -f dual-spark/lb.compose.yaml up -d
curl --fail --max-time 10 http://127.0.0.1:8888/v1/models
```

nginx uses `least_conn`, upstream keepalive, disabled response buffering, a 64 MB body limit, and 900-second read/send timeouts, matching the saved topology. The saved private deployment additionally listened on a local LAN address; this public recipe deliberately defaults to loopback. If you intentionally add a private LAN listener, put your local address only in the ignored generated file. Never expose these unauthenticated endpoints on the public internet.

Engine metrics are explicit: `/engine/head/metrics` and `/engine/worker/metrics`. Querying plain `/metrics` through a balanced location returns one selected engine, not the pair's aggregate.

## 6. Host acceptance checks

Run from the head, substituting the worker address locally:

1. Both direct `/health` endpoints return HTTP 200.
2. Direct `/v1/models` on each engine and the nginx endpoint identify `nemotron-lightning`.
3. Both explicit engine metric endpoints return the generation token counter.
4. Send the synthetic request below **once to each direct backend**, then once through nginx. All three must return HTTP 200 and a nonempty assistant response. This proves both replicas can generate; nginx alone could succeed with only one healthy backend.
5. Inspect private logs for startup failures, engine errors, and memory pressure; verify both containers remain healthy.

```bash
# On the head; repeat with worker:8890 and then 127.0.0.1:8888.
curl --fail --max-time 120 http://127.0.0.1:8801/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nemotron-lightning","messages":[{"role":"user","content":"Summarize this comment in one sentence: The new bus route makes my commute shorter."}],"max_tokens":128,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

The example uses synthetic content. A short output budget with reasoning disabled avoids spending the budget on a reasoning trace. Validate the behavior of any additional reasoning/template controls with the pinned engine before using them in your application.

## 7. Connect a bounded article/comment queue

API base on head: `http://127.0.0.1:8888/v1`; model alias: `nemotron-lightning`. Each request should carry one short article/comment or an intentionally sized batch. Use the application's task prompt, token budget, and structured-output schema. Keep input plus requested output within the configured 32768-token context. Application queueing, deduplication, retries, and storage are not included in this engine recipe.

Start with bounded total client concurrency, e.g. 8, then measure 12/16/22 if latency and memory allow. These are experiment points, not throughput guarantees. Engine caps sum to 22 running sequences, but connections, queued requests, and sequences are different quantities; `least_conn` is not a global inference scheduler. Audit callers that use direct backends if one node queues while the other has spare capacity. Use bounded retry backoff and application-level request IDs to handle ambiguous timeouts without duplicate downstream work.

While real work is running:

```bash
python3 dual-spark/scripts/sample-metrics.py --seconds 30
```

Report per-engine and combined output TPS, completed documents/sec, output lengths, time-to-first-token, total latency, queue depth, failures, and memory pressure. A model serving comparison requires controlled prompt/output distributions and repeated trials. Do not describe the saved 533 combined TPS observation as 533 TPS per user or guaranteed capacity.

## 8. Stop or roll back only this deployment

On head, stop its nginx project first if removing the complete deployment. On each node, stop its replica project:

```bash
# Head only:
docker compose -f dual-spark/lb.compose.yaml down
# Each node:
docker compose --env-file dual-spark/.env -f dual-spark/replica.compose.yaml down
```

These commands act only on this recipe's Compose projects. Restore any prior service using its own recorded launcher/configuration and verify health before switching clients back. Do not delete model caches, volumes, or unrelated containers. If this was an isolated trial, retain existing service configuration throughout.

## Agent completion report

Provide the public repository URL and commit, head/worker health PASS/FAIL, direct and balanced synthetic generation PASS/FAIL, model identity, observed concurrency/memory settings, and any limitations. Give private addresses only in the operator's private session when needed, never in public issues or commits. Say whether deployment was actually performed; configuration validation alone is not a completed deployment.

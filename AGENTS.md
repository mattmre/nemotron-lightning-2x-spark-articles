# Agent instructions

Read SETUP.md before deploying. The dual-spark directory is the two-node vLLM recipe. Root start.sh and stop.sh belong to the preserved upstream SGLang workflow.

Use the committed operating point by default: two TP=1 replicas, DSpark depth 3, eager mode, memory budget 0.45, context 32768, sequence caps 10/12. Treat performance changes as separate measured experiments.

Never copy Docker inspect output, shell history, SSH configs/keys, tokens, user content, environment files, hostnames, personal paths, or private addresses into commits. .env.example is synthetic. Keep filled-in .env files local and ignored. Commit only explicit file paths. Use a GitHub noreply identity for publication commits. Review the staged diff and scan for credentials and private machine identifiers before pushing.

Do not stop unrelated containers, kill occupied-port processes, or replace an existing production deployment implicitly. Inspect both machines first. Use separate project names and free ports for a trial; coordinated migration requires the operator's intended scope. Do not assume an existing launcher reflects running arguments.

Keep upstream copyright notices, fork history, and attribution intact. Describe authorized auth/network/filesystem checks as defensive regression validation or host acceptance checks.

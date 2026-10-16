#!/usr/bin/env python3
"""Passively sample aggregate output TPS on each replica through nginx."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
import time
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--base-url', default='http://127.0.0.1:8888')
p.add_argument('--seconds', type=float, default=30)
a = p.parse_args()
if not 1 <= a.seconds <= 3600:
    p.error('--seconds must be between 1 and 3600')

def sample(role):
    with urllib.request.urlopen(a.base_url.rstrip('/') + '/engine/' + role + '/metrics', timeout=10) as r:
        lines = r.read().decode().splitlines()
    values = {}
    for line in lines:
        m = re.match(r'^(vllm:[A-Za-z0-9_]+)(?:\{[^}]*\})?\s+(\S+)', line)
        if m:
            values[m[1]] = values.get(m[1], 0) + float(m[2])
    if 'vllm:generation_tokens_total' not in values:
        raise RuntimeError('Missing generation token counter')
    return role, (time.monotonic(), values)

roles = ('head', 'worker')
with ThreadPoolExecutor(2) as pool:
    before = dict(pool.map(sample, roles))
time.sleep(a.seconds)
with ThreadPoolExecutor(2) as pool:
    after = dict(pool.map(sample, roles))
result = {}
for role in roles:
    t0, x = before[role]
    t1, y = after[role]
    delta = y['vllm:generation_tokens_total'] - x['vllm:generation_tokens_total']
    if delta < 0:
        raise RuntimeError('Counter reset; repeat after server restart settles')
    result[role] = {
        'output_tokens_per_second': round(delta / (t1 - t0), 2),
        'sample_seconds': round(t1 - t0, 2),
        'running_at_end': y.get('vllm:num_requests_running'),
        'waiting_at_end': y.get('vllm:num_requests_waiting'),
        'kv_cache_fraction_at_end': y.get('vllm:kv_cache_usage_perc'),
    }
result['combined_output_tokens_per_second'] = round(sum(result[r]['output_tokens_per_second'] for r in roles), 2)
print(json.dumps(result, indent=2))

#!/usr/bin/env python3
"""Render a private-backend nginx config; no requests or deployment changes."""
import argparse
import ipaddress
import re
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--worker-host', required=True)
p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'nginx.generated.conf')
a = p.parse_args()
host = a.worker_host
try:
    addr = ipaddress.ip_address(host)
    if addr.version != 4 or not addr.is_private or addr.is_unspecified or addr.is_loopback or addr.is_link_local or addr.is_multicast:
        p.error('Use a private, reachable worker IPv4 address or private DNS name.')
except ValueError:
    if len(host) > 253 or not all(re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?', part) for part in host.split('.')):
        p.error('Invalid worker DNS name.')
    if host.endswith('.invalid') or host.lower() == 'localhost':
        p.error('Replace the example with your private worker hostname.')
a.output.write_text("""# Generated locally; do not commit private topology.
worker_processes 2;
worker_rlimit_nofile 65536;
events { worker_connections 8192; }
http {
  access_log off;
  upstream nemotron {
    least_conn;
    server 127.0.0.1:8801 max_fails=3 fail_timeout=30s;
    server WORKER_HOST:8890 max_fails=3 fail_timeout=30s;
    keepalive 512;
  }
  server {
    listen 127.0.0.1:8888;
    client_max_body_size 64m;
    proxy_read_timeout 900s;
    proxy_send_timeout 900s;
    proxy_connect_timeout 10s;
    proxy_buffering off;
    location = /engine/head/metrics {
      proxy_pass http://127.0.0.1:8801/metrics;
      proxy_connect_timeout 3s;
      proxy_read_timeout 5s;
    }
    location = /engine/worker/metrics {
      proxy_pass http://WORKER_HOST:8890/metrics;
      proxy_connect_timeout 3s;
      proxy_read_timeout 5s;
    }
    location / {
      proxy_pass http://nemotron;
      proxy_http_version 1.1;
      proxy_set_header Connection "";
      proxy_set_header Host $host;
      proxy_next_upstream error timeout http_502 http_503 http_504;
    }
  }
}
""".replace('WORKER_HOST', host))
print('Wrote local nginx config.')

#!/usr/bin/env python3
"""Export model-aware accelerator allocations from vLLM profiler logs."""

import json
import os
import re
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DOCKER_URL = os.environ.get("DOCKER_PROXY_URL", "http://docker-socket-proxy:2375").rstrip("/")
CONTAINER = os.environ.get("VLLM_CONTAINER", "vllm-engine")
MODEL = os.environ.get("VLLM_MODEL", "unknown")
PORT = int(os.environ.get("MODEL_MEMORY_PORT", "9836"))
GIB = 1024**3
PATTERNS = {
    "weights": re.compile(r"Model loading took ([0-9.]+) GiB memory"),
    "kv_cache": re.compile(r"Available KV cache memory: ([0-9.]+) GiB"),
    "cuda_graphs": re.compile(r"Graph capturing finished.* took ([0-9.]+) GiB"),
}


def docker_get(path):
    with urllib.request.urlopen(f"{DOCKER_URL}{path}", timeout=30) as response:
        return response.read()


def decode_docker_log(payload):
    """Decode Docker's multiplexed log framing, falling back to plain text."""
    chunks = []
    offset = 0
    while offset + 8 <= len(payload) and payload[offset] in (0, 1, 2):
        size = int.from_bytes(payload[offset + 4 : offset + 8], "big")
        end = offset + 8 + size
        if end > len(payload):
            break
        chunks.append(payload[offset + 8 : end])
        offset = end
    if chunks and offset == len(payload):
        payload = b"".join(chunks)
    return payload.decode("utf-8", errors="replace")


def parse_allocations(log_text):
    allocations = {}
    for kind, pattern in PATTERNS.items():
        matches = pattern.findall(log_text)
        if matches:
            allocations[kind] = float(matches[-1]) * GIB
    return allocations


def escape_label(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class Collector:
    def __init__(self):
        self.lock = threading.Lock()
        self.started_at = None
        self.allocations = {}
        self.error = "not collected"

    def refresh(self):
        with self.lock:
            try:
                metadata = json.loads(docker_get(f"/containers/{CONTAINER}/json"))
                started_at = metadata["State"]["StartedAt"]
                if started_at != self.started_at:
                    payload = docker_get(
                        f"/containers/{CONTAINER}/logs?stdout=1&stderr=1&timestamps=0"
                    )
                    allocations = parse_allocations(decode_docker_log(payload))
                    missing = sorted(set(PATTERNS) - set(allocations))
                    if missing:
                        raise RuntimeError(f"profiler values missing: {', '.join(missing)}")
                    self.allocations = allocations
                    self.started_at = started_at
                self.error = ""
            except Exception as exc:
                self.error = str(exc)

    def metrics(self):
        self.refresh()
        with self.lock:
            lines = [
                "# HELP model_memory_exporter_up Whether model allocations were collected successfully.\n",
                "# TYPE model_memory_exporter_up gauge\n",
                f"model_memory_exporter_up {0 if self.error else 1}\n",
                "# HELP model_accelerator_memory_bytes Accelerator memory attributed by the model runtime profiler.\n",
                "# TYPE model_accelerator_memory_bytes gauge\n",
            ]
            for kind, value in sorted(self.allocations.items()):
                lines.append(
                    'model_accelerator_memory_bytes{service="vllm-engine",'
                    f'model="{escape_label(MODEL)}",kind="{kind}"}} {value}\n'
                )
            lines.extend(
                [
                    "# HELP model_accelerator_memory_collection_error_info Last collection error.\n",
                    "# TYPE model_accelerator_memory_collection_error_info gauge\n",
                    f'model_accelerator_memory_collection_error_info{{error="{escape_label(self.error)}"}} 1\n',
                ]
            )
            return "".join(lines).encode()


COLLECTOR = Collector()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            body = b"ok\n"
            status = 200
        elif self.path == "/metrics":
            body = COLLECTOR.metrics()
            status = 200
        else:
            body = b"not found\n"
            status = 404
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string, *args):
        return


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()

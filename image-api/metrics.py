"""Prometheus metrics for image-api: per-route requests, RPC calls to the
sam3/qwen-image-edit workers, and image-edit resource-lease wait/hold time.
Mirrors the prometheus_client convention used by sam3 and qwen-image-edit-worker
(dedicated metrics port via start_http_server, not a FastAPI /metrics route)."""

from prometheus_client import Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1200)

ROUTE_REQUESTS = Counter(
    "image_api_requests_total",
    "Requests handled per route, by outcome",
    ["route", "outcome"],
)
ROUTE_LATENCY = Histogram(
    "image_api_request_duration_seconds",
    "End-to-end request latency per route",
    ["route"],
    buckets=LATENCY_BUCKETS,
)

RPC_REQUESTS = Counter(
    "image_api_rpc_requests_total",
    "Worker RPC calls, by target service and outcome",
    ["service", "operation", "outcome"],
)
RPC_LATENCY = Histogram(
    "image_api_rpc_duration_seconds",
    "Worker RPC call latency, by target service",
    ["service", "operation"],
    buckets=LATENCY_BUCKETS,
)

LEASE_WAIT = Histogram(
    "image_api_lease_wait_seconds",
    "Time spent waiting to acquire the image-edit resource lease",
    buckets=LATENCY_BUCKETS,
)
LEASE_HOLD = Histogram(
    "image_api_lease_hold_seconds",
    "Time the image-edit resource lease was held",
    buckets=LATENCY_BUCKETS,
)
LEASE_ACTIVE = Gauge(
    "image_api_lease_active", "Whether the image-edit resource lease is currently held"
)
LEASE_OUTCOMES = Counter(
    "image_api_lease_outcomes_total",
    "Resource-lease acquisition outcomes",
    ["outcome"],
)

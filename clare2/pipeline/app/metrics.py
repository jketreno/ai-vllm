"""CLARE₂ Prometheus metrics."""

import os

from prometheus_client import Counter, Gauge, Histogram, start_http_server

_started = False


def start_metrics_server(port: int | None = None) -> None:
    global _started
    if _started:
        return
    start_http_server(port or int(os.environ.get("CLARE2_METRICS_PORT", "9091")))
    _started = True


episodes_total = Counter("clare2_episodes_total", "Episode count", ["category"])
themes_active = Gauge("clare2_themes_active", "Active themes", ["category"])
corpus_tokens_total = Gauge("clare2_corpus_tokens_total", "SFT token count")
training_duration_seconds = Gauge("clare2_training_duration_seconds", "Last training duration")
training_loss_final = Gauge("clare2_training_loss_final", "Last final loss")
training_loss_by_epoch = Gauge("clare2_training_loss_by_epoch", "Epoch loss", ["epoch"])
adapter_size_bytes = Gauge("clare2_adapter_size_bytes", "Adapter size")
distillation_patterns_extracted = Counter(
    "clare2_distillation_patterns_extracted", "Distilled patterns", ["category"]
)
distillation_patterns_gated_out = Counter(
    "clare2_distillation_patterns_gated_out", "Patterns rejected by recurrence gate"
)
theme_drift_events = Counter("clare2_theme_drift_events", "Theme drift events")

routing_decisions = Counter("clare2_routing_decisions_total", "Routing decisions", ["rule"])
base_fallbacks = Counter("clare2_base_fallbacks_total", "Base model fallbacks")
active_requests = Gauge("clare2_active_requests", "Tracked in-flight requests")
active_routes = Gauge("clare2_active_routes", "Pinned active routes", ["adapter_id"])
adapter_operations = Counter(
    "clare2_adapter_operations_total", "Adapter operations", ["operation", "outcome"]
)
adapter_operation_latency = Histogram(
    "clare2_adapter_operation_duration_seconds", "Adapter operation latency", ["operation"]
)
adapter_compatibility_failures = Counter(
    "clare2_adapter_compatibility_failures_total", "Adapter compatibility failures"
)
registry_reconciliation_errors = Counter(
    "clare2_registry_reconciliation_errors_total", "Registry reconciliation errors"
)
lifecycle_phase = Gauge("clare2_lifecycle_phase", "Lifecycle phase", ["phase"])
maintenance_mode = Gauge("clare2_maintenance_mode", "Maintenance mode enabled")
maintenance_duration = Histogram(
    "clare2_maintenance_duration_seconds", "Maintenance duration"
)
lifecycle_outcomes = Counter(
    "clare2_lifecycle_outcomes_total", "Lifecycle outcomes", ["outcome"]
)
evaluation_score = Gauge(
    "clare2_evaluation_score", "Candidate and baseline evaluation score", ["adapter_id", "category"]
)
proxy_latency = Histogram("clare2_proxy_duration_seconds", "Policy proxy upstream latency")

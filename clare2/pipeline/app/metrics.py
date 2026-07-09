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
themes_active = Gauge("clare2_themes_active", "Active themes", ["project", "category"])
corpus_tokens_total = Gauge("clare2_corpus_tokens_total", "SFT token count", ["project"])
corpus_sft_pairs = Gauge("clare2_corpus_sft_pairs", "SFT pair count", ["project"])
training_duration_seconds = Gauge(
    "clare2_training_duration_seconds", "Last training duration", ["project"]
)
training_loss_final = Gauge("clare2_training_loss_final", "Last final loss", ["project"])
training_loss_by_epoch = Gauge(
    "clare2_training_loss_by_epoch", "Epoch loss", ["project", "epoch"]
)
adapter_size_bytes = Gauge("clare2_adapter_size_bytes", "Adapter size", ["project"])
distillation_runs = Counter("clare2_distillation_runs_total", "Distillation runs", ["outcome"])
distillation_sessions_pending = Gauge(
    "clare2_distillation_sessions_pending", "Session files pending distillation", ["project"]
)
distillation_sessions_last = Gauge(
    "clare2_distillation_sessions_last", "Session files processed by the last distillation run", ["project"]
)
distillation_sessions = Counter(
    "clare2_distillation_sessions_total", "Distillation session outcomes", ["project", "outcome"]
)
distillation_patterns_extracted_last = Gauge(
    "clare2_distillation_patterns_extracted_last",
    "Patterns accepted by the last distillation run",
    ["project"],
)
distillation_patterns_gated_out_last = Gauge(
    "clare2_distillation_patterns_gated_out_last",
    "Patterns rejected by recurrence gate during the last distillation run",
    ["project"],
)
distillation_patterns_raw = Counter(
    "clare2_distillation_patterns_raw_total", "Raw patterns returned by distillation", ["project"]
)
distillation_parse_errors = Counter(
    "clare2_distillation_parse_errors_total", "Distillation LLM JSON parse errors"
)
distillation_patterns_extracted = Counter(
    "clare2_distillation_patterns_extracted", "Distilled patterns", ["project", "category"]
)
distillation_patterns_gated_out = Counter(
    "clare2_distillation_patterns_gated_out", "Patterns rejected by recurrence gate"
)
distillation_last_run_timestamp = Gauge(
    "clare2_distillation_last_run_timestamp_seconds", "Last distillation run timestamp"
)
summary_runs = Counter(
    "clare2_summary_runs_total", "Summarization runs", ["project", "level", "outcome"]
)
summary_parse_errors = Counter(
    "clare2_summary_parse_errors_total", "Summary LLM JSON parse errors", ["level"]
)
summary_records_input = Gauge(
    "clare2_summary_records_input", "Input records for the last summary run", ["project", "level"]
)
summary_records_output = Gauge(
    "clare2_summary_records_output", "Output records from the last summary run", ["project", "level"]
)
summary_last_run_timestamp = Gauge(
    "clare2_summary_last_run_timestamp_seconds", "Last summary run timestamp", ["project", "level"]
)
structured_output_attempts = Counter(
    "clare2_structured_output_attempts_total",
    "Structured output validation outcomes",
    ["stage", "outcome"],
)
theme_drift_events = Counter("clare2_theme_drift_events", "Theme drift events")
corpus_sync_hosts = Counter(
    "clare2_corpus_sync_hosts_total", "Remote corpus sync outcomes", ["outcome"]
)
corpus_sync_last_run_timestamp = Gauge(
    "clare2_corpus_sync_last_run_timestamp_seconds", "Last remote corpus sync timestamp"
)

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
    "clare2_evaluation_score",
    "Candidate and baseline evaluation score",
    ["adapter_id", "project", "category"],
)
proxy_latency = Histogram("clare2_proxy_duration_seconds", "Policy proxy upstream latency")

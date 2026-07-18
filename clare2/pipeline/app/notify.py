"""Nightly lifecycle run notification emails, sent via the self-hosted mail relay."""

from __future__ import annotations

import json
import logging
import os
import smtplib
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from typing import Any

from . import corpus, metrics

log = logging.getLogger(__name__)

SMTP_HOST = os.environ.get("CLARE2_SMTP_HOST", "192.168.1.78")
SMTP_PORT = int(os.environ.get("CLARE2_SMTP_PORT", "25"))
SMTP_TIMEOUT = float(os.environ.get("CLARE2_SMTP_TIMEOUT", "15"))
NOTIFY_FROM = os.environ.get("CLARE2_NOTIFY_FROM", "james_claude@ketrenos.com")
NOTIFY_TO = os.environ.get("CLARE2_NOTIFY_TO", "james_clare2@ketrenos.com")

OUTCOME_LABELS = {
    "promoted": "PROMOTED",
    "rejected": "REJECTED",
    "skipped_no_new_content": "SKIPPED (no new content)",
    "postponed": "POSTPONED (inference active)",
    "failed": "FAILED",
}


def send_run_notification(outcome: str, **context: Any) -> None:
    """Best-effort notification email for a nightly lifecycle terminal state.

    Never raises: a failed send must not mask or interrupt the lifecycle
    state machine that just recorded the real outcome.
    """
    if not NOTIFY_TO:
        return
    try:
        subject, body = _compose(outcome, context)
        _send(subject, body)
    except Exception:
        log.exception("Failed to send run notification for outcome=%s", outcome)
        metrics.notification_sent.labels(outcome=outcome, status="error").inc()
        return
    metrics.notification_sent.labels(outcome=outcome, status="ok").inc()


def send_batch_run_notification(run_id: str, results: list[dict[str, Any]]) -> None:
    """Best-effort summary email covering every project trained in one nightly run.

    Sent once per run instead of one email per project, so a run that trains
    several projects produces a single report rather than one email per
    project outcome.
    """
    if not NOTIFY_TO:
        return
    outcome = "batch_complete"
    try:
        subject, body = _compose_batch(run_id, results)
        _send(subject, body)
    except Exception:
        log.exception("Failed to send batch run notification for run_id=%s", run_id)
        metrics.notification_sent.labels(outcome=outcome, status="error").inc()
        return
    metrics.notification_sent.labels(outcome=outcome, status="ok").inc()


def _send(subject: str, body: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = NOTIFY_FROM
    msg["To"] = NOTIFY_TO
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid()
    msg.set_content(body)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT) as smtp:
        smtp.ehlo()
        refused = smtp.send_message(msg)
        if refused:
            raise smtplib.SMTPRecipientsRefused(refused)


def _compose(outcome: str, context: dict[str, Any]) -> tuple[str, str]:
    has_project = outcome in ("promoted", "rejected")
    project = context.get("project", "unknown") if has_project else None
    label = OUTCOME_LABELS.get(outcome, outcome.upper())
    subject = f"CLARE₂ Nightly Run — {label}" + (f" ({project})" if project else "")

    lines = [
        "CLARE2 Nightly Run Summary",
        f"Outcome: {label}",
    ]
    if project:
        lines.append(f"Project: {project}")
    lines.append(f"Run ID: {context.get('run_id', 'unknown')}")
    lines.append("")

    lines.append("DISTILLATION")
    lines.extend(_distillation_lines())
    lines.append("")

    if outcome in ("promoted", "rejected"):
        lines.append("TRAINING / EVALUATION")
        lines.extend(_evaluation_lines(context))
    elif outcome == "skipped_no_new_content":
        lines.append("TRAINING")
        summaries = _project_summaries()
        if summaries:
            for skipped_project, summary in summaries.items():
                lines.append(f"  {skipped_project}: {_not_trained_reason(summary)}")
        else:
            lines.append("  No projects were discovered; run skipped.")
    elif outcome == "postponed":
        lines.append("TRAINING")
        lines.append(
            f"  Postponed because {context.get('active_sessions', 'unknown')} "
            "inference request(s) are active."
        )
        lines.append("  Active inference will be checked again every 30 seconds.")
    elif outcome == "failed":
        lines.append("FAILURE")
        lines.append(f"  adapter_id: {context.get('adapter_id') or 'n/a'}")
        lines.append(f"  error: {context.get('error')}")

    return subject, "\n".join(lines) + "\n"


def _compose_batch(run_id: str, results: list[dict[str, Any]]) -> tuple[str, str]:
    result_by_project = {r.get("project", "unknown"): r for r in results}
    project_summaries = _project_summaries(set(result_by_project))
    projects = sorted(result_by_project)
    promoted = sum(1 for r in results if r.get("outcome") == "promoted")
    rejected = sum(1 for r in results if r.get("outcome") == "rejected")
    subject = (
        f"CLARE₂ Nightly Run — {len(results)} project(s) trained "
        f"({promoted} promoted, {rejected} rejected)"
    )

    lines = [
        "CLARE2 Nightly Run Summary",
        f"Run ID: {run_id}",
        f"Projects trained: {', '.join(projects) if projects else 'none'}",
        "",
        "DISTILLATION",
    ]
    lines.extend(_distillation_lines(set(result_by_project)))
    lines.append("")

    for project, summary in project_summaries.items():
        result = result_by_project.get(project)
        if result:
            label = OUTCOME_LABELS.get(result.get("outcome"), str(result.get("outcome")).upper())
            lines.append(f"TRAINING / EVALUATION — {project} ({label})")
            lines.extend(_evaluation_lines(result))
        else:
            lines.append(f"TRAINING / EVALUATION — {project} (NOT TRAINED)")
            lines.append(f"  reason: {_not_trained_reason(summary)}")
        lines.append("")

    return subject, "\n".join(lines) + "\n"


def _distillation_lines(extra_projects: set[str] | None = None) -> list[str]:
    summaries = _project_summaries(extra_projects)
    if not summaries:
        return ["  no projects discovered"]
    lines = []
    for project, summary in summaries.items():
        lines.append(f"  {project}:")
        lines.append(
            "    sessions: "
            f"{summary['session_count']} captured, {summary['processed_count']} processed, "
            f"{summary['pending_count']} pending; latest: {summary['latest_session']}"
        )
        lines.append(f"    last_distillation: {summary['last_distillation']}")
        episodes = summary["episodes"]
        counts = ", ".join(f"{category}: {count}" for category, count in episodes.items())
        lines.append(f"    episode patterns on file: {counts or 'none'}")
        lines.append(
            "    current corpus: "
            f"{summary['sft_pairs']} SFT pair(s), ~{summary['tokens']} tokens; "
            f"last updated: {summary['corpus_updated']}"
        )
    return lines


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _project_summaries(extra_projects: set[str] | None = None) -> dict[str, dict[str, Any]]:
    stats = _load_json(corpus.CORPUS_ROOT / "meta" / "corpus_stats.json", {})
    session_index = _load_json(corpus.CORPUS_ROOT / "meta" / "session_index.json", {})
    projects = set(extra_projects or ()) | set(stats.get("projects", {}))
    for relative in ("sessions", "episodes", "themes/active", "training"):
        root = corpus.CORPUS_ROOT / relative
        if root.exists():
            projects.update(path.name for path in root.iterdir() if path.is_dir())

    indexed: dict[str, list[dict[str, Any]]] = {}
    for record in session_index.get("sessions", []):
        project = record.get("project")
        if project:
            projects.add(project)
            indexed.setdefault(project, []).append(record)

    summaries = {}
    for project in sorted(projects):
        session_files = list((corpus.CORPUS_ROOT / "sessions" / project).glob("**/*.jsonl"))
        processed = indexed.get(project, [])
        session_dates = [record.get("date") for record in processed if record.get("date")]
        for session_file in session_files:
            relative_parts = session_file.relative_to(
                corpus.CORPUS_ROOT / "sessions" / project
            ).parts
            if len(relative_parts) >= 4:
                session_dates.append("-".join(relative_parts[:3]))
        manifest = _load_json(corpus.CORPUS_ROOT / "training" / project / "manifest.json", {})
        project_stats = stats.get("projects", {}).get(project, {})
        summaries[project] = {
            "session_count": len(session_files),
            "processed_count": len(processed),
            "pending_count": max(0, len(session_files) - len(processed)),
            "latest_session": max(session_dates, default="none"),
            "last_distillation": project_stats.get("last_distillation", "never"),
            "episodes": project_stats.get("episodes", {}),
            "sft_pairs": manifest.get("total_sft_pairs", 0),
            "tokens": manifest.get("total_tokens", 0),
            "corpus_updated": manifest.get("last_updated", "never"),
        }
    return summaries


def _not_trained_reason(summary: dict[str, Any]) -> str:
    if summary["session_count"] == 0:
        return "no captured session activity"
    if summary["pending_count"]:
        return f"{summary['pending_count']} captured session(s) remain pending distillation"
    if summary["sft_pairs"] == 0:
        return "no accepted distilled patterns produced an SFT corpus"
    return "current corpus was unchanged or otherwise ineligible for this run"


def _evaluation_lines(context: dict[str, Any]) -> list[str]:
    lines = [
        f"  adapter_id: {context.get('adapter_id', 'unknown')}",
        f"  mlflow_run_id: {context.get('mlflow_run_id') or 'n/a'}",
    ]
    report = context.get("report") or {}
    candidate = report.get("candidate", {})
    baseline = report.get("baseline", {})
    if candidate or baseline:
        lines.append(
            f"  candidate pass_rate: {candidate.get('pass_rate', 'n/a')} "
            f"({candidate.get('passed', '?')}/{candidate.get('total', '?')})"
        )
        lines.append(
            f"  baseline pass_rate: {baseline.get('pass_rate', 'n/a')} "
            f"({baseline.get('passed', '?')}/{baseline.get('total', '?')})"
        )
        lines.append(f"  mandatory_pass: {report.get('mandatory_pass')}")
        lines.append(f"  no_category_regression: {report.get('no_category_regression')}")
        lines.append(f"  approved: {report.get('approved')}")
    return lines

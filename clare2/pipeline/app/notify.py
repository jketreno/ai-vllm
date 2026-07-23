"""Nightly lifecycle run notification emails, sent via the self-hosted mail relay."""

from __future__ import annotations

import html as html_lib
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

SMTP_HOST = os.environ.get("CLARE2_SMTP_HOST", "localhost")
SMTP_PORT = int(os.environ.get("CLARE2_SMTP_PORT", "25"))
SMTP_TIMEOUT = float(os.environ.get("CLARE2_SMTP_TIMEOUT", "15"))
NOTIFY_FROM = os.environ.get("CLARE2_NOTIFY_FROM", "")
NOTIFY_TO = os.environ.get("CLARE2_NOTIFY_TO", "")

OUTCOME_LABELS = {
    "promoted": "PROMOTED",
    "rejected": "REJECTED",
    "skipped_no_new_content": "SKIPPED (no new content)",
    "postponed": "POSTPONED (inference active)",
    "failed": "FAILED",
}

# Status label -> (background, text) colors for HTML badges/rows.
_STATUS_COLORS = {
    "PROMOTED": ("#e6f4ea", "#1e7b34"),
    "REJECTED": ("#fdecea", "#c5221f"),
    "NOT TRAINED": ("#f1f3f4", "#5f6368"),
    "FAILED": ("#fdecea", "#c5221f"),
    "SKIPPED (no new content)": ("#fff8e1", "#8a6d00"),
    "POSTPONED (inference active)": ("#fff8e1", "#8a6d00"),
}
_DEFAULT_STATUS_COLOR = ("#f1f3f4", "#5f6368")


def send_run_notification(outcome: str, **context: Any) -> None:
    """Best-effort notification email for a nightly lifecycle terminal state.

    Never raises: a failed send must not mask or interrupt the lifecycle
    state machine that just recorded the real outcome.
    """
    if not NOTIFY_TO:
        return
    try:
        subject, body, html_body = _compose(outcome, context)
        _send(subject, body, html_body)
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
        subject, body, html_body = _compose_batch(run_id, results)
        _send(subject, body, html_body)
    except Exception:
        log.exception("Failed to send batch run notification for run_id=%s", run_id)
        metrics.notification_sent.labels(outcome=outcome, status="error").inc()
        return
    metrics.notification_sent.labels(outcome=outcome, status="ok").inc()


def _send(subject: str, body: str, html_body: str | None = None) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = NOTIFY_FROM
    msg["To"] = NOTIFY_TO
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid()
    msg.set_content(body)
    if html_body:
        msg.add_alternative(html_body, subtype="html")
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=SMTP_TIMEOUT) as smtp:
        smtp.ehlo()
        refused = smtp.send_message(msg)
        if refused:
            raise smtplib.SMTPRecipientsRefused(refused)


def _compose(outcome: str, context: dict[str, Any]) -> tuple[str, str, str]:
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
    lines.extend(_outcome_body_lines(outcome, project, label, context))
    body = "\n".join(lines) + "\n"

    html_sections = [
        _html_meta_table(
            [("Outcome", label)]
            + ([("Project", project)] if project else [])
            + [("Run ID", context.get("run_id", "unknown"))]
        ),
        _html_section("Distillation", _html_distillation_table(_project_summaries())),
    ]
    html_sections.extend(_outcome_html_sections(outcome, project, label, context))
    html = _html_wrap(subject, "".join(html_sections))

    return subject, body, html


def _outcome_body_lines(
    outcome: str, project: str | None, label: str, context: dict[str, Any]
) -> list[str]:
    if outcome in ("promoted", "rejected"):
        return ["TRAINING / EVALUATION", *_evaluation_lines(context)]
    if outcome == "skipped_no_new_content":
        summaries = _project_summaries()
        if not summaries:
            return ["TRAINING", "  No projects were discovered; run skipped."]
        lines = ["TRAINING"]
        for skipped_project, summary in summaries.items():
            lines.append(f"  {skipped_project}: {_not_trained_reason(summary)}")
            lines.append("")
        lines.pop()
        return lines
    if outcome == "postponed":
        return [
            "TRAINING",
            f"  Postponed because {context.get('active_sessions', 'unknown')} "
            "inference request(s) are active.",
            "  Active inference will be checked again every 30 seconds.",
        ]
    if outcome == "failed":
        return [
            "FAILURE",
            f"  adapter_id: {context.get('adapter_id') or 'n/a'}",
            f"  error: {context.get('error')}",
        ]
    return []


def _outcome_html_sections(
    outcome: str, project: str | None, label: str, context: dict[str, Any]
) -> list[str]:
    if outcome in ("promoted", "rejected"):
        return [
            _html_section(
                "Training / Evaluation",
                _html_evaluation_table([(project or "unknown", label, context, {})]),
            )
        ]
    if outcome == "skipped_no_new_content":
        return [
            _html_section("Training", _html_skipped_reasons_table(_project_summaries()))
        ]
    if outcome == "postponed":
        return [
            _html_section(
                "Training",
                f"<p>Postponed because "
                f"<strong>{_esc(context.get('active_sessions', 'unknown'))}</strong> "
                "inference request(s) are active.</p>"
                "<p>Active inference will be checked again every 30 seconds.</p>",
            )
        ]
    if outcome == "failed":
        return [
            _html_section(
                "Failure",
                f'<p><span class="badge badge-fail">FAILED</span></p>'
                f"<p><strong>adapter_id:</strong> "
                f"{_esc(context.get('adapter_id') or 'n/a')}</p>"
                f"<p><strong>error:</strong> {_esc(context.get('error'))}</p>",
            )
        ]
    return []


def _html_skipped_reasons_table(summaries: dict[str, dict[str, Any]]) -> str:
    if not summaries:
        return "<p>No projects were discovered; run skipped.</p>"
    rows = "".join(
        f"<tr><td>{_esc(p)}</td><td>{_esc(_not_trained_reason(s))}</td></tr>"
        for p, s in summaries.items()
    )
    return (
        '<table class="data"><thead><tr><th>Project</th><th>Reason</th></tr>'
        f"</thead><tbody>{rows}</tbody></table>"
    )


def _compose_batch(run_id: str, results: list[dict[str, Any]]) -> tuple[str, str, str]:
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

    eval_entries: list[tuple[str, str, dict[str, Any] | None, dict[str, Any]]] = []
    for project, summary in project_summaries.items():
        result = result_by_project.get(project)
        if result:
            label = OUTCOME_LABELS.get(
                result.get("outcome"), str(result.get("outcome")).upper()
            )
            lines.append(f"TRAINING / EVALUATION — {project} ({label})")
            lines.extend(_evaluation_lines(result))
        else:
            label = "NOT TRAINED"
            lines.append(f"TRAINING / EVALUATION — {project} (NOT TRAINED)")
            lines.append(f"  reason: {_not_trained_reason(summary)}")
        eval_entries.append((project, label, result, summary))
        lines.append("")

    body = "\n".join(lines) + "\n"

    html_sections = [
        _html_meta_table(
            [
                ("Run ID", run_id),
                ("Projects trained", ", ".join(projects) if projects else "none"),
            ]
        ),
        _html_section("Distillation", _html_distillation_table(project_summaries)),
        _html_section(
            "Training / Evaluation",
            _html_evaluation_table(
                [
                    (project, label, result, summary)
                    for project, label, result, summary in eval_entries
                ]
            ),
        ),
    ]
    html = _html_wrap(subject, "".join(html_sections))

    return subject, body, html


def _distillation_lines(extra_projects: set[str] | None = None) -> list[str]:
    summaries = _project_summaries(extra_projects)
    if not summaries:
        return ["  no projects discovered"]
    lines = []
    for project, summary in summaries.items():
        lines.append(f"  {project}:")
        lines.append(
            "    sessions: "
            f"{summary['session_count']} captured, "
            f"{summary['processed_count']} processed, "
            f"{summary['pending_count']} pending; latest: {summary['latest_session']}"
        )
        lines.append(f"    last_distillation: {summary['last_distillation']}")
        episodes = summary["episodes"]
        counts = ", ".join(
            f"{category}: {count}" for category, count in episodes.items()
        )
        lines.append(f"    episode patterns on file: {counts or 'none'}")
        lines.append(
            "    current corpus: "
            f"{summary['sft_pairs']} SFT pair(s), ~{summary['tokens']} tokens; "
            f"last updated: {summary['corpus_updated']}"
        )
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _project_summaries(
    extra_projects: set[str] | None = None
) -> dict[str, dict[str, Any]]:
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
        session_files = list(
            (corpus.CORPUS_ROOT / "sessions" / project).glob("**/*.jsonl")
        )
        processed = indexed.get(project, [])
        session_dates = [
            record.get("date") for record in processed if record.get("date")
        ]
        for session_file in session_files:
            relative_parts = session_file.relative_to(
                corpus.CORPUS_ROOT / "sessions" / project
            ).parts
            if len(relative_parts) >= 4:
                session_dates.append("-".join(relative_parts[:3]))
        manifest = _load_json(
            corpus.CORPUS_ROOT / "training" / project / "manifest.json", {}
        )
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
        return (
            f"{summary['pending_count']} captured session(s) remain "
            "pending distillation"
        )
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
        lines.append(
            f"  no_category_regression: {report.get('no_category_regression')}"
        )
        lines.append(f"  approved: {report.get('approved')}")
    return lines


def _esc(value: Any) -> str:
    return html_lib.escape(str(value), quote=True)


def _badge(label: str) -> str:
    background, color = _STATUS_COLORS.get(label, _DEFAULT_STATUS_COLOR)
    return (
        f'<span class="badge" style="background:{background};color:{color};">'
        f"{_esc(label)}</span>"
    )


def _bool_cell(value: Any) -> str:
    if value is True:
        return '<span style="color:#1e7b34;font-weight:600;">&#10003;</span>'
    if value is False:
        return '<span style="color:#c5221f;font-weight:600;">&#10007;</span>'
    return "&mdash;"


def _html_wrap(title: str, sections_html: str) -> str:
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
  body {{
    margin: 0;
    padding: 0;
    background: #f4f5f7;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica,
      Arial, sans-serif;
    color: #202124;
  }}
  .wrapper {{
    max-width: 760px;
    margin: 0 auto;
    padding: 24px 16px;
  }}
  .card {{
    background: #ffffff;
    border-radius: 8px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }}
  .header {{
    background: #202940;
    color: #ffffff;
    border-radius: 8px 8px 0 0;
    padding: 20px 24px;
    margin: -24px -24px 20px -24px;
  }}
  .header h1 {{
    margin: 0;
    font-size: 18px;
    font-weight: 600;
  }}
  h2 {{
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #5f6368;
    border-bottom: 2px solid #e8eaed;
    padding-bottom: 6px;
    margin: 28px 0 12px 0;
  }}
  h2:first-of-type {{ margin-top: 0; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}
  table.meta td {{
    padding: 4px 8px 4px 0;
    vertical-align: top;
  }}
  table.meta td.label {{
    color: #5f6368;
    white-space: nowrap;
    width: 1%;
  }}
  table.data th {{
    text-align: left;
    background: #f1f3f4;
    color: #3c4043;
    padding: 8px 10px;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }}
  table.data td {{
    padding: 8px 10px;
    border-bottom: 1px solid #eceff1;
    vertical-align: top;
  }}
  table.data tr:nth-child(even) td {{
    background: #fafbfc;
  }}
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.02em;
  }}
  .mono {{
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 12px;
    color: #3c4043;
  }}
  .muted {{ color: #80868b; }}
  p {{ font-size: 13px; line-height: 1.5; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #16181d; color: #e8eaed; }}
    .card {{ background: #24262b; box-shadow: none; }}
    h2 {{ color: #9aa0a6; border-bottom-color: #3c4043; }}
    table.data th {{ background: #2f3136; color: #c8ccd0; }}
    table.data td {{ border-bottom-color: #33353a; }}
    table.data tr:nth-child(even) td {{ background: #2a2c31; }}
    table.meta td.label {{ color: #9aa0a6; }}
    .mono {{ color: #c8ccd0; }}
  }}
</style>
</head>
<body>
<div class="wrapper">
  <div class="card">
    <div class="header"><h1>{_esc(title)}</h1></div>
    {sections_html}
  </div>
</div>
</body>
</html>
"""


def _html_section(title: str, inner_html: str) -> str:
    return f"<h2>{_esc(title)}</h2>{inner_html}"


def _html_meta_table(rows: list[tuple[str, Any]]) -> str:
    body_rows = "".join(
        f'<tr><td class="label">{_esc(label)}</td>'
        f'<td class="mono">{_esc(value)}</td></tr>'
        for label, value in rows
    )
    return f'<table class="meta"><tbody>{body_rows}</tbody></table>'


def _html_distillation_table(summaries: dict[str, dict[str, Any]]) -> str:
    if not summaries:
        return '<p class="muted">No projects discovered.</p>'
    rows = []
    for project, summary in summaries.items():
        episodes = summary["episodes"]
        episode_html = (
            ", ".join(
                f"{_esc(category)}: {count}" for category, count in episodes.items()
            )
            or '<span class="muted">none</span>'
        )
        rows.append(
            "<tr>"
            f"<td><strong>{_esc(project)}</strong></td>"
            f"<td>{summary['session_count']} captured<br>"
            f"{summary['processed_count']} processed<br>"
            f"{summary['pending_count']} pending</td>"
            f"<td>{_esc(summary['latest_session'])}</td>"
            f'<td class="mono">{_esc(summary["last_distillation"])}</td>'
            f"<td>{episode_html}</td>"
            f"<td>{summary['sft_pairs']} pair(s)<br>~{summary['tokens']} tokens<br>"
            f'<span class="muted">{_esc(summary["corpus_updated"])}</span></td>'
            "</tr>"
        )
    return (
        '<table class="data"><thead><tr>'
        "<th>Project</th><th>Sessions</th><th>Latest</th>"
        "<th>Last Distillation</th><th>Episode Patterns</th><th>Corpus</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _html_evaluation_table(
    entries: list[tuple[str, str, dict[str, Any] | None, dict[str, Any]]],
) -> str:
    if not entries:
        return '<p class="muted">No training results for this run.</p>'
    rows = []
    for project, label, result, summary in entries:
        badge = _badge(label)
        if result:
            report = result.get("report") or {}
            candidate = report.get("candidate", {})
            baseline = report.get("baseline", {})
            rows.append(
                "<tr>"
                f"<td><strong>{_esc(project)}</strong><br>{badge}</td>"
                f'<td class="mono">{_esc(result.get("adapter_id", "unknown"))}</td>'
                f'<td class="mono">{_esc(result.get("mlflow_run_id") or "n/a")}</td>'
                f"<td>{_esc(candidate.get('pass_rate', 'n/a'))} "
                f"({_esc(candidate.get('passed', '?'))}/"
                f"{_esc(candidate.get('total', '?'))})</td>"
                f"<td>{_esc(baseline.get('pass_rate', 'n/a'))} "
                f"({_esc(baseline.get('passed', '?'))}/"
                f"{_esc(baseline.get('total', '?'))})</td>"
                f"<td>{_bool_cell(report.get('mandatory_pass'))}</td>"
                f"<td>{_bool_cell(report.get('no_category_regression'))}</td>"
                f"<td>{_bool_cell(report.get('approved'))}</td>"
                "</tr>"
            )
        else:
            reason = _not_trained_reason(summary) if summary else "n/a"
            rows.append(
                "<tr>"
                f"<td><strong>{_esc(project)}</strong><br>{badge}</td>"
                f'<td class="muted" colspan="6">{_esc(reason)}</td>'
                "</tr>"
            )
    return (
        '<table class="data"><thead><tr>'
        "<th>Project</th><th>Adapter</th><th>MLflow Run</th>"
        "<th>Candidate</th><th>Baseline</th><th>Mandatory</th>"
        "<th>No Regression</th><th>Approved</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )

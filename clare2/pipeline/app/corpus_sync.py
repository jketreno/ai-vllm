"""Nightly remote corpus sync: pull each developer's sessions/ over SSH before
distillation."""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone

import yaml

from . import metrics

log = logging.getLogger(__name__)

CORPUS_ROOT = pathlib.Path(os.environ.get("CORPUS_ROOT", "/corpus"))
SOURCES_PATH = pathlib.Path(
    os.environ.get("CLARE2_CORPUS_SOURCES_FILE", "/app/config/corpus_sources.yml")
)
SSH_KEY_PATH = pathlib.Path(
    os.environ.get("CLARE2_CORPUS_SYNC_KEY_FILE", "/run/secrets/clare2_corpus_sync_key")
)
RSYNC_TIMEOUT_SECONDS = int(os.environ.get("CLARE2_CORPUS_SYNC_TIMEOUT_SECONDS", "120"))

_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9.-]+$")


class CorpusSourceError(ValueError):
    """A corpus_sources.yml entry is malformed or unsafe."""


@dataclass(frozen=True)
class CorpusSource:
    host: str
    port: int
    user: str
    remote_corpus_root: str
    host_key: str


def _validate_remote_root(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise CorpusSourceError("remote_corpus_root is required")
    if ".." in pathlib.PurePosixPath(raw).parts:
        raise CorpusSourceError(f"remote_corpus_root must not contain '..': {raw}")
    return raw


def _validate_entry(entry: dict) -> CorpusSource:
    host = entry.get("host")
    if not isinstance(host, str) or not _HOSTNAME_RE.fullmatch(host):
        raise CorpusSourceError(f"invalid host: {host!r}")
    user = entry.get("user")
    if not isinstance(user, str) or not user or "/" in user:
        raise CorpusSourceError(f"invalid user: {user!r}")
    host_key = entry.get("host_key")
    if not isinstance(host_key, str) or not host_key.strip():
        raise CorpusSourceError(
            f"host_key is required for {host} — refusing unpinned host-key checking"
        )
    port = entry.get("port", 22)
    if not isinstance(port, int) or not (0 < port < 65536):
        raise CorpusSourceError(f"invalid port for {host}: {port!r}")
    remote_root = _validate_remote_root(entry.get("remote_corpus_root", ""))
    return CorpusSource(
        host=host,
        port=port,
        user=user,
        remote_corpus_root=remote_root,
        host_key=host_key,
    )


def load_sources(path: pathlib.Path | None = None) -> list[CorpusSource]:
    sources_path = path or SOURCES_PATH
    if not sources_path.exists():
        log.info(
            "No corpus_sources.yml found at %s — skipping remote sync", sources_path
        )
        return []
    document = yaml.safe_load(sources_path.read_text(encoding="utf-8")) or {}
    entries = document.get("sources") or []
    if not isinstance(entries, list):
        raise CorpusSourceError("sources must be a list")
    return [_validate_entry(entry) for entry in entries]


def _write_known_hosts(
    sources: list[CorpusSource], directory: pathlib.Path
) -> pathlib.Path:
    known_hosts_path = directory / "known_hosts"
    known_hosts_path.write_text(
        "\n".join(source.host_key for source in sources) + "\n", encoding="utf-8"
    )
    known_hosts_path.chmod(0o600)
    return known_hosts_path


def _sync_source(
    source: CorpusSource, known_hosts_path: pathlib.Path
) -> tuple[str, str]:
    # rrsync's forced authorized_keys command (e.g. `rrsync -ro .../sessions`)
    # fixes the remote root directory itself, so the remote path argument here
    # must be relative ("."), not the absolute remote_corpus_root repeated —
    # rrsync would otherwise join it onto its own fixed root and fail to find it.
    local_sessions = CORPUS_ROOT / "sessions"
    local_sessions.mkdir(parents=True, exist_ok=True)

    ssh_command = (
        f"ssh -i {SSH_KEY_PATH} "
        f"-o UserKnownHostsFile={known_hosts_path} "
        "-o StrictHostKeyChecking=yes "
        "-o BatchMode=yes "
        f"-p {source.port}"
    )
    command = [
        "rsync",
        "-az",
        "--timeout",
        str(RSYNC_TIMEOUT_SECONDS),
        "-e",
        ssh_command,
        f"{source.user}@{source.host}:./",
        f"{local_sessions}/",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=RSYNC_TIMEOUT_SECONDS + 30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        log.error('{"event":"corpus_sync_timeout","host":"%s"}', source.host)
        return "timeout", ""
    if result.returncode != 0:
        log.error(
            '{"event":"corpus_sync_failed","host":"%s","returncode":%d,"stderr":%s}',
            source.host,
            result.returncode,
            json.dumps(result.stderr[-2000:]),
        )
        return "failed", result.stderr
    log.info('{"event":"corpus_sync_succeeded","host":"%s"}', source.host)
    return "succeeded", ""


def sync_all() -> dict:
    """Sync sessions/ from every configured remote host. Never raises — a single
    unreachable host must not block distillation for other projects."""
    try:
        sources = load_sources()
    except CorpusSourceError:
        log.exception("corpus_sources.yml is invalid — skipping remote corpus sync")
        return {"hosts": 0, "succeeded": 0, "failed": 0}

    if not sources:
        return {"hosts": 0, "succeeded": 0, "failed": 0}

    if not SSH_KEY_PATH.exists():
        log.error(
            "Corpus sync key not found at %s — skipping remote corpus sync",
            SSH_KEY_PATH,
        )
        return {"hosts": len(sources), "succeeded": 0, "failed": len(sources)}

    results: dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tmp:
        known_hosts_path = _write_known_hosts(sources, pathlib.Path(tmp))
        for source in sources:
            try:
                outcome, _ = _sync_source(source, known_hosts_path)
            except Exception:
                log.exception("Unexpected error syncing corpus from %s", source.host)
                outcome = "failed"
            results[source.host] = outcome
            metrics.corpus_sync_hosts.labels(outcome=outcome).inc()

    succeeded = sum(1 for outcome in results.values() if outcome == "succeeded")
    failed = len(results) - succeeded
    summary = {
        "hosts": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "results": results,
        "completed_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    _write_status(summary)
    metrics.corpus_sync_last_run_timestamp.set(
        datetime.now(tz=timezone.utc).timestamp()
    )
    return summary


def _write_status(summary: dict) -> None:
    status_path = CORPUS_ROOT / "meta" / "corpus_sync_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

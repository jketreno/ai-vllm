"""Persisted nightly adapter lifecycle with failure recovery."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import pathlib
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

import httpx

from . import corpus, evaluator, metrics, notify
from .runtime import BASE_MODEL_ID, VLLM_URL, controller, maintenance, registry

log = logging.getLogger(__name__)
STATE_ROOT = pathlib.Path(os.environ.get("CLARE2_STATE_ROOT", "/corpus/meta"))
STATE_PATH = STATE_ROOT / "lifecycle.json"
LOCK_PATH = STATE_ROOT / "lifecycle.lock"
DOCKER_PROXY_URL = os.environ.get("CLARE2_DOCKER_PROXY_URL", "http://docker-socket-proxy:2375")
VLLM_CONTAINER = os.environ.get("CLARE2_VLLM_CONTAINER", "vllm-engine")
TRAIN_CONTAINER = os.environ.get("CLARE2_TRAIN_CONTAINER", "clare2-train")
DRAIN_TIMEOUT = float(os.environ.get("CLARE2_DRAIN_TIMEOUT", "900"))
START_TIMEOUT = float(os.environ.get("CLARE2_START_TIMEOUT", "900"))
PROMETHEUS_URL = os.environ.get("CLARE2_PROMETHEUS_URL", "http://prometheus:9090")
ACTIVE_INFERENCE_QUERY = os.environ.get(
    "CLARE2_ACTIVE_INFERENCE_QUERY", "sum(vllm:num_requests_running)"
)
TRAINING_RETRY_INTERVAL = float(os.environ.get("CLARE2_TRAINING_RETRY_INTERVAL", "30"))
IMAGE_LEASE_TTL = float(os.environ.get("CLARE2_IMAGE_LEASE_TTL", "3600"))
IMAGE_LEASE_MIN_AVAILABLE_GIB = float(
    os.environ.get("CLARE2_IMAGE_LEASE_MIN_AVAILABLE_GIB", "16")
)
IMAGE_LEASE_MEMORY_TIMEOUT = float(
    os.environ.get("CLARE2_IMAGE_LEASE_MEMORY_TIMEOUT", "120")
)

PHASES = {
    "idle",
    "postponed",
    "preparing_training",
    "waiting_for_training_container",
    "draining",
    "starting_training",
    "training",
    "restarting",
    "loading",
    "evaluating",
    "promoting",
    "recovering",
    "failed",
    "image_edit",
}

TERMINAL_OUTCOMES = {"promoted", "rejected", "skipped_no_new_content", "batch_complete"}


def status() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"phase": "idle", "run_id": None}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def _mem_available_gib() -> float:
    with open("/proc/meminfo", encoding="utf-8") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    raise RuntimeError("MemAvailable not found")


def _wait_for_image_memory() -> float:
    deadline = time.monotonic() + IMAGE_LEASE_MEMORY_TIMEOUT
    while time.monotonic() < deadline:
        available = _mem_available_gib()
        if available >= IMAGE_LEASE_MIN_AVAILABLE_GIB:
            return available
        time.sleep(2)
    raise TimeoutError(
        f"image inference needs {IMAGE_LEASE_MIN_AVAILABLE_GIB:.1f} GiB MemAvailable"
    )


def _acquire_image_edit_lease_once(request_id: str) -> dict[str, Any]:
    """Drain and stop vLLM, then grant exclusive access to unified GPU memory."""
    with single_run():
        state = reconcile_terminal_state()
        if state.get("phase") not in {"idle", "failed"}:
            phase = state.get("phase")
            raise RuntimeError(f"resources busy in lifecycle phase {phase}")
        lease_id = uuid.uuid4().hex
        maintenance.enter()
        stopped = False
        try:
            if not maintenance.wait_for_drain(DRAIN_TIMEOUT):
                raise TimeoutError("timed out waiting for in-flight inference")
            _container("stop", VLLM_CONTAINER)
            stopped = True
            available = _wait_for_image_memory()
            expires_at = datetime.now(tz=timezone.utc) + timedelta(
                seconds=IMAGE_LEASE_TTL
            )
            _set_state(
                "image_edit", reset=True, lease_id=lease_id,
                request_id=request_id, expires_at=expires_at.isoformat(),
                mem_available_gib=round(available, 2),
                acquired_monotonic=time.monotonic(),
            )
            metrics.image_lease_active.set(1)
            metrics.image_lease_outcomes.labels(outcome="acquired").inc()
            return status()
        except Exception as acquire_error:
            if stopped:
                try:
                    _container("start", VLLM_CONTAINER)
                    _wait_for_vllm()
                except Exception as restore_error:
                    _set_state(
                        "image_edit", reset=True, lease_id=lease_id,
                        request_id=request_id,
                        expires_at=datetime.now(tz=timezone.utc).isoformat(),
                        error=f"vLLM restoration failed: {restore_error}",
                    )
                    metrics.image_lease_active.set(1)
                    metrics.image_lease_outcomes.labels(
                        outcome="restore_failure"
                    ).inc()
                    raise acquire_error from restore_error
            maintenance.exit()
            _set_state("failed", reset=True, error="image lease acquisition failed")
            raise


def acquire_image_edit_lease(request_id: str) -> dict[str, Any]:
    """Queue behind another image lease while rejecting unrelated lifecycle work."""
    deadline = time.monotonic() + DRAIN_TIMEOUT
    while True:
        try:
            return _acquire_image_edit_lease_once(request_id)
        except RuntimeError as error:
            phase = status().get("phase")
            lock_race = "another lifecycle operation" in str(error)
            if phase != "image_edit" and not lock_race:
                raise
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "timed out waiting for the active image lease"
                ) from error
            time.sleep(2)


def release_image_edit_lease(lease_id: str) -> dict[str, Any]:
    """Idempotently restore vLLM and leave maintenance mode."""
    with single_run():
        state = status()
        if state.get("phase") != "image_edit":
            return {"status": "already_released"}
        if state.get("lease_id") != lease_id:
            raise RuntimeError("image lease id does not match the active lease")
        acquired = float(state.get("acquired_monotonic", time.monotonic()))
        _set_state("restarting", lease_id=lease_id)
        try:
            _container("start", VLLM_CONTAINER)
            _wait_for_vllm()
        except Exception as error:
            _set_state(
                "image_edit", reset=True, lease_id=lease_id,
                request_id=state.get("request_id"),
                expires_at=datetime.now(tz=timezone.utc).isoformat(),
                acquired_monotonic=acquired,
                error=f"vLLM restoration failed: {error}",
            )
            metrics.image_lease_outcomes.labels(outcome="restore_failure").inc()
            raise
        maintenance.exit()
        try:
            metrics.image_lease_active.set(0)
            metrics.image_lease_duration.observe(max(0, time.monotonic() - acquired))
            metrics.image_lease_outcomes.labels(outcome="released").inc()
        finally:
            _set_state("idle", reset=True)
        return {"status": "released", "lease_id": lease_id}


def reconcile_image_edit_lease() -> dict[str, Any]:
    state = status()
    if state.get("phase") != "image_edit":
        return state
    maintenance.enter()
    try:
        expires_at = datetime.fromisoformat(state["expires_at"])
    except (KeyError, ValueError):
        expires_at = datetime.now(tz=timezone.utc)
    if expires_at <= datetime.now(tz=timezone.utc):
        return release_image_edit_lease(str(state.get("lease_id", "")))
    return state


def reconcile_terminal_state() -> dict[str, Any]:
    """Recover persisted lifecycle records that reached an outcome but not idle."""
    state = status()
    if state.get("outcome") == "rejected":
        adapter_id = state.get("candidate_id")
        if adapter_id:
            _reconcile_rejected_candidate(adapter_id)
    elif state.get("outcome") == "batch_complete":
        for result in state.get("batch_results", []):
            if result.get("outcome") == "rejected":
                _reconcile_rejected_candidate(result["adapter_id"])
    if state.get("outcome") in TERMINAL_OUTCOMES and state.get("phase") not in {"idle", "failed"}:
        _set_state("idle")
        return status()
    return state


def _reconcile_rejected_candidate(adapter_id: str) -> None:
    """Align registry state after a crash between rejection state and registry transition."""
    try:
        adapter = registry.read().get("adapters", {}).get(adapter_id)
        if adapter and adapter.get("status") in {"training", "candidate", "loaded"}:
            registry.transition(adapter_id, "rejected")
    except Exception:
        log.exception("Failed to reconcile rejected adapter %s", adapter_id)


def drain_and_stop_infer() -> None:
    with single_run():
        started = time.monotonic()
        run_id = os.environ.get("CLARE2_RUN_ID") or _new_run_id()
        _set_state("draining", run_id=run_id)
        maintenance.enter()
        try:
            if not maintenance.wait_for_drain(DRAIN_TIMEOUT):
                raise TimeoutError("timed out waiting for in-flight inference")
            _container("stop", VLLM_CONTAINER)
            _set_state("training", run_id=run_id)
        except Exception as exc:
            _recover(run_id, exc)
            raise
        finally:
            metrics.maintenance_duration.observe(time.monotonic() - started)


def _active_inference_sessions() -> int:
    """Return the active inference count reported by Prometheus."""
    response = httpx.get(
        f"{PROMETHEUS_URL}/api/v1/query",
        params={"query": ACTIVE_INFERENCE_QUERY},
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "success":
        raise RuntimeError(f"Prometheus query failed: {payload.get('error', 'unknown error')}")
    result = payload.get("data", {}).get("result", [])
    if not result:
        raise RuntimeError("Prometheus returned no active-inference metric")
    return sum(max(0, int(float(series["value"][1]))) for series in result)


def _wait_for_inference_idle(run_id: str, postponement_notified: bool) -> bool:
    while True:
        try:
            active_sessions = _active_inference_sessions()
        except (httpx.HTTPError, KeyError, TypeError, ValueError, RuntimeError):
            log.exception("Unable to determine active inference sessions; postponing training")
            time.sleep(TRAINING_RETRY_INTERVAL)
            continue
        if active_sessions == 0:
            return postponement_notified
        if not postponement_notified:
            notify.send_run_notification(
                "postponed", run_id=run_id, active_sessions=active_sessions
            )
            postponement_notified = True
        _set_state(
            "postponed",
            run_id=run_id,
            postponement_notified=postponement_notified,
            active_sessions=active_sessions,
        )
        time.sleep(TRAINING_RETRY_INTERVAL)


def run_nightly_training() -> None:
    """Wait for inference to become idle, refresh SFT data, then train."""
    with single_run():
        state = reconcile_terminal_state()
        if state.get("phase") not in {"postponed", "idle", "failed"}:
            raise RuntimeError(f"cannot start nightly training from {state.get('phase')}")
        resuming = state.get("phase") == "postponed"
        run_id = state.get("run_id") if resuming else _new_run_id()
        postponement_notified = resuming and bool(state.get("postponement_notified"))
        if not resuming:
            _set_state(
                "idle",
                reset=True,
                run_id=run_id,
                postponement_notified=False,
            )
        postponement_notified = _wait_for_inference_idle(run_id, postponement_notified)

        # Rebuild at admission time so a delayed run never trains an older SFT snapshot.
        _set_state(
            "preparing_training",
            run_id=run_id,
            postponement_notified=postponement_notified,
            active_sessions=0,
        )
        corpus.assemble()
        _wait_for_training_container(run_id, postponement_notified=postponement_notified)
        started = time.monotonic()
        _set_state(
            "draining",
            run_id=run_id,
            postponement_notified=postponement_notified,
            active_sessions=0,
        )
        maintenance.enter()
        try:
            if not maintenance.wait_for_drain(DRAIN_TIMEOUT):
                raise TimeoutError("timed out waiting for in-flight inference")
            _container("stop", VLLM_CONTAINER)
            _start_training_container(run_id, postponement_notified=postponement_notified)
        except Exception as exc:
            _recover(run_id, exc)
            raise
        finally:
            metrics.maintenance_duration.observe(time.monotonic() - started)


def start_training() -> None:
    with single_run():
        state = reconcile_terminal_state()
        run_id = state.get("run_id") or _new_run_id()
        if state.get("phase") not in {"training", "idle"}:
            raise RuntimeError(f"cannot start training from {state.get('phase')}")
        try:
            _start_training_container(run_id)
        except Exception as exc:
            _recover(run_id, exc)
            raise


def start_dream_training(run_id: str) -> dict[str, Any]:
    with single_run():
        state = status()
        if state.get("phase") not in {"idle", "failed"}:
            raise RuntimeError(f"cannot start dream training from {state.get('phase')}")
        _set_state("training", run_id=run_id, dream_mode=True)
        maintenance.enter()
        return status()


def _apply_evaluation(
    adapter_id: str,
    run_id: str,
    mlflow_run_id: str | None,
    report: dict[str, Any],
    project: str = "unknown",
) -> str:
    """Promote or reject a candidate in the registry. Returns the outcome."""
    if not report["approved"]:
        registry.transition(adapter_id, "rejected")
        metrics.lifecycle_outcomes.labels(outcome="rejected").inc()
        return "rejected"

    registry.promote(adapter_id, report)
    controller.reconcile()
    metrics.lifecycle_outcomes.labels(outcome="promoted").inc()
    return "promoted"


def _record_training_metrics(
    adapter_id: str,
    project: str,
    loss: float | None,
    epoch_losses: list[float],
) -> None:
    for epoch, epoch_loss in enumerate(epoch_losses, 1):
        metrics.training_loss_by_epoch.labels(project=project, epoch=str(epoch)).set(epoch_loss)
    if loss is not None:
        metrics.training_loss_final.labels(project=project).set(loss)

    meta_path = registry.adapters_root / adapter_id / "training_meta.json"
    try:
        training_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    duration = training_meta.get("duration_seconds")
    if duration is not None:
        metrics.training_duration_seconds.labels(project=project).set(duration)

    adapter_dir = registry.adapters_root / adapter_id
    size = sum(f.stat().st_size for f in adapter_dir.rglob("*") if f.is_file())
    metrics.adapter_size_bytes.labels(project=project).set(size)


def complete_training(
    adapter_id: str,
    run_id: str,
    mlflow_run_id: str | None = None,
    loss: float | None = None,
    epoch_losses: list[float] | None = None,
) -> dict[str, Any]:
    with single_run():
        state = status()
        if state.get("completed_adapter_id") == adapter_id:
            return state
        if state.get("run_id") != run_id or state.get("phase") not in {"training", "idle"}:
            raise RuntimeError("callback does not match the active training run")
        try:
            _register_candidate(adapter_id, run_id, mlflow_run_id, loss, epoch_losses or [])
            _set_state(
                "restarting",
                run_id=run_id,
                candidate_id=adapter_id,
                mlflow_run_id=mlflow_run_id,
                trainer_start_requested=False,
            )
            _container("start", VLLM_CONTAINER)
            _wait_for_vllm()
            controller.reconcile()

            result = _evaluate_and_apply(run_id, adapter_id, mlflow_run_id)
            _set_state(
                "idle",
                run_id=run_id,
                candidate_id=adapter_id,
                completed_adapter_id=adapter_id,
                mlflow_run_id=mlflow_run_id,
                outcome=result["outcome"],
                evaluation=result["report"],
            )
            maintenance.exit()
            notify.send_run_notification(
                result["outcome"],
                adapter_id=adapter_id,
                run_id=run_id,
                mlflow_run_id=mlflow_run_id,
                report=result["report"],
                project=result["project"],
            )
            return status()
        except Exception as exc:
            _recover(run_id, exc, adapter_id=adapter_id)
            raise


def _register_candidate(
    adapter_id: str,
    run_id: str,
    mlflow_run_id: str | None,
    loss: float | None,
    epoch_losses: list[float],
) -> str:
    """Add the trained adapter to the registry if needed and record its
    training metrics. Returns the adapter's project scope."""
    candidate_path = registry.adapters_root / adapter_id / "candidate_manifest.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    project = candidate.get("project_scope", "unknown")
    if adapter_id not in registry.read()["adapters"]:
        registry.add_adapter(candidate)
    _record_training_metrics(adapter_id, project, loss, epoch_losses)
    return project


def complete_training_batch(run_id: str, adapters: list[dict[str, Any]]) -> dict[str, Any]:
    """Handle the nightly run's single batch callback covering every project
    clare2-train trained in this invocation. vLLM is restarted once — not per
    adapter — since clare2-train trains all projects sequentially in one GPU
    reservation and only signals completion after the whole batch finishes.
    Each candidate is still evaluated and promoted/rejected independently so
    one project's outcome never blocks another's."""
    with single_run():
        state = status()
        if state.get("completed_run_id") == run_id:
            return state
        if state.get("run_id") != run_id or state.get("phase") not in {"training", "idle"}:
            raise RuntimeError("callback does not match the active training run")
        in_flight_adapter_id: str | None = None
        try:
            _set_state(
                "restarting",
                run_id=run_id,
                pending_adapter_ids=[a["adapter_id"] for a in adapters],
                trainer_start_requested=False,
            )
            _container("start", VLLM_CONTAINER)
            _wait_for_vllm()
            controller.reconcile()

            results = []
            for adapter in adapters:
                in_flight_adapter_id = adapter["adapter_id"]
                results.append(_complete_one_candidate(run_id, adapter))
            in_flight_adapter_id = None

            maintenance.exit()
            _set_state(
                "idle",
                run_id=run_id,
                completed_run_id=run_id,
                outcome="batch_complete",
                batch_results=results,
            )
            notify.send_batch_run_notification(run_id=run_id, results=results)
            return status()
        except Exception as exc:
            _recover(run_id, exc, adapter_id=in_flight_adapter_id)
            raise


def _complete_one_candidate(run_id: str, adapter: dict[str, Any]) -> dict[str, Any]:
    adapter_id = adapter["adapter_id"]
    mlflow_run_id = adapter.get("mlflow_run_id")
    project = _register_candidate(
        adapter_id, run_id, mlflow_run_id, adapter.get("loss"), adapter.get("epoch_losses") or []
    )
    result = _evaluate_and_apply(run_id, adapter_id, mlflow_run_id, project=project)
    return {"adapter_id": adapter_id, "mlflow_run_id": mlflow_run_id, **result}


def _evaluate_and_apply(
    run_id: str,
    adapter_id: str,
    mlflow_run_id: str | None,
    project: str | None = None,
) -> dict[str, Any]:
    """Load the candidate against the current baseline, evaluate it, and
    promote or reject it in the registry. Returns project/outcome/report."""
    if project is None:
        candidate_path = registry.adapters_root / adapter_id / "candidate_manifest.json"
        project = json.loads(candidate_path.read_text(encoding="utf-8")).get("project_scope", "unknown")

    document = registry.read()
    current_id = document["aliases"]["current"]
    baseline_id = current_id or BASE_MODEL_ID
    _set_state("loading", run_id=run_id, candidate_id=adapter_id)
    if current_id:
        controller.ensure_loaded(current_id)
    controller.ensure_loaded(adapter_id)

    _set_state("evaluating", run_id=run_id, candidate_id=adapter_id)
    report = evaluator.compare(adapter_id, baseline_id, _invoke_probe, project=project)
    outcome = _apply_evaluation(adapter_id, run_id, mlflow_run_id, report, project=project)
    return {"project": project, "outcome": outcome, "report": report}


def complete_training_skipped(run_id: str) -> dict[str, Any]:
    """Handle a training run where every project's corpus was unchanged, so
    clare2-train trained nothing and there is no candidate to evaluate.
    Restarts vLLM and returns to idle — mirrors complete_training() minus
    the candidate load/evaluate/promote steps, which don't apply here."""
    with single_run():
        state = status()
        if state.get("completed_adapter_id") == f"skipped:{run_id}":
            return state
        if state.get("run_id") != run_id or state.get("phase") not in {"training", "idle"}:
            raise RuntimeError("callback does not match the active training run")
        try:
            _set_state("restarting", run_id=run_id, trainer_start_requested=False)
            _container("start", VLLM_CONTAINER)
            _wait_for_vllm()
            controller.reconcile()
            metrics.lifecycle_outcomes.labels(outcome="skipped_no_new_content").inc()
            maintenance.exit()
            _set_state(
                "idle",
                run_id=run_id,
                completed_adapter_id=f"skipped:{run_id}",
                outcome="skipped_no_new_content",
            )
            notify.send_run_notification("skipped_no_new_content", run_id=run_id)
            return status()
        except Exception as exc:
            _recover(run_id, exc)
            raise


def rollback() -> dict[str, Any]:
    with single_run():
        maintenance.enter()
        if not maintenance.wait_for_drain(DRAIN_TIMEOUT):
            maintenance.exit()
            raise TimeoutError("timed out draining before rollback")
        prior = registry.read()["aliases"].copy()
        document, adapter_id = registry.rollback()
        try:
            controller.ensure_loaded(adapter_id)
            completion = _invoke_probe(
                adapter_id,
                {"prompt": "Reply with exactly: CLARE_ROLLBACK_OK", "expected_keyword": "CLARE_ROLLBACK_OK"},
            )
            if "CLARE_ROLLBACK_OK" not in completion:
                raise RuntimeError("rollback smoke request failed")
            metrics.lifecycle_outcomes.labels(outcome="rollback").inc()
            return {"current": document["aliases"]["current"], "rollback": document["aliases"]["rollback"]}
        except Exception:
            registry.update(lambda data: data["aliases"].update(prior))
            if prior["current"]:
                controller.ensure_loaded(prior["current"])
            raise
        finally:
            maintenance.exit()


def _recover(run_id: str, error: Exception, adapter_id: str | None = None) -> None:
    log.exception("Lifecycle failure; recovering prior approved adapter")
    _set_state(
        "recovering",
        run_id=run_id,
        candidate_id=adapter_id,
        error=str(error),
        trainer_start_requested=False,
    )
    try:
        if adapter_id and adapter_id in registry.read()["adapters"]:
            registry.transition(adapter_id, "failed")
        _container("start", VLLM_CONTAINER)
        _wait_for_vllm()
        controller.reconcile()
        current = registry.read()["aliases"]["current"]
        if current:
            controller.ensure_loaded(current)
        metrics.lifecycle_outcomes.labels(outcome="recovered").inc()
    finally:
        maintenance.exit()
        _set_state("failed", run_id=run_id, candidate_id=adapter_id, error=str(error))
        notify.send_run_notification(
            "failed", run_id=run_id, adapter_id=adapter_id, error=str(error)
        )


def _invoke_probe(model: str, probe: dict[str, Any]) -> str:
    response = httpx.post(
        f"{VLLM_URL}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": probe["prompt"]}],
            "temperature": 0,
            "seed": 42,
            "max_tokens": 256,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"].get("content") or ""


def _container(action: str, name: str) -> None:
    if action not in {"start", "stop"}:
        raise ValueError("unsupported container action")
    response = httpx.post(f"{DOCKER_PROXY_URL}/containers/{name}/{action}", timeout=30)
    if response.status_code not in {204, 304}:
        response.raise_for_status()


def _container_exists(name: str) -> bool:
    response = httpx.get(f"{DOCKER_PROXY_URL}/containers/{name}/json", timeout=30)
    if response.status_code == 404:
        return False
    response.raise_for_status()
    return True


def _wait_for_training_container(run_id: str, **fields: Any) -> None:
    """Keep inference available while Compose finishes creating the trainer."""
    while not _container_exists(TRAIN_CONTAINER):
        _set_state(
            "waiting_for_training_container",
            run_id=run_id,
            waiting_for=TRAIN_CONTAINER,
            **fields,
        )
        log.info("Training container %s is not available yet; waiting", TRAIN_CONTAINER)
        time.sleep(TRAINING_RETRY_INTERVAL)


def _start_training_container(run_id: str, **fields: Any) -> None:
    """Start the trainer, treating a transient Docker 404 as a wait state."""
    while True:
        _set_state(
            "starting_training",
            run_id=run_id,
            trainer_start_requested=True,
            **fields,
        )
        try:
            _container("start", TRAIN_CONTAINER)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            _set_state(
                "waiting_for_training_container",
                run_id=run_id,
                waiting_for=TRAIN_CONTAINER,
                **fields,
            )
            log.info("Training container %s disappeared before start; waiting", TRAIN_CONTAINER)
            time.sleep(TRAINING_RETRY_INTERVAL)
            continue
        _set_state(
            "training",
            run_id=run_id,
            trainer_start_requested=True,
            **fields,
        )
        return


def _wait_for_vllm() -> None:
    deadline = time.monotonic() + START_TIMEOUT
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{VLLM_URL}/health", timeout=5)
            if response.is_success:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise TimeoutError("vLLM did not become healthy")


def _set_state(phase: str, reset: bool = False, **fields: Any) -> None:
    if phase not in PHASES:
        raise ValueError(f"unknown lifecycle phase: {phase}")
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    data = {} if reset else status()
    if "error" not in fields and phase not in {"recovering", "failed"}:
        data.pop("error", None)
    data.update(fields)
    data["phase"] = phase
    data["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
    _atomic_json(STATE_PATH, data)
    for label in PHASES:
        metrics.lifecycle_phase.labels(phase=label).set(1 if label == phase else 0)


def _atomic_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _new_run_id() -> str:
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{stamp}-{uuid.uuid4().hex[:12]}"


@contextmanager
def single_run() -> Iterator[None]:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another lifecycle operation is active") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)

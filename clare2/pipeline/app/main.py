"""CLARE₂ control plane and authenticated policy proxy."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from . import corpus, corpus_sync, distiller, lifecycle, metrics, summarizer
from .proxy import router_api as proxy_router
from .registry import RegistryError
from .routing import RouteError
from .runtime import controller, initialize_registry, maintenance, registry, router
from .security import bearer_dependency, secret_value, verify_callback

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","component":"%(name)s","event":"%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger(__name__)
app = FastAPI(title="CLARE₂ Policy Proxy", version="1.0.0")
SCHEDULER_TIMEZONE = os.environ.get("CLARE2_SCHEDULER_TIMEZONE", "America/Los_Angeles")
scheduler = BackgroundScheduler(timezone=SCHEDULER_TIMEZONE)
operator_auth = bearer_dependency(secret_value("CLARE2_OPERATOR_TOKEN"))
internal_route_auth = bearer_dependency(secret_value("CLARE2_CALLBACK_SECRET"))


class TrainingDonePayload(BaseModel):
    adapter_id: str
    run_id: str
    mlflow_run_id: str | None = None
    loss: float | None = None
    epoch_losses: list[float] = Field(default_factory=list)


class TrainedAdapterPayload(BaseModel):
    adapter_id: str
    mlflow_run_id: str | None = None
    loss: float | None = None
    epoch_losses: list[float] = Field(default_factory=list)


class TrainingBatchDonePayload(BaseModel):
    run_id: str
    adapters: list[TrainedAdapterPayload]


class TrainingSkippedPayload(BaseModel):
    run_id: str


class SummarizePayload(BaseModel):
    reference_at: datetime | None = None


class RouteCreatePayload(BaseModel):
    project: str
    task_kind: str
    capabilities: list[str] = Field(default_factory=list)


class DreamTrainingStartPayload(BaseModel):
    run_id: str


class ImageLeasePayload(BaseModel):
    request_id: str = Field(min_length=1, max_length=128)


def sync_distill_and_assemble() -> dict:
    sync_result = corpus_sync.sync_all()
    distill_result = distiller.run_daily()
    assemble_result = corpus.assemble()
    return {
        "sync": sync_result,
        "distill": distill_result,
        "assemble": assemble_result,
    }


@app.on_event("startup")
def startup() -> None:
    initialize_registry()
    metrics.start_metrics_server()
    scheduler.add_job(
        corpus_sync.sync_all, "cron", hour=21, minute=30, id="corpus_sync"
    )
    scheduler.add_job(
        sync_distill_and_assemble, "cron", hour=22, minute=0, id="distill_daily"
    )
    scheduler.add_job(
        summarizer.run_scheduled, "cron", hour=22, minute=30, id="summarize"
    )
    scheduler.add_job(corpus.assemble, "cron", hour=23, minute=30, id="corpus_assemble")
    scheduler.add_job(
        lifecycle.run_nightly_training, "cron", hour=0, minute=0, id="train"
    )
    scheduler.add_job(
        lifecycle.reconcile_image_edit_lease,
        "interval",
        minutes=1,
        id="image_lease_reconcile",
    )
    scheduler.start()
    try:
        lifecycle.reconcile_terminal_state()
        lifecycle.reconcile_image_edit_lease()
        controller.reconcile()
    except Exception:
        log.exception("Initial vLLM reconciliation failed")


@app.on_event("shutdown")
def shutdown() -> None:
    scheduler.shutdown(wait=False)


@app.get("/health")
def health() -> dict:
    return {
        "status": "maintenance" if maintenance.enabled else "ok",
        "active_requests": maintenance.active,
        "ts": datetime.now(tz=timezone.utc).isoformat(),
    }


@app.post("/training/done")
async def training_done(
    payload: TrainingDonePayload,
    request: Request,
    background_tasks: BackgroundTasks,
    x_clare_timestamp: str | None = Header(default=None),
    x_clare_signature: str | None = Header(default=None),
) -> dict:
    body = await request.body()
    verify_callback(
        secret_value("CLARE2_CALLBACK_SECRET"),
        body,
        x_clare_timestamp,
        x_clare_signature,
    )
    state = lifecycle.status()
    if state.get("completed_adapter_id") == payload.adapter_id:
        return {"status": "already_completed"}
    background_tasks.add_task(
        lifecycle.complete_training,
        payload.adapter_id,
        payload.run_id,
        payload.mlflow_run_id,
        payload.loss,
        payload.epoch_losses,
    )
    return {"status": "accepted"}


@app.post("/training/batch-done")
async def training_batch_done(
    payload: TrainingBatchDonePayload,
    request: Request,
    background_tasks: BackgroundTasks,
    x_clare_timestamp: str | None = Header(default=None),
    x_clare_signature: str | None = Header(default=None),
) -> dict:
    body = await request.body()
    verify_callback(
        secret_value("CLARE2_CALLBACK_SECRET"),
        body,
        x_clare_timestamp,
        x_clare_signature,
    )
    state = lifecycle.status()
    if state.get("completed_run_id") == payload.run_id:
        return {"status": "already_completed"}
    background_tasks.add_task(
        lifecycle.complete_training_batch,
        payload.run_id,
        [adapter.model_dump() for adapter in payload.adapters],
    )
    return {"status": "accepted"}


@app.post("/training/skipped")
async def training_skipped(
    payload: TrainingSkippedPayload,
    request: Request,
    background_tasks: BackgroundTasks,
    x_clare_timestamp: str | None = Header(default=None),
    x_clare_signature: str | None = Header(default=None),
) -> dict:
    body = await request.body()
    verify_callback(
        secret_value("CLARE2_CALLBACK_SECRET"),
        body,
        x_clare_timestamp,
        x_clare_signature,
    )
    state = lifecycle.status()
    if state.get("completed_adapter_id") == f"skipped:{payload.run_id}":
        return {"status": "already_completed"}
    background_tasks.add_task(lifecycle.complete_training_skipped, payload.run_id)
    return {"status": "accepted"}


@app.get("/operator/adapters", dependencies=[Depends(operator_auth)])
def adapter_inventory() -> dict:
    document = registry.read()
    return {
        "base": document["base"],
        "aliases": document["aliases"],
        "adapters": list(document["adapters"].values()),
    }


@app.get("/operator/status", dependencies=[Depends(operator_auth)])
def operator_status() -> dict:
    document = registry.read()
    return {
        "aliases": document["aliases"],
        "maintenance": maintenance.enabled,
        "active_requests": maintenance.active,
        "lifecycle": lifecycle.status(),
    }


@app.post("/operator/promote/{adapter_id}", dependencies=[Depends(operator_auth)])
def promote(adapter_id: str) -> dict:
    adapter = registry.read()["adapters"].get(adapter_id)
    if not adapter or not adapter.get("evaluation", {}).get("approved"):
        raise HTTPException(
            status_code=409, detail="adapter has no approved evaluation"
        )
    document = registry.promote(adapter_id, adapter["evaluation"])
    return document["aliases"]


@app.post("/operator/rollback", dependencies=[Depends(operator_auth)])
def rollback() -> dict:
    try:
        return lifecycle.rollback()
    except (RegistryError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/operator/maintenance/{action}", dependencies=[Depends(operator_auth)])
def set_maintenance(action: str) -> dict:
    if action == "enter":
        maintenance.enter()
    elif action == "exit":
        maintenance.exit()
    else:
        raise HTTPException(status_code=400, detail="action must be enter or exit")
    return {"maintenance": maintenance.enabled}


@app.post("/operator/resource-leases/image-edit", dependencies=[Depends(operator_auth)])
def acquire_image_edit_lease(payload: ImageLeasePayload) -> dict:
    try:
        return lifecycle.acquire_image_edit_lease(payload.request_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.delete(
    "/operator/resource-leases/image-edit/{lease_id}",
    dependencies=[Depends(operator_auth)],
)
def release_image_edit_lease(lease_id: str) -> dict:
    try:
        return lifecycle.release_image_edit_lease(lease_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/operator/training/dream/start", dependencies=[Depends(operator_auth)])
def start_dream_training(payload: DreamTrainingStartPayload) -> dict:
    try:
        return lifecycle.start_dream_training(payload.run_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/corpus/sync", dependencies=[Depends(operator_auth)])
def trigger_corpus_sync(background_tasks: BackgroundTasks) -> dict:
    background_tasks.add_task(corpus_sync.sync_all)
    return {"status": "started"}


@app.post("/distill/trigger", dependencies=[Depends(operator_auth)])
def trigger_distill(background_tasks: BackgroundTasks) -> dict:
    background_tasks.add_task(sync_distill_and_assemble)
    return {"status": "started"}


@app.get("/distill/status", dependencies=[Depends(operator_auth)])
def distill_status() -> dict:
    stats_path = corpus.CORPUS_ROOT / "meta" / "corpus_stats.json"
    if not stats_path.exists():
        return {"error": "corpus_stats.json not found"}
    return json.loads(stats_path.read_text(encoding="utf-8"))


@app.post("/corpus/assemble", dependencies=[Depends(operator_auth)])
def trigger_corpus(background_tasks: BackgroundTasks) -> dict:
    background_tasks.add_task(corpus.assemble)
    return {"status": "started"}


@app.post("/summarize/{level}", dependencies=[Depends(operator_auth)])
def trigger_summarize(
    level: str,
    payload: SummarizePayload,
    background_tasks: BackgroundTasks,
) -> dict:
    operations = {
        "weekly": summarizer.run_weekly,
        "monthly": summarizer.run_monthly,
        "quarterly": summarizer.run_quarterly,
        "scheduled": summarizer.run_scheduled,
    }
    operation = operations.get(level)
    if operation is None:
        raise HTTPException(status_code=400, detail="unsupported summary level")
    background_tasks.add_task(operation, payload.reference_at)
    return {"status": "started", "level": level}


@app.post("/internal/routes", dependencies=[Depends(internal_route_auth)])
def create_internal_route(payload: RouteCreatePayload) -> dict:
    try:
        route = router.create_route(
            payload.project, payload.task_kind, payload.capabilities
        )
    except RouteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "route_id": route.route_id,
        "project_id": route.project_id,
        "adapter_id": route.adapter_id,
        "policy_rule": route.policy_rule,
        "expires_at": route.expires_at.isoformat(),
    }


@app.get("/internal/routes/{route_id}", dependencies=[Depends(internal_route_auth)])
def internal_route_status(route_id: str) -> dict:
    try:
        route = router.get(route_id)
    except RouteError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    return {
        "route_id": route.route_id,
        "adapter_id": route.adapter_id,
        "policy_rule": route.policy_rule,
        "available": True,
        "expires_at": route.expires_at.isoformat(),
    }


@app.get("/internal/routes", dependencies=[Depends(internal_route_auth)])
def internal_route_list(project: str) -> list[dict]:
    try:
        return router.list_approved(project)
    except RouteError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


app.include_router(proxy_router)

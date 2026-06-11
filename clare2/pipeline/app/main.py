"""CLARE₂ control plane and authenticated policy proxy."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from . import corpus, distiller, lifecycle, metrics, summarizer
from .proxy import router_api as proxy_router
from .registry import RegistryError
from .runtime import controller, initialize_registry, maintenance, registry
from .security import bearer_dependency, secret_value, verify_callback

logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","component":"%(name)s","event":"%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger(__name__)
app = FastAPI(title="CLARE₂ Policy Proxy", version="1.0.0")
scheduler = BackgroundScheduler(timezone="UTC")
operator_auth = bearer_dependency(secret_value("CLARE2_OPERATOR_TOKEN"))


class TrainingDonePayload(BaseModel):
    adapter_id: str
    run_id: str
    loss: float | None = None
    epoch_losses: list[float] = Field(default_factory=list)


class SummarizePayload(BaseModel):
    reference_at: datetime | None = None


@app.on_event("startup")
def startup() -> None:
    initialize_registry()
    metrics.start_metrics_server()
    scheduler.add_job(distiller.run_daily, "cron", hour=22, minute=0, id="distill_daily")
    scheduler.add_job(summarizer.run_scheduled, "cron", hour=22, minute=30, id="summarize")
    scheduler.add_job(corpus.assemble, "cron", hour=23, minute=30, id="corpus_assemble")
    scheduler.add_job(lifecycle.drain_and_stop_infer, "cron", hour=23, minute=45, id="drain")
    scheduler.add_job(lifecycle.start_training, "cron", hour=0, minute=0, id="train")
    scheduler.start()
    try:
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
    for epoch, loss in enumerate(payload.epoch_losses, 1):
        metrics.training_loss_by_epoch.labels(epoch=str(epoch)).set(loss)
    if payload.loss is not None:
        metrics.training_loss_final.set(payload.loss)
    background_tasks.add_task(lifecycle.complete_training, payload.adapter_id, payload.run_id)
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
        raise HTTPException(status_code=409, detail="adapter has no approved evaluation")
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


@app.post("/distill/trigger", dependencies=[Depends(operator_auth)])
def trigger_distill(background_tasks: BackgroundTasks) -> dict:
    background_tasks.add_task(distiller.run_daily)
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


app.include_router(proxy_router)

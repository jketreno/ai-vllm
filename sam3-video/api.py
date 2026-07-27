"""Private session RPC for SAM video propagation and checkpoint export."""

from __future__ import annotations

import io
import json
import os
import secrets
import shutil
import threading
import uuid
import zipfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

PROTOCOL_VERSION = "1"
SESSION_ROOT = Path(os.environ.get("SAM3_VIDEO_SESSION_ROOT", "/data/sessions"))
TOKEN_FILE = Path(
    os.environ.get("SAM3_VIDEO_TOKEN_FILE", "/run/secrets/phai_service_token")
)
MAX_VIDEO_BYTES = int(
    os.environ.get("SAM3_VIDEO_MAX_BYTES", str(20 * 1024 * 1024 * 1024))
)
BACKEND = os.environ.get("SAM3_VIDEO_BACKEND", "box-keyframe-fallback")
MODEL_REVISION = os.environ.get("SAM3_VIDEO_MODEL_REVISION", "not-loaded")
app = FastAPI(title="SAM3 video worker", version="0.1.0")
cancelled: set[str] = set()
session_locks: dict[str, threading.Lock] = {}


def _token() -> str:
    try:
        value = TOKEN_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        value = os.environ.get("SAM3_VIDEO_TOKEN", "")
    if len(value) < 32:
        raise HTTPException(503, "service credential is not configured")
    return value


def require_service(authorization: str = Header(default="")) -> None:
    scheme, _, supplied = authorization.partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(
        supplied, _token()
    ):
        raise HTTPException(401, "invalid service bearer token")


class Seed(BaseModel):
    object_id: str = Field(min_length=1, max_length=200)
    timestamp_us: int = Field(ge=0)
    box: list[float] = Field(min_length=4, max_length=4)
    label: str | None = Field(default=None, max_length=500)


class SeedBatch(BaseModel):
    seeds: list[Seed] = Field(min_length=1, max_length=100)


class PropagationRequest(BaseModel):
    sample_interval_us: int = Field(default=500_000, ge=100_000, le=10_000_000)
    checkpoint_interval_us: int = Field(
        default=10_000_000, ge=1_000_000, le=120_000_000
    )


def _session(session_id: str) -> Path:
    try:
        parsed = uuid.UUID(session_id)
    except ValueError as error:
        raise HTTPException(404, "session not found") from error
    path = SESSION_ROOT / str(parsed)
    if not path.is_dir():
        raise HTTPException(404, "session not found")
    return path


@app.get("/health/live")
def live():
    return {"status": "ok"}


@app.get("/health/ready")
def ready(response: Response):
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    status = "ready" if BACKEND == "sam3.1" else "degraded"
    if status == "degraded":
        response.status_code = 206
    return {
        "status": status,
        "backend": BACKEND,
        "model_revision": MODEL_REVISION,
    }


@app.get("/v1/capabilities", dependencies=[Depends(require_service)])
def capabilities():
    return {
        "protocol_version": PROTOCOL_VERSION,
        "worker": "sam3-video",
        "backend": BACKEND,
        "model_revision": MODEL_REVISION,
        "operations": [
            "session_create",
            "seed_boxes",
            "propagate",
            "checkpoint",
            "cancel",
            "masklet_export",
        ],
    }


@app.post("/v1/sessions", dependencies=[Depends(require_service)])
async def create_session(video: UploadFile = File(...)):
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    session_id = str(uuid.uuid4())
    path = SESSION_ROOT / session_id
    path.mkdir(mode=0o700)
    destination = path / "source"
    size = 0
    with destination.open("wb") as output:
        while chunk := await video.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_VIDEO_BYTES:
                shutil.rmtree(path)
                raise HTTPException(413, "video exceeds configured limit")
            output.write(chunk)
    capture = cv2.VideoCapture(str(destination))
    if not capture.isOpened():
        shutil.rmtree(path)
        raise HTTPException(415, "video decoder could not open input")
    metadata = {
        "session_id": session_id,
        "size_bytes": size,
        "fps": capture.get(cv2.CAP_PROP_FPS),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    capture.release()
    (path / "session.json").write_text(json.dumps(metadata), encoding="utf-8")
    (path / "seeds.json").write_text("[]", encoding="utf-8")
    session_locks[session_id] = threading.Lock()
    return metadata


@app.post("/v1/sessions/{session_id}/seeds", dependencies=[Depends(require_service)])
def add_seeds(session_id: str, request: SeedBatch):
    path = _session(session_id)
    seed_path = path / "seeds.json"
    existing = json.loads(seed_path.read_text(encoding="utf-8"))
    existing.extend(seed.model_dump() for seed in request.seeds)
    seed_path.write_text(json.dumps(existing), encoding="utf-8")
    return {"session_id": session_id, "seed_count": len(existing)}


def _propagate_boxes(
    session_id: str, path: Path, request: PropagationRequest
) -> dict:
    metadata = json.loads((path / "session.json").read_text(encoding="utf-8"))
    seeds = json.loads((path / "seeds.json").read_text(encoding="utf-8"))
    fps = float(metadata["fps"] or 25.0)
    step = max(1, round(fps * request.sample_interval_us / 1_000_000))
    masklet = path / "masklets"
    masklet.mkdir(exist_ok=True)
    outputs = []
    for frame_index in range(0, metadata["frame_count"], step):
        if session_id in cancelled:
            return {"status": "cancelled", "frames": len(outputs)}
        timestamp_us = round(frame_index / fps * 1_000_000)
        for seed in seeds:
            x1, y1, x2, y2 = [round(value) for value in seed["box"]]
            mask = np.zeros(
                (metadata["height"], metadata["width"]), dtype=np.uint8
            )
            mask[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)] = 255
            filename = f"{seed['object_id']}-{timestamp_us}.png"
            cv2.imwrite(str(masklet / filename), mask)
            outputs.append(
                {
                    "object_id": seed["object_id"],
                    "timestamp_us": timestamp_us,
                    "mask": filename,
                    "confidence": 0.1,
                    "provisional": True,
                }
            )
    manifest = {
        "status": "degraded" if BACKEND != "sam3.1" else "succeeded",
        "backend": BACKEND,
        "model_revision": MODEL_REVISION,
        "masklets": outputs,
    }
    (path / "checkpoint.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


@app.post(
    "/v1/sessions/{session_id}/propagate",
    dependencies=[Depends(require_service)],
)
def propagate(session_id: str, request: PropagationRequest):
    path = _session(session_id)
    lock = session_locks.setdefault(session_id, threading.Lock())
    if not lock.acquire(blocking=False):
        raise HTTPException(409, "propagation is already active")
    try:
        cancelled.discard(session_id)
        return _propagate_boxes(session_id, path, request)
    finally:
        lock.release()


@app.get(
    "/v1/sessions/{session_id}/checkpoint",
    dependencies=[Depends(require_service)],
)
def checkpoint(session_id: str):
    path = _session(session_id) / "checkpoint.json"
    if not path.exists():
        raise HTTPException(404, "checkpoint not available")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post(
    "/v1/sessions/{session_id}/cancel",
    dependencies=[Depends(require_service)],
)
def cancel(session_id: str):
    _session(session_id)
    cancelled.add(session_id)
    return {"session_id": session_id, "status": "cancelling"}


@app.get(
    "/v1/sessions/{session_id}/export",
    dependencies=[Depends(require_service)],
)
def export(session_id: str):
    path = _session(session_id)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename in ("session.json", "seeds.json", "checkpoint.json"):
            candidate = path / filename
            if candidate.exists():
                archive.write(candidate, filename)
        for mask in sorted((path / "masklets").glob("*.png")):
            archive.write(mask, f"masklets/{mask.name}")
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{session_id}.zip"'},
    )

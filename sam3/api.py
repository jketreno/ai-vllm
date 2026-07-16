"""Internal HTTP API for the SAM3 model shared with the Streamlit UI."""

import base64
import io
import json

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from managers.annotation_manager import SAM3Annotator


app = FastAPI(title="ai-vllm SAM3 API", version="1.0.0")
annotator = SAM3Annotator()
COLORS = (
    (239, 68, 68), (34, 197, 94), (59, 130, 246), (234, 179, 8),
    (168, 85, 247), (236, 72, 153), (20, 184, 166), (249, 115, 22),
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": annotator.model is not None}


async def _read_request(file, prompts):
    try:
        concepts = json.loads(prompts)
        if not isinstance(concepts, list) or not concepts:
            raise ValueError
        concepts = [value for value in concepts if isinstance(value, str) and value.strip()][:24]
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except (ValueError, json.JSONDecodeError, UnidentifiedImageError, OSError) as error:
        raise HTTPException(400, "Invalid image or JSON prompt list") from error
    return image, concepts


def _normalize_mask(mask, canvas):
    mask = np.squeeze(mask).astype(bool)
    if mask.shape == canvas.shape[:2]:
        return mask
    return cv2.resize(
        mask.astype(np.uint8),
        (canvas.shape[1], canvas.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def _segment_image(image, concepts, threshold):
    _, processor = annotator.initialize()
    processor.confidence_threshold = 0.05
    canvas = np.asarray(image).copy().astype(np.float32)
    state = processor.set_image(image)
    segments = []
    for concept_index, concept in enumerate(concepts):
        processor.reset_all_prompts(state)
        output = processor.set_text_prompt(state=state, prompt=concept)
        scores = output["scores"].detach().cpu().numpy()
        boxes = output["boxes"].detach().cpu().numpy()
        masks = output["masks"].detach().cpu().numpy()
        for detection_index, score in enumerate(scores):
            if float(score) < threshold:
                continue
            mask = _normalize_mask(masks[detection_index], canvas)
            color = COLORS[(concept_index + detection_index) % len(COLORS)]
            canvas[mask] = canvas[mask] * 0.52 + np.array(color) * 0.48
            segments.append({
                "concept": concept,
                "score": round(float(score), 4),
                "box": [round(float(value), 1) for value in boxes[detection_index]],
                "color": "#" + "".join(f"{channel:02x}" for channel in color),
                "area_pixels": int(mask.sum()),
            })
    return canvas, segments


def _response(image, canvas, segments):
    encoded = io.BytesIO()
    Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).save(encoded, format="PNG")
    overlay = base64.b64encode(encoded.getvalue()).decode("ascii")
    return {
        "segments": segments,
        "overlay_image": f"data:image/png;base64,{overlay}",
        "width": image.width,
        "height": image.height,
    }


@app.post("/v1/segment")
async def segment(
    file: UploadFile = File(...),
    prompts: str = Form(...),
    threshold: float = Form(0.15),
):
    image, concepts = await _read_request(file, prompts)
    with annotator.inference_lock, torch.inference_mode():
        canvas, segments = _segment_image(image, concepts, threshold)
    return _response(image, canvas, segments)

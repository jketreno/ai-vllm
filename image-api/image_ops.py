"""CPU-only image composition and response formatting."""

import base64
import io

import cv2
import numpy as np
from fastapi import HTTPException
from PIL import Image


COLORS = ((239, 68, 68), (34, 197, 94), (59, 130, 246), (234, 179, 8),
          (168, 85, 247), (236, 72, 153), (20, 184, 166), (249, 115, 22))


def png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def image_response(image: Image.Image) -> dict:
    return {
        "width": image.width,
        "height": image.height,
        "image_png_base64": base64.b64encode(png_bytes(image)).decode("ascii"),
    }


def rpc_image(result: dict) -> Image.Image:
    attachment = next((item for item in result.get("attachments", []) if item.get("name") == "image"), None)
    if not attachment:
        raise HTTPException(502, "model worker did not return an image")
    return Image.open(io.BytesIO(base64.b64decode(attachment["data_base64"]))).convert("RGB")


def segment_response(source: Image.Image, result: dict, fields: set[str]) -> dict:
    attachment_map = {item["name"]: item for item in result.get("attachments", [])}
    canvas = np.asarray(source.convert("RGB")).copy().astype(np.float32)
    segments = []
    for index, raw in enumerate(result.get("data", {}).get("segments", [])):
        attachment = attachment_map.get(raw["mask_attachment"])
        if not attachment:
            raise HTTPException(502, "SAM3 worker omitted a segment mask")
        mask_bytes = base64.b64decode(attachment["data_base64"])
        mask_image = Image.open(io.BytesIO(mask_bytes)).convert("L")
        mask = np.asarray(mask_image) > 0
        color = COLORS[index % len(COLORS)]
        canvas[mask] = canvas[mask] * 0.52 + np.array(color) * 0.48
        segment = {
            **raw,
            "color": "#" + "".join(f"{channel:02x}" for channel in color),
            "area_pixels": int(mask.sum()),
        }
        segment.pop("mask_attachment", None)
        if "mask" in fields:
            segment["mask"] = "data:image/png;base64," + base64.b64encode(mask_bytes).decode("ascii")
        if "polygon" in fields:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygon = []
            if contours:
                largest = max(contours, key=cv2.contourArea)
                approx = cv2.approxPolyDP(largest, 0.02 * cv2.arcLength(largest, True), True)
                polygon = [[round(float(point[0][0]), 1), round(float(point[0][1]), 1)] for point in approx]
            segment["polygon"] = polygon
        segments.append(segment)
    overlay = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    return {
        "segments": segments,
        "overlay_image": "data:image/png;base64," + base64.b64encode(png_bytes(overlay)).decode("ascii"),
        "width": source.width,
        "height": source.height,
    }


def outpaint_canvas(image: Image.Image, width: int, height: int, anchor: str):
    if width < image.width or height < image.height:
        raise HTTPException(400, "target dimensions must be >= source dimensions")
    positions = {
        "center": ((width - image.width) // 2, (height - image.height) // 2),
        "top-left": (0, 0), "top-right": (width - image.width, 0),
        "bottom-left": (0, height - image.height),
        "bottom-right": (width - image.width, height - image.height),
        "top": ((width - image.width) // 2, 0),
        "bottom": ((width - image.width) // 2, height - image.height),
        "left": (0, (height - image.height) // 2),
        "right": (width - image.width, (height - image.height) // 2),
    }
    if anchor not in positions:
        raise HTTPException(400, f"anchor must be one of {sorted(positions)}")
    position = positions[anchor]
    canvas = Image.new("RGB", (width, height))
    canvas.paste(image, position)
    mask = Image.new("L", (width, height), 255)
    mask.paste(Image.new("L", image.size, 0), position)
    return canvas, mask


def transform_image(image: Image.Image, crop, rotate_degrees: float, expand_canvas: bool):
    if any(value is not None for value in crop):
        if any(value is None for value in crop):
            raise HTTPException(400, "all crop fields are required together")
        left, top, width, height = map(int, crop)
        if width <= 0 or height <= 0 or left < 0 or top < 0 or left + width > image.width or top + height > image.height:
            raise HTTPException(400, "crop region falls outside image bounds")
        image = image.crop((left, top, left + width, top + height))
    if rotate_degrees % 360:
        image = image.rotate(-rotate_degrees, expand=expand_canvas, resample=Image.Resampling.BICUBIC)
    return image

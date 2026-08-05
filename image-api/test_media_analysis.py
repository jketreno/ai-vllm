"""Contract tests for ketr.phai structured media operations."""

import importlib.util
import io
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from PIL import Image

MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR))
media_inference = importlib.import_module("media_inference")
spec = importlib.util.spec_from_file_location(
    "media_analysis_contract", MODULE_DIR / "media_analysis.py"
)
media = importlib.util.module_from_spec(spec)
sys.modules["media_analysis_contract"] = media
spec.loader.exec_module(media)


class AsyncUpload:
    content_type = "image/png"

    async def read(self, _size: int) -> bytes:
        return b"\x89PNG\r\n\x1a\n"


def image_upload() -> AsyncUpload:
    return AsyncUpload()


@pytest.mark.asyncio
async def test_completion_labels_workload_and_uses_extended_timeout():
    captured = {}

    class FakeClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, _url, headers, json):
            captured["headers"] = headers
            captured["request"] = json
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"value":"ok"}'}}]},
                request=httpx.Request("POST", "http://policy"),
            )

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    with patch.object(
        media_inference.httpx, "AsyncClient", FakeClient
    ), patch.object(
        media_inference, "_policy_token", return_value="token"
    ):
        result = await media._completion(
            "prompt", schema, workload="semantic_report"
        )

    assert result == {"value": "ok"}
    assert captured["headers"]["X-Inference-Workload"] == "semantic_report"
    assert captured["timeout"].read == media.INFERENCE_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_completion_preserves_permanent_upstream_rejection():
    class FakeClient:
        def __init__(self, timeout):
            del timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, _url, headers, json):
            del headers, json
            return httpx.Response(
                400,
                json={"message": "maximum context length exceeded"},
                request=httpx.Request("POST", "http://policy"),
            )

    with patch.object(
        media_inference.httpx, "AsyncClient", FakeClient
    ), patch.object(
        media_inference, "_policy_token", return_value="token"
    ):
        with pytest.raises(media.HTTPException) as raised:
            await media._completion("prompt", {"type": "object"})

    assert raised.value.status_code == 400
    assert "maximum context length exceeded" in raised.value.detail


@pytest.mark.asyncio
async def test_completion_labels_max_tokens_truncation_as_output_truncated():
    class FakeClient:
        def __init__(self, timeout):
            del timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, _url, headers, json):
            del headers, json
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": '{"value": "unterminat'},
                        }
                    ]
                },
                request=httpx.Request("POST", "http://policy"),
            )

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    with patch.object(
        media_inference.httpx, "AsyncClient", FakeClient
    ), patch.object(
        media_inference, "_policy_token", return_value="token"
    ):
        with pytest.raises(media.HTTPException) as raised:
            await media._completion(
                "prompt", schema, workload="semantic_report"
            )

    assert raised.value.status_code == 503
    assert raised.value.detail["reason_code"] == "output_truncated"
    assert raised.value.headers["Retry-After"] == "5"


def _truncated_response(content: str):
    return httpx.Response(
        200,
        json={
            "choices": [
                {"finish_reason": "length", "message": {"content": content}}
            ]
        },
        request=httpx.Request("POST", "http://policy"),
    )


class _TruncatedClient:
    def __init__(self, content: str):
        self._content = content

    def __call__(self, timeout):
        del timeout
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def post(self, _url, headers, json):
        del headers, json
        return _truncated_response(self._content)


@pytest.mark.asyncio
async def test_completion_repairs_truncation_mid_array():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "caption": {"type": "string"},
            "summary": {"type": "string"},
            "observations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                },
            },
            "known_facts": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["caption", "summary", "observations", "known_facts"],
    }
    content = (
        '{"caption": "A dog in a park.", "summary": "Longer summary text.", '
        '"observations": [{"summary": "first, complete"}, '
        '{"summary": "second, still writ'
    )
    with patch.object(
        media_inference.httpx, "AsyncClient", _TruncatedClient(content)
    ), patch.object(
        media_inference, "_policy_token", return_value="token"
    ):
        result = await media._completion(
            "prompt", schema, workload="semantic_report"
        )

    assert result["caption"] == "A dog in a park."
    assert result["summary"] == "Longer summary text."
    assert result["observations"] == [{"summary": "first, complete"}]
    assert result["known_facts"] == []


@pytest.mark.asyncio
async def test_completion_repairs_truncation_inside_nested_object():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "caption": {"type": "string"},
            "summary": {"type": "string"},
            "known_facts": {"type": "array", "items": {"type": "string"}},
            "place_resolution": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "confidence": {"type": "number"},
                    "visual_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "spatial_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "required": ["caption", "summary", "known_facts", "place_resolution"],
    }
    content = (
        '{"caption": "A castle on a hill.", "summary": "Detailed summary.", '
        '"known_facts": ["fact one"], '
        '"place_resolution": {"status": "possible", "confidence": 0.6, '
        '"visual_evidence": ["matches architectural style"], '
        '"spatial_evidence": ["GPS coordinates are very cl'
    )
    with patch.object(
        media_inference.httpx, "AsyncClient", _TruncatedClient(content)
    ), patch.object(
        media_inference, "_policy_token", return_value="token"
    ):
        result = await media._completion(
            "prompt", schema, workload="semantic_report"
        )

    assert result["caption"] == "A castle on a hill."
    assert result["known_facts"] == ["fact one"]
    place = result["place_resolution"]
    assert place["status"] == "possible"
    assert place["visual_evidence"] == ["matches architectural style"]
    assert place["spatial_evidence"] == []


@pytest.mark.asyncio
async def test_completion_repairs_truncation_inside_focus_target_box():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "caption": {"type": "string"},
            "summary": {"type": "string"},
            "focus_targets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "display_label": {"type": "string"},
                        "box": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "w": {"type": "number"},
                                "h": {"type": "number"},
                            },
                        },
                    },
                },
            },
        },
        "required": ["caption", "summary", "focus_targets"],
    }
    content = (
        '{"caption": "A dog in a park.", "summary": "Longer summary text.", '
        '"focus_targets": [{"display_label": "dog", '
        '"box": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}}, '
        '{"display_label": "bench", "box": {"x": 0.5, "y": 0.'
    )
    with patch.object(
        media_inference.httpx, "AsyncClient", _TruncatedClient(content)
    ), patch.object(
        media_inference, "_policy_token", return_value="token"
    ):
        result = await media._completion(
            "prompt", schema, workload="semantic_report"
        )

    assert result["focus_targets"] == [
        {"display_label": "dog", "box": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}}
    ]


@pytest.mark.asyncio
async def test_completion_still_returns_503_when_cutoff_is_too_early_to_repair():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "caption": {"type": "string"},
            "summary": {"type": "string"},
        },
        "required": ["caption", "summary"],
    }
    content = '{"caption": "cut off mid str'
    with patch.object(
        media_inference.httpx, "AsyncClient", _TruncatedClient(content)
    ), patch.object(
        media_inference, "_policy_token", return_value="token"
    ):
        with pytest.raises(media.HTTPException) as raised:
            await media._completion(
                "prompt", schema, workload="semantic_report"
            )

    assert raised.value.status_code == 503
    assert raised.value.detail["reason_code"] == "output_truncated"


@pytest.mark.asyncio
async def test_completion_preserves_generic_json_decode_failure():
    class FakeClient:
        def __init__(self, timeout):
            del timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, _url, headers, json):
            del headers, json
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "not json at all"},
                        }
                    ]
                },
                request=httpx.Request("POST", "http://policy"),
            )

    with patch.object(
        media_inference.httpx, "AsyncClient", FakeClient
    ), patch.object(
        media_inference, "_policy_token", return_value="token"
    ):
        with pytest.raises(media.HTTPException) as raised:
            await media._completion("prompt", {"type": "object"})

    assert raised.value.status_code == 503
    assert "structured media analysis unavailable" in raised.value.detail


@pytest.mark.asyncio
async def test_capacity_preserves_saturation_retry_after():
    class FakeClient:
        def __init__(self, timeout):
            assert timeout == 5

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, _url, headers):
            assert headers["X-Inference-Workload"] == "semantic_report"
            return httpx.Response(
                429,
                json={"available": False, "retry_after": 75},
                headers={"Retry-After": "75"},
                request=httpx.Request("GET", "http://policy/capacity"),
            )

    with patch.object(
        media_inference.httpx, "AsyncClient", FakeClient
    ), patch.object(
        media_inference, "_policy_token", return_value="token"
    ):
        response = await media.inference_capacity("semantic_report")

    assert response.status_code == 429
    assert response.headers["retry-after"] == "75"


def _semantic_report_result(**overrides) -> dict:
    result = {
        "caption": "A blue square.",
        "concise_caption": "Blue square",
        "narrative_role": "detail",
        "scale": "close",
        "confidence": 0.9,
        "concepts": ["blue", "square"],
        "focus_targets": [
            {
                "display_label": "blue square in the center",
                "sam_prompt": "blue square",
                "role": "primary",
                "subject_type": "object",
                "extent": "whole_subject",
                "segmentability": "high",
                "confidence": 0.95,
                "location": {"horizontal": "center", "vertical": "center"},
                "box": {"x": 0.3, "y": 0.3, "w": 0.4, "h": 0.4},
                "gaze_direction": "not_applicable",
                "reason": "The square is the principal visible subject.",
            }
        ],
        "visible_text": [],
        "observations": [],
        "summary": "A blue square photographed outdoors.",
        "concise_summary": "Blue square outdoors.",
        "known_facts": ["The image shows a blue square."],
        "inferences": [],
        "evidence_types": ["semantic_summary"],
        "place_resolution": {
            "status": "none",
            "candidate_id": None,
            "name": None,
            "confidence": 0,
            "visual_evidence": [],
            "spatial_evidence": [],
        },
    }
    result.update(overrides)
    return result


@pytest.mark.asyncio
async def test_semantic_image_adds_model_and_schema_provenance():
    result = _semantic_report_result()
    with patch.object(media, "_completion", new=AsyncMock(return_value=result)):
        response = await media.semantic_image(image_upload(), observations=None)
    assert response["schema_version"] == "1"
    assert response["contract_version"] == "0.2.0"
    assert len(response["schema_fingerprint"]) == 64
    assert response["prompt_version"] == "phai-report-v7"
    assert response["caption"] == "A blue square."
    assert response["summary"] == "A blue square photographed outdoors."


@pytest.mark.asyncio
async def test_visual_semantics_uses_visual_only_schema_and_workload():
    completion = AsyncMock(return_value=_semantic_report_result())

    with patch.object(media, "_completion", new=completion):
        response = await media.visual_semantics_image(image_upload())

    assert response["caption"] == "A blue square."
    assert completion.await_args.args[1] == media.VISUAL_SEMANTIC_SCHEMA
    assert "summary" not in media.VISUAL_SEMANTIC_SCHEMA["properties"]
    assert completion.await_args.kwargs["workload"] == "visual_semantics"
    assert len(completion.await_args.args[2]) == 1


@pytest.mark.asyncio
async def test_context_report_is_text_only_and_receives_current_context():
    completion = AsyncMock(return_value=_semantic_report_result())
    request = media.ContextReportRequest(
        asset={"id": "asset-1", "original_filename": "family.jpg"},
        visual_semantics={"caption": "Two people beside a lake."},
        observations=[
            {
                "observation_type": "context_enrichment",
                "payload": {"status": "resolved"},
            }
        ],
        accepted_identities=[{"display_name": "Alice"}],
        location_override={
            "latitude": 44.5,
            "longitude": -107.5,
            "source_label": "Wyoming",
        },
    )

    with patch.object(media, "_completion", new=completion):
        response = await media.context_report(request)

    prompt = completion.await_args.args[0]
    assert "Two people beside a lake" in prompt
    assert "Alice" in prompt
    assert "Wyoming" in prompt
    assert completion.await_args.args[1] == media.CONTEXT_REPORT_SCHEMA
    assert len(completion.await_args.args) == 2
    assert completion.await_args.kwargs["workload"] == "context_report"
    assert response["summary"] == "A blue square photographed outdoors."


@pytest.mark.asyncio
async def test_semantic_image_forwards_evidence_to_the_prompt():
    completion = AsyncMock(return_value=_semantic_report_result())
    observations = json.dumps(
        [{"observation_type": "metadata", "payload": {"capture_time": "2022"}}]
    )
    with patch.object(media, "_completion", new=completion):
        await media.semantic_image(image_upload(), observations=observations)

    prompt = completion.await_args.args[0]
    assert "capture_time" in prompt
    assert "ordered photographic focus plan" in prompt
    assert "arbitrary leaves" in prompt
    assert "literal 1-6 word noun phrase" in prompt
    assert "normalized {x, y, w, h}" in prompt
    assert "gaze_direction" in prompt
    assert "narrative_role" in prompt
    assert completion.await_args.kwargs["workload"] == "semantic_report"
    assert completion.await_args.kwargs["max_tokens"] == media.SEMANTIC_IMAGE_MAX_TOKENS


@pytest.mark.asyncio
async def test_semantic_window_rejects_timestamp_count_mismatch():
    with pytest.raises(media.HTTPException, match="timestamps"):
        await media.semantic_window(
            frames=[image_upload()],
            timestamps_us=json.dumps([0, 1]),
            transcript=None,
        )


def test_semantic_report_schema_forbids_untracked_fields():
    assert media.SEMANTIC_REPORT_SCHEMA["additionalProperties"] is False
    assert media.OBSERVATION_SCHEMA["additionalProperties"] is False
    place_schema = media.SEMANTIC_REPORT_SCHEMA["properties"]["place_resolution"]
    assert place_schema["additionalProperties"] is False
    assert set(place_schema["properties"]["status"]["enum"]) == {
        "none",
        "possible",
        "resolved",
    }
    focus_schema = media.SEMANTIC_REPORT_SCHEMA["properties"]["focus_targets"]
    assert focus_schema["maxItems"] == 8
    assert focus_schema["items"]["additionalProperties"] is False
    # sam_prompts is derived server-side from focus_targets[].sam_prompt
    # (_normalize_focus_plan); asking the model for it too let it satisfy
    # the schema with a list disconnected from focus_targets.
    assert "sam_prompts" not in media.SEMANTIC_REPORT_SCHEMA["properties"]
    assert "sam_prompts" not in media.SEMANTIC_REPORT_SCHEMA["required"]
    # uncertainties has zero downstream readers; dropped entirely.
    assert "uncertainties" not in media.SEMANTIC_REPORT_SCHEMA["properties"]
    assert "uncertainties" not in media.SEMANTIC_REPORT_SCHEMA["required"]
    # id/priority are server-assigned from array order in
    # _normalize_focus_plan; asking the model for them is pure ceremony.
    focus_item = focus_schema["items"]
    assert "id" not in focus_item["properties"]
    assert "priority" not in focus_item["properties"]
    assert "box" in focus_item["required"]
    assert "gaze_direction" in focus_item["required"]
    assert set(focus_item["properties"]["gaze_direction"]["enum"]) == {
        "toward_camera",
        "left",
        "right",
        "up",
        "down",
        "away",
        "not_applicable",
    }
    observation_item = media.SEMANTIC_REPORT_SCHEMA["properties"]["observations"][
        "items"
    ]
    assert "evidence" not in observation_item["properties"]
    assert media.SEMANTIC_REPORT_SCHEMA["properties"]["observations"][
        "maxItems"
    ] == 24
    assert media.SEMANTIC_REPORT_SCHEMA["properties"]["concepts"]["maxItems"] == 24
    assert (
        media.SEMANTIC_REPORT_SCHEMA["properties"]["visible_text"]["maxItems"] == 32
    )
    narrative_role_schema = media.SEMANTIC_REPORT_SCHEMA["properties"]["narrative_role"]
    assert set(narrative_role_schema["enum"]) == {
        "establishing",
        "detail",
        "portrait",
        "action",
        "transition",
        "climax",
        "closing",
    }
    assert set(media.SEMANTIC_REPORT_SCHEMA["properties"]["scale"]["enum"]) == {
        "wide",
        "medium",
        "close",
        "detail",
    }


@pytest.mark.asyncio
async def test_semantic_image_derives_sam_prompts_from_ranked_focus_targets():
    # sam_prompts is not part of the model-facing schema (removed so the
    # model can't satisfy it with content disconnected from focus_targets),
    # but the response field is always rebuilt server-side; a stray/legacy
    # sam_prompts key on the raw model dict must still be ignored.
    result = _semantic_report_result(
        sam_prompts=["unrelated model output"],
        focus_targets=[
            {
                "display_label": "girl in a blue dress",
                "sam_prompt": "girl in blue dress",
                "role": "primary",
                "subject_type": "person",
                "extent": "whole_subject",
                "segmentability": "high",
                "confidence": 0.96,
                "location": {"horizontal": "right", "vertical": "center"},
                "box": {"x": 0.5, "y": 0.1, "w": 0.3, "h": 0.8},
                "gaze_direction": "toward_camera",
                "reason": "Principal person.",
            },
            {
                "display_label": "small flag in the pastries",
                "sam_prompt": "small flag",
                "role": "supporting",
                "subject_type": "object",
                "extent": "detail",
                "segmentability": "low",
                "confidence": 0.7,
                "location": {"horizontal": "right", "vertical": "bottom"},
                "box": {"x": 0.7, "y": 0.8, "w": 0.1, "h": 0.1},
                "gaze_direction": "not_applicable",
                "reason": "Interesting but too small for reliable segmentation.",
            },
        ],
    )
    with patch.object(media, "_completion", new=AsyncMock(return_value=result)):
        response = await media.semantic_image(image_upload(), observations=None)

    assert response["sam_prompts"] == ["girl in blue dress"]
    # id/priority are assigned server-side from array order.
    assert response["focus_targets"][0]["id"] == "focus-1"
    assert response["focus_targets"][0]["priority"] == 1
    assert response["focus_targets"][1]["id"] == "focus-2"
    assert response["focus_targets"][1]["priority"] == 2


@pytest.mark.asyncio
async def test_semantic_image_requires_grounded_poi_resolution():
    completion = AsyncMock(return_value=_semantic_report_result())
    observations = json.dumps(
        [
            {
                "observation_type": "context_enrichment",
                "payload": {
                    "facts": [
                        {
                            "kind": "poi_candidate",
                            "payload": {
                                "asserted": False,
                                "candidates": [
                                    {
                                        "overture_id": "poi-1",
                                        "name": "XYZ Cathedral",
                                        "category": "cathedral",
                                        "relationship": "inside",
                                        "distance_m": 0,
                                    }
                                ],
                            },
                        }
                    ]
                },
            }
        ]
    )

    with patch.object(media, "_completion", new=completion):
        await media.semantic_image(image_upload(), observations=observations)

    prompt = completion.await_args.args[0]
    assert "A suggestive name alone is not evidence" in prompt
    assert "select at most one supplied candidate" in prompt
    assert "both visual and spatial evidence" in prompt
    assert "XYZ Cathedral" in prompt


@pytest.mark.asyncio
async def test_semantic_image_keeps_transcription_and_diarization_independent():
    completion = AsyncMock(
        return_value=_semantic_report_result(
            summary=(
                "A party is shown. No transcript text was produced due to "
                "unavailable speaker separation."
            ),
            concise_summary="Anonymous transcript available.",
            known_facts=[
                "No transcript was produced because diarization was unavailable."
            ],
            evidence_types=["speech_status"],
        )
    )
    observations = json.dumps(
        [
            {
                "observation_type": "speech_status",
                "payload": {
                    "transcription": "available",
                    "diarization": "unavailable",
                },
            }
        ]
    )

    with patch.object(media, "_completion", new=completion):
        result = await media.semantic_image(
            image_upload(), observations=observations
        )

    prompt = completion.await_args.args[0]
    independence_rule = (
        "Diarization availability never determines transcription availability"
    )
    assert independence_rule in prompt
    assert result["prompt_version"] == "phai-report-v7"
    assert "due to unavailable speaker separation" not in result["summary"]
    assert result["summary"].endswith(
        "Anonymous speaker separation was unavailable and did not affect "
        "transcription."
    )
    assert result["known_facts"][-2:] == [
        "Transcription ran, but no transcript text segments were produced.",
        "Anonymous speaker separation was unavailable and did not affect "
        "transcription.",
    ]


@pytest.mark.asyncio
async def test_semantic_image_speech_normalization_tolerates_missing_known_facts():
    result = _semantic_report_result(
        summary="A party is shown.",
        concise_summary="Party.",
        evidence_types=["speech_status"],
    )
    del result["known_facts"]
    completion = AsyncMock(return_value=result)
    observations = json.dumps(
        [
            {
                "observation_type": "speech_status",
                "payload": {
                    "transcription": "available",
                    "diarization": "unavailable",
                },
            }
        ]
    )

    with patch.object(media, "_completion", new=completion):
        result = await media.semantic_image(
            image_upload(), observations=observations
        )

    assert result["known_facts"] == [
        "Transcription ran, but no transcript text segments were produced.",
        "Anonymous speaker separation was unavailable and did not affect "
        "transcription.",
    ]


def _jpeg_bytes(width: int, height: int, exif_orientation: int | None = None) -> bytes:
    image = Image.new("RGB", (width, height), color=(10, 20, 30))
    buffer = io.BytesIO()
    if exif_orientation is not None:
        exif = image.getexif()
        exif[0x0112] = exif_orientation
        image.save(buffer, format="JPEG", exif=exif)
    else:
        image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _png_bytes(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), color=(10, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class FixedUpload:
    def __init__(self, payload: bytes, content_type: str):
        self.content_type = content_type
        self._payload = payload

    async def read(self, _size: int) -> bytes:
        return self._payload


@pytest.mark.asyncio
async def test_downscale_shrinks_oversized_jpeg_and_applies_exif_orientation():
    # Orientation 6 = rotate 270 (needs 90 CW to display correctly), so a
    # portrait-tagged landscape buffer should come out portrait after
    # exif_transpose, then be capped to the configured max edge.
    payload = _jpeg_bytes(3000, 2000, exif_orientation=6)
    with patch.object(media_inference, "VISION_MAX_EDGE", 1568):
        result_bytes, media_type = await media_inference.read_image(
            FixedUpload(payload, "image/jpeg")
        )
    assert media_type == "image/jpeg"
    with Image.open(io.BytesIO(result_bytes)) as image:
        assert max(image.size) <= 1568
        assert image.size[0] < image.size[1]  # portrait after transpose


@pytest.mark.asyncio
async def test_downscale_leaves_small_images_byte_identical():
    payload = _jpeg_bytes(800, 600)
    with patch.object(media_inference, "VISION_MAX_EDGE", 1568):
        result_bytes, media_type = await media_inference.read_image(
            FixedUpload(payload, "image/jpeg")
        )
    assert result_bytes == payload
    assert media_type == "image/jpeg"


@pytest.mark.asyncio
async def test_downscale_keeps_png_as_png():
    payload = _png_bytes(3000, 3000)
    with patch.object(media_inference, "VISION_MAX_EDGE", 1568):
        result_bytes, media_type = await media_inference.read_image(
            FixedUpload(payload, "image/png")
        )
    assert media_type == "image/png"
    with Image.open(io.BytesIO(result_bytes)) as image:
        assert image.format == "PNG"
        assert max(image.size) <= 1568


@pytest.mark.asyncio
async def test_downscale_disabled_by_zero_max_edge():
    payload = _jpeg_bytes(3000, 2000)
    with patch.object(media_inference, "VISION_MAX_EDGE", 0):
        result_bytes, media_type = await media_inference.read_image(
            FixedUpload(payload, "image/jpeg")
        )
    assert result_bytes == payload
    assert media_type == "image/jpeg"


def test_evidence_truncation_omits_rather_than_rejects():
    observations = [
        {"observation_type": "metadata", "payload": {"note": "x" * 2000}}
        for _ in range(200)
    ]
    with patch.object(media, "EVIDENCE_MAX_CHARS", 5000):
        prompt = media._evidence_prompt(observations)
    assert "additional evidence items omitted" in prompt
    # No HTTPException should be raised for oversized evidence.


def test_evidence_within_budget_is_not_truncated():
    observations = [{"observation_type": "metadata", "payload": {"note": "small"}}]
    prompt = media._evidence_prompt(observations)
    assert "omitted" not in prompt
    assert "small" in prompt


@pytest.mark.asyncio
async def test_semantic_window_uses_configured_max_tokens():
    completion = AsyncMock(return_value=_semantic_report_result())
    with patch.object(media, "_completion", new=completion):
        await media.semantic_window(
            frames=[image_upload()],
            timestamps_us=json.dumps([0]),
            transcript=None,
            observations=None,
        )
    assert (
        completion.await_args.kwargs["max_tokens"] == media.SEMANTIC_WINDOW_MAX_TOKENS
    )


def test_schema_fingerprint_matches_recomputed_hash():
    from media_contracts import schema_fingerprint

    assert media_inference.SCHEMA_FINGERPRINT == schema_fingerprint()

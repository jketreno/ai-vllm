"""Contract tests for ketr.phai structured media operations."""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

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
        "confidence": 0.9,
        "concepts": ["blue", "square"],
        "sam_prompts": ["blue square"],
        "visible_text": [],
        "uncertainties": [],
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
    assert response["contract_version"] == "0.1.0"
    assert len(response["schema_fingerprint"]) == 64
    assert response["prompt_version"] == "phai-report-v4"
    assert response["caption"] == "A blue square."
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
    assert completion.await_args.kwargs["workload"] == "semantic_report"


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
    assert result["prompt_version"] == "phai-report-v4"
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

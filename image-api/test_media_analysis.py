"""Contract tests for ketr.phai structured media operations."""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR))
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
async def test_semantic_image_adds_model_and_schema_provenance():
    result = {
        "caption": "A blue square.",
        "concise_caption": "Blue square",
        "confidence": 0.9,
        "concepts": ["blue", "square"],
        "sam_prompts": ["blue square"],
        "visible_text": [],
        "uncertainties": [],
        "observations": [],
    }
    with patch.object(media, "_completion", new=AsyncMock(return_value=result)):
        response = await media.semantic_image(image_upload())
    assert response["schema_version"] == "1"
    assert response["contract_version"] == "0.1.0"
    assert len(response["schema_fingerprint"]) == 64
    assert response["prompt_version"] == "phai-media-v1"
    assert response["caption"] == "A blue square."


@pytest.mark.asyncio
async def test_semantic_window_rejects_timestamp_count_mismatch():
    with pytest.raises(media.HTTPException, match="timestamps"):
        await media.semantic_window(
            frames=[image_upload()],
            timestamps_us=json.dumps([0, 1]),
            transcript=None,
        )


def test_semantic_schema_forbids_untracked_fields():
    assert media.SEMANTIC_SCHEMA["additionalProperties"] is False
    assert media.OBSERVATION_SCHEMA["additionalProperties"] is False


@pytest.mark.asyncio
async def test_report_synthesis_keeps_transcription_and_diarization_independent():
    completion = AsyncMock(
        return_value={
            "summary": "Transcription completed without speaker labels.",
            "concise_summary": "Anonymous transcript available.",
            "known_facts": [],
            "inferences": [],
            "uncertainties": [],
            "evidence_types": ["speech_status"],
        }
    )
    request = media.EvidenceRequest(
        asset={"id": "asset"},
        observations=[
            {
                "observation_type": "speech_status",
                "payload": {
                    "transcription": "available",
                    "diarization": "unavailable",
                },
            }
        ],
    )

    with patch.object(media, "_completion", new=completion):
        result = await media.report_synthesis(request)

    prompt = completion.await_args.args[0]
    independence_rule = (
        "Diarization availability never determines transcription availability"
    )
    assert independence_rule in prompt
    assert result["prompt_version"] == "phai-report-v2"

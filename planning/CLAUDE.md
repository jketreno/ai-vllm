# Project: Photo Management Suite — AI Pipeline

## What this is
An existing photo management suite (EXIF, GPS tagging, dates, database) being extended
with an AI pipeline: object cataloging, captioning, face clustering with human review,
composition-aware auto-crops with outpaint/inpaint, event enrichment from date+GPS, and
photobook page/story generation.

**Read `PIPELINE.md` before making architecture decisions. It is the design source of
truth from the planning session.**

## Environment
- Model services run on an NVIDIA DGX Spark (128 GB unified memory):
  - SAM3 — open-vocabulary segmentation (concept-promptable)
  - Qwen3.6-27B-FP8 — captions, topics, narratives
  - qwen-image-edit — outpaint/inpaint
- A face embedding model (InsightFace/ArcFace class) still needs to be added.
- Memory is tight with all three services hot; analysis and image-edit workloads are
  scheduled as separate phases (see PIPELINE.md §2, §11).

## Core architecture rules
- All model outputs land in the derived data layer (DB); downstream features read from
  the DB, never re-invoke models for data that already exists.
- Batch and single-photo processing use the identical per-photo DAG; batch = fan-out.
- Originals are immutable. Crops, outpaints, and inpaints are versioned derivatives
  with the operation and parameters recorded.
- Every stage write is idempotent, keyed on (photo_id, stage, model_version).
- Face identity decisions from the user (Yes/No) become persistent must-link /
  cannot-link constraints; re-clustering must never undo confirmed decisions.

## Build order
Schema + queue → SAM3 catalog → faces + review UI → captions → crop engine →
enrichment + photobook. See PIPELINE.md §12 for rationale.

## Before assuming anything
PIPELINE.md §13 lists open questions (DB engine, Qwen VL vs text-only, service API
shapes, UI stack, storage layout). Ask the user rather than guessing; update this file
and PIPELINE.md with the answers.

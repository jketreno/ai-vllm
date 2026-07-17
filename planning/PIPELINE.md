# Photo AI Pipeline — Architecture Spec

This document is the design handoff for adding AI-driven cataloging, captioning, face
clustering, crop/edit suggestions, event enrichment, and photobook generation to an
existing photo management suite. It was produced in a design session; treat it as the
source of truth for architecture decisions unless the user overrides it.

## 1. Existing system (do not rebuild)

The photo management suite already handles:
- EXIF extraction and storage
- GPS tagging
- Date/time metadata
- A database of raw photo details

All new work layers on top of this. New AI-derived data goes into new tables (Section 9),
keyed to the existing photo records. Originals are immutable; every pixel edit is stored
as a versioned derivative with the operation recorded.

## 2. Model services (already deployed on NVIDIA DGX Spark)

| Service | Role | Notes |
|---|---|---|
| SAM3 | Open-vocabulary segmentation: masks, boxes, labels per concept prompt | Concept-promptable; it does NOT spontaneously enumerate a scene |
| Qwen3.6-27B-FP8 | Captioning, topic proposal, narrative writing | VERIFY it is the vision (VL) variant — see Open Questions |
| qwen-image-edit | Outpainting and inpainting | Used by the crop/edit engine only |
| Face embedder (TO ADD) | 512-d face embeddings for identity clustering | InsightFace/ArcFace class model; small footprint |

### Memory budget (DGX Spark, 128 GB unified)
SAM3 (~3 GB) + Qwen 27B FP8 (~30 GB + KV cache) + qwen-image-edit (~40 GB bf16) can
co-reside but it is snug once batch KV cache and diffusion activations are added.
Design decision: the pipeline runs in phases rather than keeping everything hot —
analysis phase (SAM3 + Qwen + face embedder) and edit phase (qwen-image-edit) are
scheduled separately. Alternatively quantize the edit model. Do not architect anything
that requires all three at peak load simultaneously.

## 3. Architecture overview

```
photo(s) in  →  INGEST QUEUE  →  per-photo DAG:
                                   ├─ SAM3 catalog
                                   ├─ face embed + cluster
                                   └─ caption (consumes catalog + metadata)
                                          ↓
                              DERIVED DATA LAYER (database)
                                          ↓
             ┌────────────────────┬──────────────────────┐
        crop/edit engine    event enrichment       episode grouping
             └────────────────────┴──────────────────────┘
                                          ↓
                                 photobook builder
```

Core principle: every model output lands in the derived data layer; every downstream
feature reads from that layer instead of re-invoking models. Batch vs. single-photo is
the same per-photo DAG — batch just fans out over the queue. A single fresh upload runs
the same code path with a queue depth of one.

## 4. Stage 1 — SAM3 cataloging

SAM3 needs concept prompts. Strategy is two-pass:

1. **Concept proposal**: ask the VLM to list every distinct object, person, animal, and
   scene element visible in the photo.
2. **Grounding**: feed each proposed concept to SAM3 to get masks, bounding boxes, and
   confidence scores.

Supplement with a fixed vocabulary sweep for cross-library consistency (people, pets,
vehicles, food, buildings, water, sky, text/signage, common landmarks). The fixed list
guarantees "find all photos with dogs" behaves the same across the whole library even as
the VLM's proposals vary.

Store per detected object: label, RLE-encoded mask, bbox, area fraction, confidence,
and normalized position. All of these are consumed by the crop engine (Stage 4).

## 5. Stage 2 — Captioning

Never caption the raw image alone. Prompt context includes:
- SAM3 object list (labels + rough positions)
- EXIF date/time
- Reverse-geocoded location name
- Known people present (from face clustering, once identities are confirmed)

Target output: one rich sentence plus an optional short paragraph. Example quality bar:
"Emma and Jake at Snoqualmie Falls, October 2024" beats "two people near a waterfall."

Re-caption photos when face identities change (person renamed, clusters merged), since
captions embed names. Track a caption_version / dirty flag for this.

## 6. Stage 3 — Faces

SAM3 locates faces but does not identify them. Pipeline:

detect → align → embed (512-d, ArcFace-class model) → cluster.

- Clustering: HDBSCAN on cosine distance. Handles unknown cluster count and noise
  (strangers in backgrounds) gracefully.
- Embeddings stored in pgvector (if Postgres) → also powers "all photos of person X".
- **Human-in-the-loop (active learning)**: surface the most ambiguous decisions first —
  cluster pairs whose centroids sit just past the merge threshold, or single photos
  between clusters. Each user Yes/No becomes a must-link / cannot-link constraint;
  re-cluster with constraints applied. Persist constraints so re-clustering never
  un-does confirmed decisions.
- Persons are first-class entities the user can name. person ↔ face-cluster mapping
  survives re-clustering via the constraint set.

## 7. Stage 4 — Crop / edit engine

Candidate generation + scoring. No ML required for the core; it consumes Stage 1–3 data.

**Candidates**: sweep aspect ratios (1:1, 4:5, 3:2, 16:9, original) × positions × scales.

**Score components** (weighted sum; tune weights empirically):
- Subject placement on rule-of-thirds and phi-grid intersections. Subject = most
  salient SAM3 mask; faces weighted heaviest.
- Face headroom and lead room: leave space in the direction a face is looking.
- Horizon on a third-line; never mid-frame unless symmetric composition detected.
- Amputation penalty: crop edge must not intersect face masks or cut limbs/objects at
  awkward boundaries (mask-boundary intersection test).
- Subject area fraction within a pleasing range (not too small, not wall-to-wall).

**Rotation**: detect horizon / dominant verticals (Hough transform, or derive horizon
from SAM3 sky/water mask boundary) and propose leveling.

**Outpaint**: if the top-scoring crop extends past the frame, generate the outpaint mask
(the region outside the original) and submit to qwen-image-edit.

**Object removal**: user picks a detected object in the UI; its SAM3 mask already
exists. Dilate slightly, send mask + image to qwen-image-edit for inpainting.

All edits are versions; original files are never modified.

## 8. Stage 5 — Event enrichment

Operates per episode (Section 10), not per photo. For each (date, place) cluster:
- Reverse geocoding: self-hosted Nominatim (keeps GPS data local).
- Historical weather: Open-Meteo historical API.
- Holidays/observances for the locale and date.
- Notable events: Wikipedia current-events portal and/or a web search API for
  date + place.

Cache by (date, geohash) — 200 vacation photos share one context. Results go to an
`events` table linked to episodes.

## 9. Data model (additions to existing DB)

Assumes Postgres + pgvector; adapt if the existing DB differs (see Open Questions).

- `objects` — photo_id, label, mask_rle, bbox, area_frac, confidence, source
  (vlm_proposed | vocab_sweep)
- `faces` — photo_id, bbox, landmarks, embedding vector(512), quality score
- `persons` — id, display_name, cover_face_id
- `person_faces` — person_id, face_id, link_type (auto | confirmed), plus
  must-link / cannot-link constraint records
- `captions` — photo_id, text, version, context_hash (inputs used), model
- `crop_suggestions` — photo_id, aspect, rect, rotation_deg, score, score_breakdown,
  needs_outpaint bool, status (proposed | accepted | rejected)
- `edits` — photo_id, parent_version, operation (crop | outpaint | inpaint | rotate),
  params, output_uri
- `episodes` — time range, centroid GPS, geohash, place name, photo count
- `episode_photos` — episode_id, photo_id
- `events` — episode_id, kind (weather | holiday | news | local), summary, source_url
- `pages` — book_id, episode_id, topic_title, narrative_text, layout, status
- `page_photos` — page_id, photo_id, rank, reason
- Optional: `image_embeddings` — photo_id, CLIP-class vector for diversity selection
  and semantic search

## 10. Stages 6–7 — Episodes, topics, photobook

**Episode segmentation**: sort photos by timestamp; split on adaptive time gaps
(hours-scale within a day) or location jumps beyond a distance threshold. Episodes are
the page candidates.

**Topic proposal**: for each episode, Qwen proposes a title + one-line pitch from
captions, people present, place, and enrichment events. Show these to the user as
selectable page topics.

**On topic selection, pre-select photos** by scoring:
- Technical quality (sharpness, exposure)
- Face coverage: everyone present in the episode appears at least once
- Visual diversity: maximize spread in image-embedding space (avoid near-duplicates)

**Narrative**: Qwen writes page text from the selected photos' captions + event context.
Page count follows episode count; user can merge/split episodes and swap photos. All of
this is assembly over the derived data layer — no new model inference except the two
Qwen text calls per page.

## 11. Orchestration

- Job queue with per-stage workers. Celery is fine; **Temporal is recommended** because
  Stages 3 and 6 are literally long-running workflows that pause on human signals
  (face Yes/No, topic pick) — Temporal models that natively and durably.
- Per-photo DAG: catalog → {caption, faces} with caption depending on catalog output.
- Idempotency: every stage writes keyed on (photo_id, stage, model_version) so re-runs
  and model upgrades are clean re-derivations, not duplicates.
- GPU scheduling: analysis phase and edit phase are separate queues; do not co-schedule
  peak loads of Qwen batch captioning and diffusion edits (Section 2 memory budget).

## 12. Build order

1. Derived data layer schema + ingest queue + per-photo DAG skeleton
2. Stage 1 (SAM3 catalog) end to end, batch over existing library
3. Stage 3 (face embed + cluster) + the Yes/No review UI
4. Stage 2 (captions, consuming 1 + 3)
5. Stage 4 (crop scoring; then outpaint/inpaint wiring)
6. Stage 5 (enrichment) and Stages 6–7 (episodes → topics → pages)

Rationale: crops and stories are only as good as the derived data beneath them, and the
face-review UI surfaces identity errors that captions would otherwise bake in.

## 13. Open questions (ask the user before assuming)

1. Is the deployed Qwen3.6-27B-FP8 the **vision (VL)** variant? If text-only, captions
   must be synthesized from catalog + metadata and will be flatter; a VL model is
   strongly preferred for Stages 2 and 6.
2. What is the existing database engine and schema? (Spec assumes Postgres + pgvector.)
3. How are the three model services exposed — HTTP endpoints? What API shape
   (OpenAI-compatible for Qwen? custom for SAM3/edit)?
4. Existing app stack for the review UIs (face Yes/No, crop accept, topic picker)?
5. Photo storage layout (local FS paths? object store?) for reading originals and
   writing edit versions.
6. Is internet access acceptable for enrichment, or must lookups be proxied/cached in
   a specific way?

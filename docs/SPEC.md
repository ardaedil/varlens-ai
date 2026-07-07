# VARLens AI MVP Specification

## Overview

VARLens AI is a web application for educational analysis of short soccer clips. Version 1 accepts a single uploaded clip, estimates foul sanction severity and foul-action type, and returns an uncertainty-aware explanation of what a referee-style review would inspect.

The product must not claim official referee authority. Every result must include uncertainty, limitations, model provenance, and `official_decision_claimed: false`.

## Goals

- Ingest one short uploaded clip per request.
- Validate file type, size, duration, and requested analysis scope.
- Return top sanction and action-type predictions with alternatives.
- Generate deterministic explanations from structured model output.
- Reject unsupported v1 scopes with typed errors.
- Process uploads transiently and delete source clips by default.

## Non-Goals

- Definitive officiating.
- Live broadcast analysis.
- Offside adjudication.
- Standalone handball determination.
- Penalty/no-penalty judgment without a geometry module.
- Public redistribution of third-party copyrighted clips.

## Personas

- End user: uploads a short controversial soccer clip and receives likely foul/sanction interpretation, visual focus points, and uncertainty.
- Operator/admin: monitors health, model version, errors, and safety without seeing or retaining raw uploads by default.

## Functional Requirements

- The backend exposes `POST /api/v1/analyze`, `GET /api/v1/health`, and `GET /api/v1/model-info`.
- The frontend exposes an analysis workbench with upload, clip preview, scope selection, duration entry, loading/error states, and results.
- Transient uploads are written to `VARLENS_UPLOAD_TMP_DIR`, defaulting to `tmp/uploads`.
- The response contract is `packages/contracts/analyze.schema.json`.
- The canonical label taxonomy is `config/labels.json`.
- Explanation copy is deterministic-first and sourced from structured labels, confidence, and limitations.

## V1 Taxonomy

Sanction labels:

- `no_offence`
- `offence_no_card`
- `offence_yellow`
- `offence_red`

Action type labels:

- `standing_tackle`
- `tackle`
- `holding`
- `pushing`
- `challenge`
- `dive`
- `high_leg`
- `elbowing`
- `unknown`

Supported scope:

- `foul_review_context`

Unsupported scopes:

- `offside`
- `handball`
- `penalty_no_penalty`
- `mistaken_identity`

## API Rules

`POST /api/v1/analyze` accepts multipart form data:

- `file`: one video upload.
- `scope`: defaults to `foul_review_context`.
- `clip_duration_seconds`: optional client-provided duration estimate.

Default limits:

- Maximum size: 50 MB.
- Maximum duration: 15 seconds.
- Allowed media types: MP4, WebM, QuickTime, Matroska.

Errors must use the shared error envelope:

| Status | Code | Meaning |
| --- | --- | --- |
| 400 | `invalid_request` | Missing or invalid request parts |
| 413 | `clip_too_large` | File exceeds max size |
| 415 | `unsupported_media_type` | Format not supported |
| 422 | `clip_too_long` | Clip exceeds v1 duration |
| 422 | `unsupported_scope` | Requested scope is not available in v1 |
| 429 | `rate_limited` | Too many requests |
| 500 | `inference_failed` | Preprocessing or inference failure |
| 503 | `model_unavailable` | Model/backend degraded |

## Model Path

The current implementation uses a deterministic stub adapter in `services/api/app/inference/stub_model.py`. It is contract-compliant and exists to unblock API, UI, and test development.

The intended production path is a VideoMAE classifier trained on SoccerNet-MVFoul:

- Use MVFoul as primary supervised data.
- Treat each available view as a training sample while preserving action grouping for evaluation.
- Report macro F1, balanced accuracy, confusion matrices, and calibration summaries.
- Export model version and label mapping with every model artifact.

## Privacy Requirements

- Delete uploaded clips after analysis unless `VARLENS_DEBUG_RETAIN_UPLOADS=true`.
- Do not log raw frames, public object URLs, or source video bytes.
- Retain structured prediction metadata and redacted operational logs only.
- Do not ship public sample clips unless redistribution rights are confirmed.

## Acceptance Criteria

- Invalid file types and over-limit clips return typed errors.
- Valid requests return schema-compliant JSON.
- Successful responses include confidence, alternatives, limitations, and provenance.
- No response claims an official decision.
- Unsupported offside, handball, and penalty scopes return `unsupported_scope`.
- Uploads are removed after processing in default mode.
- Health/model-info expose model version and supported labels.

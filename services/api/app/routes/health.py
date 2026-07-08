from __future__ import annotations

from fastapi import APIRouter

from services.api.app.inference.backend import get_inference_backend
from services.api.app.inference.catalog import (
    ACTION_LABELS,
    DATASET_FAMILY,
    SANCTION_LABELS,
    SUPPORTED_SCOPES,
    UNSUPPORTED_SCOPES,
)

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, object]:
    backend = get_inference_backend()
    return {
        "service": "varlens-api",
        "status": "ok" if backend.info.available else "degraded",
        "model_available": backend.info.available,
        "model_version": backend.info.model_version,
        "inference_backend": backend.info.backend_name,
        "uses_stub": backend.info.uses_stub,
        "details": backend.info.details,
    }


@router.get("/model-info")
async def model_info() -> dict[str, object]:
    backend = get_inference_backend()
    return {
        "spec_version": "1.0.0",
        "model_version": backend.info.model_version,
        "dataset_family": DATASET_FAMILY,
        "supported_scopes": list(SUPPORTED_SCOPES),
        "unsupported_scopes": list(UNSUPPORTED_SCOPES),
        "sanction_labels": SANCTION_LABELS,
        "action_type_labels": ACTION_LABELS,
        "inference_backend": backend.info.backend_name,
        "uses_stub": backend.info.uses_stub,
        "details": backend.info.details,
        "limitations": [
            "Single uploaded clip only.",
            "No official decision claims.",
            "No offside, handball, or penalty/no-penalty adjudication in v1.",
        ],
    }

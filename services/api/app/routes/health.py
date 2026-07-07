from __future__ import annotations

from fastapi import APIRouter

from services.api.app.core.settings import settings
from services.api.app.inference.stub_model import ACTION_LABELS, SANCTION_LABELS

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, object]:
    return {
        "service": "varlens-api",
        "status": "ok" if settings.model_available else "degraded",
        "model_available": settings.model_available,
        "model_version": settings.model_version,
    }


@router.get("/model-info")
async def model_info() -> dict[str, object]:
    return {
        "spec_version": settings.spec_version,
        "model_version": settings.model_version,
        "dataset_family": "SoccerNet-MVFoul",
        "supported_scopes": ["foul_review_context"],
        "unsupported_scopes": ["offside", "handball", "penalty_no_penalty", "mistaken_identity"],
        "sanction_labels": SANCTION_LABELS,
        "action_type_labels": ACTION_LABELS,
        "limitations": [
            "Single uploaded clip only.",
            "No official decision claims.",
            "No offside, handball, or penalty/no-penalty adjudication in v1.",
        ],
    }

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SPEC_VERSION = "1.0.0"
MODEL_VERSION = "videomae-mvfoul-v1-stub"

SanctionLabel = Literal[
    "no_offence",
    "offence_no_card",
    "offence_yellow",
    "offence_red",
]

ActionTypeLabel = Literal[
    "standing_tackle",
    "tackle",
    "holding",
    "pushing",
    "challenge",
    "dive",
    "high_leg",
    "elbowing",
    "unknown",
]

SupportedScope = Literal["foul_review_context"]
UnsupportedScope = Literal["offside", "handball", "penalty_no_penalty", "mistaken_identity"]
AnalysisScope = SupportedScope | UnsupportedScope

VarReviewCategory = Literal["not_applicable", "direct_red_related", "unsupported_in_v1"]
ErrorCode = Literal[
    "invalid_request",
    "clip_too_large",
    "unsupported_media_type",
    "clip_too_long",
    "unsupported_scope",
    "rate_limited",
    "inference_failed",
    "model_unavailable",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictionAlternative(StrictModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class SanctionPrediction(StrictModel):
    label: SanctionLabel
    confidence: float = Field(ge=0, le=1)
    alternatives: list[PredictionAlternative] = Field(default_factory=list, max_length=3)


class ActionTypePrediction(StrictModel):
    label: ActionTypeLabel
    confidence: float = Field(ge=0, le=1)
    alternatives: list[PredictionAlternative] = Field(default_factory=list, max_length=3)


class ReviewContext(StrictModel):
    var_review_category: VarReviewCategory
    official_decision_claimed: Literal[False] = False


class Provenance(StrictModel):
    dataset_family: str
    rules_source: str
    frames_sampled: int = Field(ge=0)


class AnalyzeResponse(StrictModel):
    spec_version: Literal["1.0.0"] = SPEC_VERSION
    model_version: str
    request_id: str
    status: Literal["ok"] = "ok"
    sanction_prediction: SanctionPrediction
    action_type_prediction: ActionTypePrediction
    viewer_focus: list[str] = Field(min_length=2, max_length=6)
    explanation: str
    limitations: list[str] = Field(min_length=1, max_length=6)
    review_context: ReviewContext
    provenance: Provenance


class ErrorBody(StrictModel):
    code: ErrorCode
    message: str
    details: dict[str, str | int | float | bool | None] | None = None


class ErrorResponse(StrictModel):
    spec_version: Literal["1.0.0"] = SPEC_VERSION
    request_id: str
    status: Literal["error"] = "error"
    error: ErrorBody

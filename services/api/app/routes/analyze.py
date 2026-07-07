from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from packages.contracts.analyze import AnalyzeResponse, Provenance, ReviewContext
from services.api.app.core.settings import settings
from services.api.app.explanations.templates import build_explanation
from services.api.app.inference.stub_model import analyze_clip
from services.api.app.security.errors import ApiError, error_response
from services.api.app.security.uploads import (
    delete_transient_upload,
    validate_duration,
    validate_media_type,
    validate_scope,
    write_transient_upload,
)

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile | None = File(default=None),
    scope: str = Form(default="foul_review_context"),
    clip_duration_seconds: float | None = Form(default=None),
) -> AnalyzeResponse | JSONResponse:
    request_id = str(uuid4())

    try:
        validate_scope(scope)
        validate_duration(clip_duration_seconds, settings)

        if file is None:
            raise ApiError(code="invalid_request", message="A video file is required.")

        if not settings.model_available:
            raise ApiError(
                code="model_unavailable",
                message="The analysis model is not available.",
            )

        validate_media_type(file, settings)
        temp_path, size_bytes = await write_transient_upload(file, settings)

        try:
            model_version, sanction_prediction, action_prediction = analyze_clip(
                path=temp_path,
                filename=file.filename or "",
                content_type=file.content_type or "",
                size_bytes=size_bytes,
            )
            explanation, viewer_focus, limitations = build_explanation(
                sanction_prediction,
                action_prediction,
            )

            var_category = (
                "direct_red_related"
                if sanction_prediction.label == "offence_red"
                else "not_applicable"
            )

            response = AnalyzeResponse(
                model_version=model_version,
                request_id=request_id,
                sanction_prediction=sanction_prediction,
                action_type_prediction=action_prediction,
                viewer_focus=viewer_focus,
                explanation=explanation,
                limitations=limitations,
                review_context=ReviewContext(
                    var_review_category=var_category,
                    official_decision_claimed=False,
                ),
                provenance=Provenance(
                    dataset_family="SoccerNet-MVFoul",
                    rules_source="IFAB Laws 2026/27",
                    frames_sampled=settings.frames_sampled,
                ),
            )
        finally:
            upload_deleted = delete_transient_upload(temp_path, settings)

        if not upload_deleted:
            response.limitations.append("Source upload cleanup failed; operator cleanup is required.")

        return response
    except ApiError as exc:
        return error_response(
            code=exc.code,
            message=exc.message,
            request_id=request_id,
            details=exc.details,
        )
    except Exception:
        return error_response(
            code="inference_failed",
            message="Analysis failed while preprocessing or running inference.",
            request_id=request_id,
        )

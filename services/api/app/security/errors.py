from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from fastapi.responses import JSONResponse

from packages.contracts.analyze import ErrorCode, ErrorResponse


ERROR_STATUS: dict[str, int] = {
    "invalid_request": 400,
    "clip_too_large": 413,
    "unsupported_media_type": 415,
    "clip_too_long": 422,
    "unsupported_scope": 422,
    "rate_limited": 429,
    "inference_failed": 500,
    "model_unavailable": 503,
}


@dataclass
class ApiError(Exception):
    code: ErrorCode
    message: str
    details: dict[str, str | int | float | bool | None] | None = None

    @property
    def status_code(self) -> int:
        return ERROR_STATUS[self.code]


def error_payload(
    *,
    code: ErrorCode,
    message: str,
    request_id: str | None = None,
    details: dict[str, str | int | float | bool | None] | None = None,
) -> ErrorResponse:
    return ErrorResponse(
        request_id=request_id or str(uuid4()),
        error={
            "code": code,
            "message": message,
            "details": details,
        },
    )


def error_response(
    *,
    code: ErrorCode,
    message: str,
    request_id: str | None = None,
    details: dict[str, str | int | float | bool | None] | None = None,
) -> JSONResponse:
    payload = error_payload(
        code=code,
        message=message,
        request_id=request_id,
        details=details,
    )
    return JSONResponse(
        status_code=ERROR_STATUS[code],
        content=payload.model_dump(exclude_none=True),
    )

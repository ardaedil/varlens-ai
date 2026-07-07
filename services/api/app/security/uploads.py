from __future__ import annotations

import secrets
import logging
from pathlib import Path

from fastapi import UploadFile

from services.api.app.core.settings import Settings
from services.api.app.security.errors import ApiError

logger = logging.getLogger("varlens.api.uploads")

SUPPORTED_EXTENSIONS = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
}


def validate_scope(scope: str) -> None:
    if scope != "foul_review_context":
        raise ApiError(
            code="unsupported_scope",
            message="VARLens v1 only supports foul and sanction explanation.",
            details={"requested_scope": scope},
        )


def validate_duration(duration_seconds: float | None, settings: Settings) -> None:
    if duration_seconds is None:
        return
    if duration_seconds <= 0:
        raise ApiError(
            code="invalid_request",
            message="Clip duration must be greater than zero.",
            details={"duration_seconds": duration_seconds},
        )
    if duration_seconds > settings.max_duration_seconds:
        raise ApiError(
            code="clip_too_long",
            message=f"Clip duration exceeds the {settings.max_duration_seconds}-second v1 limit.",
            details={
                "duration_seconds": duration_seconds,
                "max_duration_seconds": settings.max_duration_seconds,
            },
        )


def validate_media_type(upload: UploadFile, settings: Settings) -> None:
    content_type = upload.content_type or ""
    suffix = Path(upload.filename or "").suffix.lower()
    if content_type not in settings.allowed_media_types:
        raise ApiError(
            code="unsupported_media_type",
            message="Unsupported video format.",
            details={
                "content_type": content_type or "unknown",
                "allowed_media_types": ", ".join(settings.allowed_media_types),
            },
        )
    if suffix and suffix in SUPPORTED_EXTENSIONS and SUPPORTED_EXTENSIONS[suffix] != content_type:
        raise ApiError(
            code="unsupported_media_type",
            message="File extension does not match the declared media type.",
            details={"filename": upload.filename or "", "content_type": content_type},
        )


async def write_transient_upload(upload: UploadFile, settings: Settings) -> tuple[Path, int]:
    suffix = Path(upload.filename or "").suffix.lower() or ".bin"
    upload_dir = Path(settings.upload_tmp_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"varlens-{secrets.token_hex(12)}{suffix}"
    total = 0
    try:
        with path.open("wb") as handle:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    raise ApiError(
                        code="clip_too_large",
                        message="Clip exceeds the configured v1 upload size limit.",
                        details={
                            "max_upload_bytes": settings.max_upload_bytes,
                            "bytes_read": total,
                        },
                    )
                handle.write(chunk)
    except Exception:
        _try_unlink(path)
        raise

    if total == 0:
        _try_unlink(path)
        raise ApiError(code="invalid_request", message="Uploaded clip is empty.")

    return path, total


def _try_unlink(path: Path) -> bool:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.exception("failed to delete transient upload path=%s", path)
        return False
    return True


def delete_transient_upload(path: Path, settings: Settings) -> bool:
    if settings.debug_retain_uploads:
        return True
    return _try_unlink(path)

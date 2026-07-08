from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    spec_version: str = "1.0.0"
    model_version: str = "videomae-mvfoul-v1-stub"
    max_upload_bytes: int = _env_int("VARLENS_MAX_UPLOAD_BYTES", 50 * 1024 * 1024)
    max_duration_seconds: int = _env_int("VARLENS_MAX_DURATION_SECONDS", 15)
    model_available: bool = _env_bool("VARLENS_MODEL_AVAILABLE", True)
    inference_backend: str = os.getenv("VARLENS_INFERENCE_BACKEND", "auto")
    sanction_model_dir: str | None = os.getenv("VARLENS_SANCTION_MODEL_DIR")
    action_model_dir: str | None = os.getenv("VARLENS_ACTION_MODEL_DIR")
    debug_retain_uploads: bool = _env_bool("VARLENS_DEBUG_RETAIN_UPLOADS", False)
    upload_tmp_dir: str = os.getenv("VARLENS_UPLOAD_TMP_DIR", "tmp/uploads")
    frames_sampled: int = _env_int("VARLENS_FRAMES_SAMPLED", 16)
    allowed_origins: tuple[str, ...] = (
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    )
    allowed_media_types: tuple[str, ...] = (
        "video/mp4",
        "video/webm",
        "video/quicktime",
        "video/x-matroska",
    )


settings = Settings()

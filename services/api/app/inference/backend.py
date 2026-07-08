from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from packages.contracts.analyze import ActionTypePrediction, SanctionPrediction
from services.api.app.core.settings import settings

DEFAULT_STUB_MODEL_VERSION = "videomae-mvfoul-v1-stub"


@dataclass(frozen=True)
class BackendInfo:
    backend_name: str
    model_version: str
    available: bool
    uses_stub: bool
    details: dict[str, str] = field(default_factory=dict)


class InferenceBackend(Protocol):
    info: BackendInfo

    def analyze_clip(
        self,
        *,
        path: Path,
        filename: str,
        content_type: str,
        size_bytes: int,
    ) -> tuple[str, SanctionPrediction, ActionTypePrediction]:
        ...


class DisabledInferenceBackend:
    def __init__(self, *, reason: str) -> None:
        self.info = BackendInfo(
            backend_name="disabled",
            model_version=settings.model_version,
            available=False,
            uses_stub=False,
            details={"reason": reason},
        )

    def analyze_clip(
        self,
        *,
        path: Path,
        filename: str,
        content_type: str,
        size_bytes: int,
    ) -> tuple[str, SanctionPrediction, ActionTypePrediction]:
        raise RuntimeError(self.info.details["reason"])


class StubInferenceBackend:
    def __init__(self, *, fallback_reason: str | None = None) -> None:
        from services.api.app.inference.stub_model import analyze_clip

        details: dict[str, str] = {}
        if fallback_reason:
            details["fallback_reason"] = fallback_reason
        self.info = BackendInfo(
            backend_name="stub",
            model_version=settings.model_version or DEFAULT_STUB_MODEL_VERSION,
            available=True,
            uses_stub=True,
            details=details,
        )
        self._analyze_clip = analyze_clip

    def analyze_clip(
        self,
        *,
        path: Path,
        filename: str,
        content_type: str,
        size_bytes: int,
    ) -> tuple[str, SanctionPrediction, ActionTypePrediction]:
        return self._analyze_clip(
            path=path,
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
        )


_backend_instance: InferenceBackend | None = None


def get_inference_backend() -> InferenceBackend:
    global _backend_instance
    if _backend_instance is None:
        _backend_instance = _build_inference_backend()
    return _backend_instance


def reset_inference_backend() -> None:
    global _backend_instance
    _backend_instance = None


def _build_inference_backend() -> InferenceBackend:
    if not settings.model_available:
        return DisabledInferenceBackend(reason="The analysis model is disabled by configuration.")

    backend_name = settings.inference_backend.strip().lower()
    if backend_name == "stub":
        return StubInferenceBackend()

    if backend_name not in {"auto", "videomae"}:
        return DisabledInferenceBackend(
            reason=f"Unsupported inference backend '{settings.inference_backend}'.",
        )

    try:
        from services.api.app.inference.videomae_model import (
            VideoMAEBackendLoadError,
            VideoMAEInferenceBackend,
        )

        return VideoMAEInferenceBackend.from_settings(settings)
    except VideoMAEBackendLoadError as exc:
        if backend_name == "videomae":
            return DisabledInferenceBackend(reason=str(exc))
        return StubInferenceBackend(fallback_reason=str(exc))

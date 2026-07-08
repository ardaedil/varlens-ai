from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from packages.contracts.analyze import (
    ActionTypePrediction,
    PredictionAlternative,
    SanctionPrediction,
)
from services.api.app.inference.backend import BackendInfo
from services.api.app.inference.catalog import ACTION_LABELS, SANCTION_LABELS


class VideoMAEBackendLoadError(RuntimeError):
    """Raised when the VideoMAE backend cannot be initialized safely."""


class VideoMAEInferenceBackend:
    def __init__(
        self,
        *,
        sanction_pipeline: Any,
        action_pipeline: Any,
        info: BackendInfo,
    ) -> None:
        self._sanction_pipeline = sanction_pipeline
        self._action_pipeline = action_pipeline
        self.info = info

    @classmethod
    def from_settings(cls, app_settings: Any) -> "VideoMAEInferenceBackend":
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise VideoMAEBackendLoadError(
                "Transformers video inference dependencies are missing. "
                "Install services/api/requirements-ml.txt to enable the real backend."
            ) from exc

        sanction_dir = _resolve_model_dir(app_settings.sanction_model_dir, task_name="sanction")
        action_dir = _resolve_model_dir(app_settings.action_model_dir, task_name="action")
        sanction_summary = _load_training_summary(sanction_dir, expected_label_field="sanction_label")
        action_summary = _load_training_summary(action_dir, expected_label_field="action_type_label")

        try:
            sanction_pipeline = pipeline(
                task="video-classification",
                model=str(sanction_dir),
                device=-1,
            )
            action_pipeline = pipeline(
                task="video-classification",
                model=str(action_dir),
                device=-1,
            )
        except Exception as exc:  # pragma: no cover - depends on optional ML stack and artifacts
            raise VideoMAEBackendLoadError(
                f"Could not load VideoMAE artifacts from '{sanction_dir}' and '{action_dir}': {exc}"
            ) from exc

        sanction_version = str(sanction_summary.get("model_version", sanction_dir.name))
        action_version = str(action_summary.get("model_version", action_dir.name))
        bundle_version = _bundle_version(
            configured=app_settings.model_version,
            sanction_version=sanction_version,
            action_version=action_version,
        )

        info = BackendInfo(
            backend_name="videomae",
            model_version=bundle_version,
            available=True,
            uses_stub=False,
            details={
                "sanction_model_version": sanction_version,
                "action_model_version": action_version,
                "sanction_model_dir": str(sanction_dir),
                "action_model_dir": str(action_dir),
            },
        )
        return cls(
            sanction_pipeline=sanction_pipeline,
            action_pipeline=action_pipeline,
            info=info,
        )

    def analyze_clip(
        self,
        *,
        path: Path,
        filename: str,
        content_type: str,
        size_bytes: int,
    ) -> tuple[str, SanctionPrediction, ActionTypePrediction]:
        del filename, content_type, size_bytes
        sanction_prediction = _coerce_prediction(
            self._sanction_pipeline(str(path), top_k=len(SANCTION_LABELS)),
            expected_labels=SANCTION_LABELS,
            prediction_kind="sanction",
        )
        action_prediction = _coerce_prediction(
            self._action_pipeline(str(path), top_k=len(ACTION_LABELS)),
            expected_labels=ACTION_LABELS,
            prediction_kind="action",
        )
        return self.info.model_version, sanction_prediction, action_prediction


def _resolve_model_dir(raw_path: str | None, *, task_name: str) -> Path:
    if not raw_path:
        raise VideoMAEBackendLoadError(
            f"VARLENS_{task_name.upper()}_MODEL_DIR is required for the VideoMAE backend."
        )
    model_dir = Path(raw_path).expanduser().resolve()
    if not model_dir.exists() or not model_dir.is_dir():
        raise VideoMAEBackendLoadError(
            f"Configured {task_name} model directory does not exist: {model_dir}"
        )
    return model_dir


def _load_training_summary(model_dir: Path, *, expected_label_field: str) -> dict[str, Any]:
    summary_path = model_dir / "training_summary.json"
    if not summary_path.exists():
        raise VideoMAEBackendLoadError(
            f"Missing training_summary.json in model directory '{model_dir}'."
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    label_field = summary.get("label_field")
    if label_field != expected_label_field:
        raise VideoMAEBackendLoadError(
            f"Model directory '{model_dir}' has label field '{label_field}', expected "
            f"'{expected_label_field}'."
        )
    return summary


def _bundle_version(*, configured: str, sanction_version: str, action_version: str) -> str:
    if configured and configured != "videomae-mvfoul-v1-stub":
        return configured
    if sanction_version == action_version:
        return sanction_version
    return f"bundle:{sanction_version}+{action_version}"


def _coerce_prediction(
    predictions: Any,
    *,
    expected_labels: list[str],
    prediction_kind: str,
) -> SanctionPrediction | ActionTypePrediction:
    rows = predictions[0] if predictions and isinstance(predictions[0], list) else predictions
    ranked: list[tuple[str, float]] = []
    seen_labels: set[str] = set()
    for row in rows:
        label = str(row["label"])
        if label not in expected_labels or label in seen_labels:
            continue
        seen_labels.add(label)
        ranked.append((label, round(float(row["score"]), 6)))

    if not ranked:
        raise RuntimeError(f"The {prediction_kind} model did not return any valid labels.")

    top_label, top_score = ranked[0]
    alternatives = [
        PredictionAlternative(label=label, confidence=score)
        for label, score in ranked[1:4]
    ]
    if prediction_kind == "sanction":
        return SanctionPrediction(
            label=top_label,  # type: ignore[arg-type]
            confidence=top_score,
            alternatives=alternatives,
        )
    return ActionTypePrediction(
        label=top_label,  # type: ignore[arg-type]
        confidence=top_score,
        alternatives=alternatives,
    )

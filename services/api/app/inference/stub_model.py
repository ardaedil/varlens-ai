from __future__ import annotations

import hashlib
from pathlib import Path

from packages.contracts.analyze import (
    ActionTypePrediction,
    MODEL_VERSION,
    SanctionPrediction,
)

SANCTION_LABELS = ["no_offence", "offence_no_card", "offence_yellow", "offence_red"]
ACTION_LABELS = [
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


def _digest_file(path: Path, *, filename: str, content_type: str, size_bytes: int) -> bytes:
    digest = hashlib.sha256()
    digest.update(filename.encode("utf-8", errors="ignore"))
    digest.update(content_type.encode("utf-8", errors="ignore"))
    digest.update(str(size_bytes).encode("ascii"))
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.digest()


def _prediction(labels: list[str], digest: bytes, offset: int) -> tuple[str, float, list[dict[str, float | str]]]:
    raw_scores = []
    for index, label in enumerate(labels):
        value = digest[(index + offset) % len(digest)] / 255
        raw_scores.append((label, 0.15 + value))
    total = sum(score for _, score in raw_scores)
    normalized = [(label, score / total) for label, score in raw_scores]
    ranked = sorted(normalized, key=lambda item: item[1], reverse=True)
    label, confidence = ranked[0]
    alternatives = [
        {"label": alt_label, "confidence": round(alt_confidence, 3)}
        for alt_label, alt_confidence in ranked[1:4]
        if alt_confidence >= 0.05
    ]
    # Keep the stub useful for UI work by making top confidence visibly decisive enough.
    confidence = min(0.91, max(0.52, confidence + 0.35))
    return label, round(confidence, 3), alternatives


def analyze_clip(
    *,
    path: Path,
    filename: str,
    content_type: str,
    size_bytes: int,
) -> tuple[str, SanctionPrediction, ActionTypePrediction]:
    digest = _digest_file(
        path,
        filename=filename,
        content_type=content_type,
        size_bytes=size_bytes,
    )
    sanction_label, sanction_confidence, sanction_alternatives = _prediction(
        SANCTION_LABELS,
        digest,
        offset=0,
    )
    action_label, action_confidence, action_alternatives = _prediction(
        ACTION_LABELS,
        digest,
        offset=7,
    )
    return (
        MODEL_VERSION,
        SanctionPrediction(
            label=sanction_label,
            confidence=sanction_confidence,
            alternatives=sanction_alternatives,
        ),
        ActionTypePrediction(
            label=action_label,
            confidence=action_confidence,
            alternatives=action_alternatives,
        ),
    )

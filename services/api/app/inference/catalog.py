from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
LABELS_PATH = ROOT / "config" / "labels.json"
DATASET_FAMILY = "SoccerNet-MVFoul"
SUPPORTED_SCOPES = ("foul_review_context",)


@lru_cache(maxsize=1)
def load_label_catalog() -> dict[str, Any]:
    return json.loads(LABELS_PATH.read_text(encoding="utf-8"))


CATALOG = load_label_catalog()
SANCTION_LABELS = [item["id"] for item in CATALOG["sanction_labels"]]
ACTION_LABELS = [item["id"] for item in CATALOG["action_type_labels"]]
UNSUPPORTED_SCOPES = tuple(CATALOG["unsupported_scopes"])

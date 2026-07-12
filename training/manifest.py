from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_LABELS_PATH = Path(__file__).resolve().parents[1] / "config" / "labels.json"
SUPPORTED_SPLITS = ("train", "valid", "test", "challenge")
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "valid",
    "valid": "valid",
    "validation": "valid",
    "dev": "valid",
    "test": "test",
    "testing": "test",
    "challenge": "challenge",
}
TOKEN_RE = re.compile(r"[^a-z0-9]+")


class ManifestError(ValueError):
    """Raised when MVFoul annotations cannot be normalized safely."""


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    split: str
    action_id: str
    view_id: str
    group_id: str
    video_path: str
    relative_video_path: str
    view_type: str
    sanction_label: str
    action_type_label: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def load_label_catalog(labels_path: Path | None = None) -> dict[str, Any]:
    path = labels_path or DEFAULT_LABELS_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_token(value: str) -> str:
    return TOKEN_RE.sub(" ", value.strip().lower()).strip()


def normalize_split(value: str | None, *, default: str | None = None) -> str:
    if value is None:
        if default is None:
            raise ManifestError("Split is required when it cannot be inferred from annotations.")
        return normalize_split(default)

    normalized = SPLIT_ALIASES.get(normalize_token(value))
    if normalized is None:
        raise ManifestError(
            f"Unsupported split '{value}'. Expected one of: {', '.join(SUPPORTED_SPLITS)}."
        )
    return normalized


def normalize_view_type(value: str | None) -> str:
    if not value:
        return "unknown"
    token = normalize_token(value)
    if "replay" in token:
        return "replay"
    if "live" in token:
        return "live"
    return "unknown"


def _first_present(mapping: dict[str, Any], *keys: str) -> Any | None:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _catalog_lookup(
    catalog: list[dict[str, str]],
    *,
    allow_unknown: bool = False,
) -> tuple[dict[str, str], list[str]]:
    lookup: dict[str, str] = {}
    ordered_ids: list[str] = []
    for item in catalog:
        label_id = item["id"]
        ordered_ids.append(label_id)
        for candidate in (item["id"], item.get("raw_label", ""), item.get("display", "")):
            if candidate:
                lookup[normalize_token(candidate)] = label_id
    if allow_unknown:
        lookup[""] = "unknown"
    return lookup, ordered_ids


def _normalize_label(
    value: str | None,
    *,
    lookup: dict[str, str],
    label_kind: str,
    allow_unknown: bool = False,
) -> str:
    if value in (None, ""):
        if allow_unknown:
            return "unknown"
        raise ManifestError(f"Missing {label_kind} label.")

    normalized = lookup.get(normalize_token(str(value)))
    if normalized is None:
        raise ManifestError(f"Unknown {label_kind} label '{value}'.")
    return normalized


def _extract_actions(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("actions", "annotations", "events", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        actions_map = payload.get("Actions")
        if isinstance(actions_map, dict):
            extracted: list[dict[str, Any]] = []
            for key, value in actions_map.items():
                if isinstance(value, dict):
                    enriched = dict(value)
                    enriched.setdefault("id", key)
                    extracted.append(enriched)
            return extracted
    raise ManifestError("Could not find an action list in the provided MVFoul annotations.")


def _guess_split(annotation_path: Path, *, default_split: str | None = None) -> str:
    for part in annotation_path.parts:
        token = normalize_token(part)
        if token in SPLIT_ALIASES:
            return SPLIT_ALIASES[token]
    if default_split is not None:
        return normalize_split(default_split)
    raise ManifestError(
        f"Could not infer a split from '{annotation_path}'. Pass --default-split explicitly."
    )


def _resolve_video_path(
    raw_video_path: str,
    *,
    video_root: Path | None,
) -> tuple[Path, str]:
    raw_path = Path(raw_video_path)
    if raw_path.is_absolute():
        resolved = raw_path
    elif video_root is not None:
        resolved = (video_root / raw_path).resolve()
    else:
        resolved = raw_path

    if video_root is not None:
        try:
            relative = resolved.relative_to(video_root.resolve()).as_posix()
        except ValueError:
            relative = raw_path.as_posix()
    else:
        relative = raw_path.as_posix()

    return resolved, relative


def _normalize_soccernet_clip_path(raw_path: str) -> str:
    normalized = Path(raw_path.replace("\\", "/")).as_posix().lstrip("/")
    parts = normalized.split("/")
    if len(parts) >= 3 and parts[0].lower() == "dataset":
        split_token = normalize_token(parts[1])
        split_name = SPLIT_ALIASES.get(split_token)
        if split_name is not None:
            return "/".join([split_name, *parts[2:]])
    return normalized


def _soccernet_sanction_label(action: dict[str, Any]) -> str:
    offence = normalize_token(str(action.get("Offence", "")))
    severity = normalize_token(str(action.get("Severity", "")))
    handball = normalize_token(str(action.get("Handball", "")))

    if handball == "handball":
        raise ManifestError("Handball samples are outside the current VARLens v1 foul taxonomy.")

    if offence in {"", "no offence", "between"}:
        return "no_offence"
    if offence != "offence":
        raise ManifestError(f"Unsupported offence label '{action.get('Offence')}'.")

    if severity in {"0", "0 0", "0.0"}:
        return "offence_no_card"
    if severity in {"1", "1 0", "1.0"}:
        return "offence_yellow"
    if severity in {"2", "2 0", "2.0", "3", "3 0", "3.0"}:
        return "offence_red"
    raise ManifestError(f"Unsupported severity value '{action.get('Severity')}'.")


def _soccernet_action_label(action: dict[str, Any]) -> str:
    raw_label = str(action.get("Action class", ""))
    token = normalize_token(raw_label)
    if token == "":
        return "unknown"

    aliases = {
        "standing tackle": "standing_tackle",
        "standing tackling": "standing_tackle",
        "tackle": "tackle",
        "tackling": "tackle",
        "holding": "holding",
        "pushing": "pushing",
        "challenge": "challenge",
        "dive": "dive",
        "high leg": "high_leg",
        "elbowing": "elbowing",
    }
    normalized = aliases.get(token)
    if normalized is None:
        raise ManifestError(f"Unsupported SoccerNet action class '{raw_label}'.")
    return normalized


def _soccernet_views(action: dict[str, Any]) -> list[dict[str, Any]]:
    raw_clips = action.get("Clips")
    if not isinstance(raw_clips, list) or not raw_clips:
        raise ManifestError(f"Action '{action.get('id', 'unknown')}' has no SoccerNet clips.")

    views: list[dict[str, Any]] = []
    for clip_index, clip in enumerate(raw_clips):
        if not isinstance(clip, dict):
            raise ManifestError("SoccerNet clip entries must be JSON objects.")
        clip_url = clip.get("Url")
        if not clip_url:
            raise ManifestError("SoccerNet clip entry is missing 'Url'.")
        clip_path = _normalize_soccernet_clip_path(str(clip_url))
        if not clip_path.lower().endswith(".mp4"):
            clip_path = f"{clip_path}.mp4"
        camera_type = str(clip.get("Camera type", ""))
        views.append(
            {
                "id": f"clip-{clip_index:02d}",
                "video_path": clip_path,
                "view_type": camera_type,
            }
        )
    return views


def _records_from_action(
    *,
    action: dict[str, Any],
    action_index: int,
    annotation_path: Path,
    video_root: Path | None,
    split_default: str,
    sanction_lookup: dict[str, str],
    action_lookup: dict[str, str],
    verify_files: bool,
) -> list[ManifestRecord]:
    if "Clips" in action and "Action class" in action:
        action = dict(action)
        action.setdefault("action_id", action.get("id"))
        action["sanction_label"] = _soccernet_sanction_label(action)
        action["action_type_label"] = _soccernet_action_label(action)
        action["views"] = _soccernet_views(action)

    action_id = str(
        _first_present(action, "id", "action_id", "uid") or f"{annotation_path.stem}-{action_index:05d}"
    )
    group_id = str(_first_present(action, "group_id", "scene_id", "sequence_id") or action_id)
    split = normalize_split(
        _first_present(action, "split", "subset", "partition"),
        default=split_default,
    )
    sanction_label = _normalize_label(
        _first_present(action, "sanction_label", "sanction", "card_label", "label"),
        lookup=sanction_lookup,
        label_kind="sanction",
    )
    action_type_label = _normalize_label(
        _first_present(
            action,
            "action_type_label",
            "action_type",
            "foul_action",
            "action_label",
            "action",
        ),
        lookup=action_lookup,
        label_kind="action type",
        allow_unknown=True,
    )

    raw_views = action.get("views") or action.get("clips") or []
    if not raw_views and _first_present(
        action,
        "video_path",
        "path",
        "file_path",
        "file_name",
        "clip_path",
    ):
        raw_views = [action]
    if not isinstance(raw_views, list) or not raw_views:
        raise ManifestError(f"Action '{action_id}' has no usable views.")

    records: list[ManifestRecord] = []
    for view_index, view in enumerate(raw_views):
        if not isinstance(view, dict):
            raise ManifestError(f"Action '{action_id}' contains a non-object view entry.")
        raw_video_path = _first_present(
            view,
            "video_path",
            "path",
            "file_path",
            "file_name",
            "video",
            "clip_path",
        )
        if raw_video_path is None:
            raise ManifestError(f"View {view_index} for action '{action_id}' is missing a video path.")
        resolved_path, relative_path = _resolve_video_path(
            str(raw_video_path),
            video_root=video_root,
        )
        if verify_files and not resolved_path.exists():
            raise ManifestError(
                f"Video file '{resolved_path}' for action '{action_id}' does not exist."
            )
        view_id = str(_first_present(view, "id", "view_id") or f"view-{view_index:02d}")
        view_type = normalize_view_type(
            _first_present(view, "view_type", "type", "role", "camera_type")
        )
        records.append(
            ManifestRecord(
                sample_id=f"{split}/{action_id}/{view_id}",
                split=split,
                action_id=action_id,
                view_id=view_id,
                group_id=group_id,
                video_path=str(resolved_path),
                relative_video_path=relative_path,
                view_type=view_type,
                sanction_label=sanction_label,
                action_type_label=action_type_label,
            )
        )
    return records


def summarize_records(records: list[ManifestRecord]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "record_count": len(records),
        "action_count": len({record.action_id for record in records}),
        "split_counts": {},
        "view_type_counts": dict(Counter(record.view_type for record in records)),
        "sanction_label_counts": dict(Counter(record.sanction_label for record in records)),
        "action_type_label_counts": dict(Counter(record.action_type_label for record in records)),
    }
    split_counts: dict[str, dict[str, int]] = {}
    for split in SUPPORTED_SPLITS:
        split_records = [record for record in records if record.split == split]
        if not split_records:
            continue
        split_counts[split] = {
            "records": len(split_records),
            "actions": len({record.action_id for record in split_records}),
        }
    summary["split_counts"] = split_counts
    return summary


def build_manifest_document(
    *,
    annotation_paths: list[Path],
    video_root: Path | None = None,
    default_split: str | None = None,
    verify_files: bool = False,
) -> dict[str, Any]:
    catalog = load_label_catalog()
    sanction_lookup, _ = _catalog_lookup(catalog["sanction_labels"])
    action_lookup, _ = _catalog_lookup(catalog["action_type_labels"], allow_unknown=True)

    records: list[ManifestRecord] = []
    for annotation_path in annotation_paths:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        actions = _extract_actions(payload)
        split_default = _guess_split(annotation_path, default_split=default_split)
        for action_index, action in enumerate(actions):
            records.extend(
                _records_from_action(
                    action=action,
                    action_index=action_index,
                    annotation_path=annotation_path,
                    video_root=video_root,
                    split_default=split_default,
                    sanction_lookup=sanction_lookup,
                    action_lookup=action_lookup,
                    verify_files=verify_files,
                )
            )

    return {
        "manifest_version": "1.0.0",
        "dataset_family": "SoccerNet-MVFoul",
        "task": "mvfoul",
        "annotation_sources": [str(path) for path in annotation_paths],
        "video_root": str(video_root) if video_root is not None else None,
        "summary": summarize_records(records),
        "records": [record.to_dict() for record in records],
    }


def write_manifest(document: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def load_manifest_records(manifest_path: Path) -> list[ManifestRecord]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) and "records" in payload else payload
    return [ManifestRecord(**record) for record in records]


def filter_records(records: list[ManifestRecord], splits: set[str]) -> list[ManifestRecord]:
    return [record for record in records if record.split in splits]


def build_label_maps(
    records: list[ManifestRecord],
    *,
    label_field: str,
) -> tuple[dict[str, int], dict[int, str]]:
    catalog = load_label_catalog()
    if label_field == "sanction_label":
        ordered_labels = [item["id"] for item in catalog["sanction_labels"]]
    elif label_field == "action_type_label":
        ordered_labels = [item["id"] for item in catalog["action_type_labels"]]
    else:
        raise ManifestError(f"Unsupported label field '{label_field}'.")

    present_labels = {getattr(record, label_field) for record in records}
    labels = [label for label in ordered_labels if label in present_labels]
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label

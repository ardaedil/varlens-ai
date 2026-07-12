from pathlib import Path
from uuid import uuid4

from training.manifest import build_manifest_document, build_label_maps, load_manifest_records


def _prepare_case_dir(name: str) -> Path:
    case_dir = Path("tmp") / f"{name}-{uuid4().hex}"
    case_dir.mkdir(parents=True, exist_ok=False)
    return case_dir


def test_build_manifest_document_normalizes_labels_and_splits():
    case_dir = _prepare_case_dir("training-test-manifest-1")
    annotations = case_dir / "validation_annotations.json"
    annotations.write_text(
        """
        {
          "actions": [
            {
              "id": "action-001",
              "split": "validation",
              "sanction_label": "Offence + Yellow Card",
              "action_type_label": "Holding",
              "views": [
                { "id": "live-01", "video_path": "clip-1.mp4", "view_type": "live" },
                { "id": "replay-01", "video_path": "clip-2.mp4", "view_type": "replay" }
              ]
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    document = build_manifest_document(annotation_paths=[annotations], default_split="valid")
    assert document["summary"]["record_count"] == 2
    assert document["summary"]["split_counts"]["valid"]["actions"] == 1

    first_record = document["records"][0]
    assert first_record["split"] == "valid"
    assert first_record["sanction_label"] == "offence_yellow"
    assert first_record["action_type_label"] == "holding"


def test_load_manifest_records_and_build_label_maps():
    case_dir = _prepare_case_dir("training-test-manifest-2")
    manifest = case_dir / "manifest.json"
    manifest.write_text(
        """
        {
          "records": [
            {
              "sample_id": "train/action-001/view-01",
              "split": "train",
              "action_id": "action-001",
              "view_id": "view-01",
              "group_id": "action-001",
              "video_path": "C:/tmp/clip-1.mp4",
              "relative_video_path": "clip-1.mp4",
              "view_type": "live",
              "sanction_label": "offence_yellow",
              "action_type_label": "holding"
            },
            {
              "sample_id": "valid/action-002/view-01",
              "split": "valid",
              "action_id": "action-002",
              "view_id": "view-01",
              "group_id": "action-002",
              "video_path": "C:/tmp/clip-2.mp4",
              "relative_video_path": "clip-2.mp4",
              "view_type": "replay",
              "sanction_label": "offence_no_card",
              "action_type_label": "pushing"
            }
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    records = load_manifest_records(manifest)
    label2id, id2label = build_label_maps(records, label_field="sanction_label")

    assert len(records) == 2
    assert label2id == {"offence_no_card": 0, "offence_yellow": 1}
    assert id2label == {0: "offence_no_card", 1: "offence_yellow"}


def test_build_manifest_document_supports_official_soccernet_actions():
    case_dir = _prepare_case_dir("training-test-manifest-soccernet")
    annotations = case_dir / "train_annotations.json"
    video_root = case_dir / "Dataset" / "Train" / "action_0"
    video_root.mkdir(parents=True, exist_ok=True)
    (video_root / "clip_0.mp4").write_bytes(b"fake-clip-0")
    (video_root / "clip_1.mp4").write_bytes(b"fake-clip-1")

    annotations.write_text(
        """
        {
          "Set": "Train",
          "Number of actions": 1,
          "Actions": {
            "0": {
              "UrlLocal": "england_epl\\\\2014-2015\\\\2015-02-21 - 18-00 Chelsea 1 - 1 Burnley",
              "Offence": "Offence",
              "Contact": "With contact",
              "Bodypart": "Upper body",
              "Upper body part": "Use of shoulder",
              "Action class": "Challenge",
              "Severity": "1.0",
              "Multiple fouls": "",
              "Try to play": "",
              "Touch ball": "",
              "Handball": "No handball",
              "Handball offence": "",
              "Clips": [
                {
                  "Url": "Dataset/Train/action_0/clip_0",
                  "Camera type": "Main camera center",
                  "Timestamp": 1730826,
                  "Replay speed": 1.0
                },
                {
                  "Url": "Dataset/Train/action_0/clip_1",
                  "Camera type": "Close-up player or field referee",
                  "Timestamp": 1744173,
                  "Replay speed": 1.8
                }
              ]
            }
          }
        }
        """.strip(),
        encoding="utf-8",
    )

    document = build_manifest_document(
        annotation_paths=[annotations],
        video_root=case_dir,
        default_split="train",
        verify_files=True,
    )

    assert document["summary"]["record_count"] == 2
    assert document["summary"]["split_counts"]["train"]["actions"] == 1
    assert document["records"][0]["sanction_label"] == "offence_yellow"
    assert document["records"][0]["action_type_label"] == "challenge"

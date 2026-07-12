from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.manifest import build_label_maps, filter_records, load_manifest_records, summarize_records
from training.metrics import compute_classification_metrics

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "mvfoul_videomae.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VideoMAE classifier for VARLens MVFoul v1.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--task", choices=["sanction", "action"], help="Override the training task.")
    parser.add_argument("--train-split")
    parser.add_argument("--eval-split")
    parser.add_argument("--test-split")
    parser.add_argument("--model-checkpoint")
    parser.add_argument("--model-version")
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-train-epochs", type=float)
    parser.add_argument("--per-device-train-batch-size", type=int)
    parser.add_argument("--per-device-eval-batch-size", type=int)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_training_config(config_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    overrides = {
        "task": args.task,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "test_split": args.test_split,
        "model_checkpoint": args.model_checkpoint,
        "model_version": args.model_version,
        "sample_rate": args.sample_rate,
        "fps": args.fps,
        "num_frames": args.num_frames,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    config.setdefault("model_checkpoint", "MCG-NJU/videomae-base")
    config.setdefault("task", "sanction")
    config.setdefault("train_split", "train")
    config.setdefault("eval_split", "valid")
    config.setdefault("test_split", "test")
    config.setdefault("sample_rate", 4)
    config.setdefault("fps", 30)
    config.setdefault("num_frames", None)
    config.setdefault("learning_rate", 5e-5)
    config.setdefault("num_train_epochs", 4)
    config.setdefault("per_device_train_batch_size", 2)
    config.setdefault("per_device_eval_batch_size", 2)
    config.setdefault("warmup_ratio", 0.1)
    config.setdefault("weight_decay", 0.01)
    config.setdefault("logging_steps", 10)
    config.setdefault("save_only_model", True)
    config.setdefault("model_version", f"videomae-mvfoul-v1-{config['task']}")
    return config


def label_field_for_task(task: str) -> str:
    if task == "sanction":
        return "sanction_label"
    if task == "action":
        return "action_type_label"
    raise ValueError(f"Unsupported task '{task}'.")


def slice_records(records: list[Any], limit: int | None) -> list[Any]:
    if limit is None:
        return records
    return records[:limit]


def persist_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_dry_run_summary(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    label2id: dict[str, int],
    id2label: dict[int, str],
    train_records: list[Any],
    eval_records: list[Any],
    test_records: list[Any],
) -> None:
    clip_duration = None
    if config["num_frames"]:
        clip_duration = round(config["num_frames"] * config["sample_rate"] / config["fps"], 3)

    payload = {
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "task": config["task"],
        "label_field": label_field_for_task(config["task"]),
        "model_checkpoint": config["model_checkpoint"],
        "model_version": config["model_version"],
        "train_split": config["train_split"],
        "eval_split": config["eval_split"],
        "test_split": config["test_split"],
        "sample_rate": config["sample_rate"],
        "fps": config["fps"],
        "num_frames": config["num_frames"],
        "clip_duration_seconds": clip_duration,
        "label2id": label2id,
        "id2label": {str(key): value for key, value in id2label.items()},
        "train_summary": summarize_records(train_records),
        "eval_summary": summarize_records(eval_records),
        "test_summary": summarize_records(test_records),
    }
    persist_json(args.output_dir / "dry-run-summary.json", payload)
    print(f"Dry run summary written to {args.output_dir / 'dry-run-summary.json'}")


def main() -> None:
    args = parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")

    config = load_training_config(args.config, args)
    records = load_manifest_records(args.manifest)
    label_field = label_field_for_task(config["task"])
    label2id, id2label = build_label_maps(records, label_field=label_field)

    train_records = slice_records(
        filter_records(records, {config["train_split"]}),
        args.max_train_samples,
    )
    eval_records = slice_records(
        filter_records(records, {config["eval_split"]}),
        args.max_eval_samples,
    )
    test_records = slice_records(
        filter_records(records, {config["test_split"]}),
        args.max_test_samples,
    )

    if not train_records:
        raise SystemExit(f"No training records found for split '{config['train_split']}'.")
    if not eval_records:
        raise SystemExit(f"No evaluation records found for split '{config['eval_split']}'.")

    if args.dry_run:
        write_dry_run_summary(
            args=args,
            config=config,
            label2id=label2id,
            id2label=id2label,
            train_records=train_records,
            eval_records=eval_records,
            test_records=test_records,
        )
        return

    try:
        import numpy as np
        import torch
        from pytorchvideo.data.encoded_video import EncodedVideo
        from pytorchvideo.transforms import ApplyTransformToKey, Normalize, RandomShortSideScale
        from pytorchvideo.transforms import UniformTemporalSubsample
        from torch.utils.data import Dataset
        from torchvision.transforms import Compose, Lambda, RandomCrop, RandomHorizontalFlip, Resize
        from transformers import Trainer, TrainingArguments
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install training/requirements.txt before "
            f"running a non-dry pass. Missing import: {exc.name}"
        ) from exc

    model_checkpoint = config["model_checkpoint"]
    image_processor = VideoMAEImageProcessor.from_pretrained(model_checkpoint)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_checkpoint,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    mean = image_processor.image_mean
    std = image_processor.image_std
    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]
    resize_to = (height, width)
    num_frames = config["num_frames"] or model.config.num_frames
    sample_rate = config["sample_rate"]
    fps = config["fps"]
    clip_duration = num_frames * sample_rate / fps

    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        RandomShortSideScale(min_size=256, max_size=320),
                        RandomCrop(resize_to),
                        RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )
    eval_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        Resize(resize_to),
                    ]
                ),
            ),
        ]
    )

    class MVFoulVideoDataset(Dataset):
        def __init__(
            self,
            records: list[Any],
            *,
            label_field: str,
            label2id: dict[str, int],
            clip_duration: float,
            transform: Any,
            training: bool,
        ) -> None:
            self.records = records
            self.label_field = label_field
            self.label2id = label2id
            self.clip_duration = clip_duration
            self.transform = transform
            self.training = training

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            record = self.records[index]
            encoded_video = EncodedVideo.from_path(record.video_path, decode_audio=False)
            duration = getattr(encoded_video, "duration", None) or self.clip_duration
            start_limit = max(duration - self.clip_duration, 0.0)
            start_sec = random.uniform(0.0, start_limit) if self.training and start_limit else 0.0
            clip = encoded_video.get_clip(start_sec=start_sec, end_sec=start_sec + self.clip_duration)
            if not clip or "video" not in clip:
                raise ValueError(f"Could not decode clip for sample '{record.sample_id}'.")
            sample = {
                "video": clip["video"],
                "label": self.label2id[getattr(record, self.label_field)],
                "sample_id": record.sample_id,
            }
            if self.transform is not None:
                sample = self.transform(sample)
                sample["label"] = self.label2id[getattr(record, self.label_field)]
                sample["sample_id"] = record.sample_id
            return sample

    train_dataset = MVFoulVideoDataset(
        train_records,
        label_field=label_field,
        label2id=label2id,
        clip_duration=clip_duration,
        transform=train_transform,
        training=True,
    )
    eval_dataset = MVFoulVideoDataset(
        eval_records,
        label_field=label_field,
        label2id=label2id,
        clip_duration=clip_duration,
        transform=eval_transform,
        training=False,
    )
    test_dataset = (
        MVFoulVideoDataset(
            test_records,
            label_field=label_field,
            label2id=label2id,
            clip_duration=clip_duration,
            transform=eval_transform,
            training=False,
        )
        if test_records
        else None
    )

    def collate_fn(examples: list[dict[str, Any]]) -> dict[str, Any]:
        pixel_values = torch.stack([example["video"].permute(1, 0, 2, 3) for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        predictions = np.argmax(eval_pred.predictions, axis=1).tolist()
        references = eval_pred.label_ids.tolist()
        metrics = compute_classification_metrics(references, predictions, id2label=id2label)
        return {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "balanced_accuracy": metrics["balanced_accuracy"],
        }

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        logging_steps=config["logging_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        save_only_model=config["save_only_model"],
        dataloader_num_workers=args.num_workers,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model()
    image_processor.save_pretrained(args.output_dir)

    train_summary = {
        "task": config["task"],
        "label_field": label_field,
        "label2id": label2id,
        "id2label": {str(key): value for key, value in id2label.items()},
        "model_checkpoint": model_checkpoint,
        "model_version": config["model_version"],
        "sample_rate": sample_rate,
        "fps": fps,
        "num_frames": num_frames,
        "clip_duration_seconds": round(clip_duration, 3),
    }
    train_summary["eval_metrics"] = trainer.evaluate()

    if test_dataset is not None:
        test_predictions = trainer.predict(test_dataset)
        predicted_ids = np.argmax(test_predictions.predictions, axis=1).tolist()
        reference_ids = test_predictions.label_ids.tolist()
        train_summary["test_metrics"] = compute_classification_metrics(
            reference_ids,
            predicted_ids,
            id2label=id2label,
        )
        persist_json(
            args.output_dir / "test_predictions.json",
            {
                "references": reference_ids,
                "predictions": predicted_ids,
                "id2label": {str(key): value for key, value in id2label.items()},
            },
        )

    persist_json(args.output_dir / "training_summary.json", train_summary)


if __name__ == "__main__":
    main()

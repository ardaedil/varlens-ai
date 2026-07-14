from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
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
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--warmup-ratio", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--class-weighting",
        choices=["none", "inverse_frequency", "effective_num"],
    )
    parser.add_argument("--class-weight-beta", type=float)
    parser.add_argument("--label-smoothing", type=float)
    parser.add_argument("--loss-function", choices=["cross_entropy", "focal"])
    parser.add_argument("--focal-gamma", type=float)
    parser.add_argument("--train-sampler", choices=["none", "weighted_random"])
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
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "class_weighting": args.class_weighting,
        "class_weight_beta": args.class_weight_beta,
        "label_smoothing": args.label_smoothing,
        "loss_function": args.loss_function,
        "focal_gamma": args.focal_gamma,
        "train_sampler": args.train_sampler,
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
    config.setdefault("gradient_accumulation_steps", 1)
    config.setdefault("warmup_ratio", 0.1)
    config.setdefault("weight_decay", 0.01)
    config.setdefault("logging_steps", 10)
    config.setdefault("fp16", False)
    config.setdefault("bf16", False)
    config.setdefault("class_weighting", "effective_num")
    config.setdefault("class_weight_beta", 0.9999)
    config.setdefault("label_smoothing", 0.0)
    config.setdefault("loss_function", "focal" if config["task"] == "action" else "cross_entropy")
    config.setdefault("focal_gamma", 2.0)
    config.setdefault("train_sampler", "weighted_random" if config["task"] == "action" else "none")
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


def count_labels(records: list[Any], *, label_field: str, labels: list[str]) -> dict[str, int]:
    counts = Counter(getattr(record, label_field) for record in records)
    return {label: counts.get(label, 0) for label in labels}


def compute_class_weights(
    counts: dict[str, int],
    *,
    strategy: str,
    beta: float = 0.9999,
) -> list[float]:
    """Return mean-one weights in the supplied label order."""
    if strategy not in {"none", "inverse_frequency", "effective_num"}:
        raise ValueError(f"Unsupported class-weighting strategy '{strategy}'.")
    if not counts:
        raise ValueError("At least one class is required to compute class weights.")
    if any(count <= 0 for count in counts.values()):
        missing = [label for label, count in counts.items() if count <= 0]
        raise ValueError(f"Cannot compute class weights for classes without training samples: {missing}")
    if strategy == "none":
        return [1.0 for _ in counts]
    if not 0.0 <= beta < 1.0:
        raise ValueError("class-weight-beta must be greater than or equal to 0 and less than 1.")

    total = sum(counts.values())
    class_count = len(counts)
    if strategy == "inverse_frequency":
        raw_weights = [total / (class_count * count) for count in counts.values()]
    else:
        raw_weights = [
            (1.0 - beta) / (1.0 - math.pow(beta, count)) if beta else 1.0
            for count in counts.values()
        ]
    mean_weight = sum(raw_weights) / len(raw_weights)
    return [round(weight / mean_weight, 6) for weight in raw_weights]


def compute_sample_weights(
    records: list[Any],
    *,
    label_field: str,
    label_weights: dict[str, float],
) -> list[float]:
    if not records:
        return []
    return [float(label_weights[getattr(record, label_field)]) for record in records]


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
    train_label_counts: dict[str, int],
    class_weights: list[float],
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
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "per_device_eval_batch_size": config["per_device_eval_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "fp16": config["fp16"],
        "bf16": config["bf16"],
        "clip_duration_seconds": clip_duration,
        "label2id": label2id,
        "id2label": {str(key): value for key, value in id2label.items()},
        "train_summary": summarize_records(train_records),
        "eval_summary": summarize_records(eval_records),
        "test_summary": summarize_records(test_records),
        "class_weighting": config["class_weighting"],
        "class_weight_beta": config["class_weight_beta"],
        "label_smoothing": config["label_smoothing"],
        "loss_function": config["loss_function"],
        "focal_gamma": config["focal_gamma"],
        "train_sampler": config["train_sampler"],
        "train_label_counts": train_label_counts,
        "class_weights": class_weights,
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

    ordered_labels = [id2label[index] for index in range(len(id2label))]
    train_label_counts = count_labels(train_records, label_field=label_field, labels=ordered_labels)
    class_weights = compute_class_weights(
        train_label_counts,
        strategy=config["class_weighting"],
        beta=config["class_weight_beta"],
    )

    if args.dry_run:
        write_dry_run_summary(
            args=args,
            config=config,
            label2id=label2id,
            id2label=id2label,
            train_records=train_records,
            eval_records=eval_records,
            test_records=test_records,
            train_label_counts=train_label_counts,
            class_weights=class_weights,
        )
        return

    try:
        import numpy as np
        import torch
        from pytorchvideo.data.encoded_video import EncodedVideo
        from pytorchvideo.transforms import ApplyTransformToKey, Normalize, RandomShortSideScale
        from pytorchvideo.transforms import UniformTemporalSubsample
        from torch.utils.data import Dataset, WeightedRandomSampler
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

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    label_weight_map = dict(zip(ordered_labels, class_weights))
    sample_weights = compute_sample_weights(
        train_records,
        label_field=label_field,
        label_weights=label_weight_map,
    )

    class WeightedLossTrainer(Trainer):
        def _get_train_sampler(self):
            if config["train_sampler"] == "weighted_random":
                weights = torch.tensor(sample_weights, dtype=torch.double)
                return WeightedRandomSampler(weights, num_samples=len(sample_weights), replacement=True)
            return super()._get_train_sampler()

        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            weights = class_weights_tensor.to(outputs.logits.device)
            per_example_loss = torch.nn.functional.cross_entropy(
                outputs.logits,
                labels,
                weight=weights,
                label_smoothing=config["label_smoothing"],
                reduction="none",
            )
            if config["loss_function"] == "focal":
                probabilities = torch.softmax(outputs.logits, dim=-1)
                true_class_probabilities = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
                focal_factor = torch.pow(1.0 - true_class_probabilities, config["focal_gamma"])
                loss = (focal_factor * per_example_loss).mean()
            else:
                loss = per_example_loss.mean()
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_train_epochs"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        logging_steps=config["logging_steps"],
        fp16=config["fp16"],
        bf16=config["bf16"],
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        save_only_model=config["save_only_model"],
        dataloader_num_workers=args.num_workers,
        report_to=[],
    )

    trainer = WeightedLossTrainer(
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
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "per_device_eval_batch_size": config["per_device_eval_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "fp16": config["fp16"],
        "bf16": config["bf16"],
        "clip_duration_seconds": round(clip_duration, 3),
        "class_weighting": config["class_weighting"],
        "class_weight_beta": config["class_weight_beta"],
        "label_smoothing": config["label_smoothing"],
        "loss_function": config["loss_function"],
        "focal_gamma": config["focal_gamma"],
        "train_sampler": config["train_sampler"],
        "train_label_counts": train_label_counts,
        "class_weights": class_weights,
        "best_checkpoint": trainer.state.best_model_checkpoint,
    }
    eval_predictions = trainer.predict(eval_dataset, metric_key_prefix="eval")
    eval_predicted_ids = np.argmax(eval_predictions.predictions, axis=1).tolist()
    eval_reference_ids = eval_predictions.label_ids.tolist()
    train_summary["eval_metrics"] = compute_classification_metrics(
        eval_reference_ids,
        eval_predicted_ids,
        id2label=id2label,
    )
    train_summary["eval_runtime_metrics"] = eval_predictions.metrics
    persist_json(
        args.output_dir / "eval_predictions.json",
        {
            "model_version": config["model_version"],
            "task": config["task"],
            "split": config["eval_split"],
            "references": eval_reference_ids,
            "predictions": eval_predicted_ids,
            "id2label": {str(key): value for key, value in id2label.items()},
        },
    )

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
                "model_version": config["model_version"],
                "task": config["task"],
                "split": config["test_split"],
                "references": reference_ids,
                "predictions": predicted_ids,
                "id2label": {str(key): value for key, value in id2label.items()},
            },
        )

    persist_json(args.output_dir / "training_summary.json", train_summary)


if __name__ == "__main__":
    main()

from pathlib import Path
from types import SimpleNamespace

from training.train_videomae import (
    DEFAULT_CONFIG_PATH,
    compute_class_weights,
    compute_sample_weights,
    load_training_config,
)


def test_effective_number_weights_are_mean_one_and_favor_rare_classes():
    weights = compute_class_weights(
        {"common": 100, "rare": 10},
        strategy="effective_num",
        beta=0.9999,
    )

    assert weights[1] > weights[0]
    assert round(sum(weights) / len(weights), 6) == 1.0


def test_none_class_weighting_returns_uniform_weights():
    assert compute_class_weights(
        {"a": 100, "b": 1},
        strategy="none",
    ) == [1.0, 1.0]


def test_compute_sample_weights_uses_label_field_values():
    records = [
        SimpleNamespace(action_type_label="rare"),
        SimpleNamespace(action_type_label="common"),
        SimpleNamespace(action_type_label="rare"),
    ]

    assert compute_sample_weights(
        records,
        label_field="action_type_label",
        label_weights={"common": 0.5, "rare": 2.0},
    ) == [2.0, 0.5, 2.0]


def test_action_defaults_enable_focal_loss_and_weighted_sampling():
    config_path = Path("__does_not_exist__.json")
    args = SimpleNamespace(
        task="action",
        train_split=None,
        eval_split=None,
        test_split=None,
        model_checkpoint=None,
        model_version=None,
        sample_rate=None,
        fps=None,
        num_frames=None,
        learning_rate=None,
        num_train_epochs=None,
        per_device_train_batch_size=None,
        per_device_eval_batch_size=None,
        warmup_ratio=None,
        weight_decay=None,
        logging_steps=None,
        class_weighting=None,
        class_weight_beta=None,
        label_smoothing=None,
        loss_function=None,
        focal_gamma=None,
        train_sampler=None,
    )

    config = load_training_config(config_path, args)

    assert config["loss_function"] == "focal"
    assert config["train_sampler"] == "weighted_random"


def test_default_config_preserves_action_specific_quality_settings():
    args = SimpleNamespace(
        task="action",
        train_split=None,
        eval_split=None,
        test_split=None,
        model_checkpoint=None,
        model_version=None,
        sample_rate=None,
        fps=None,
        num_frames=None,
        learning_rate=None,
        num_train_epochs=None,
        per_device_train_batch_size=None,
        per_device_eval_batch_size=None,
        warmup_ratio=None,
        weight_decay=None,
        logging_steps=None,
        class_weighting=None,
        class_weight_beta=None,
        label_smoothing=None,
        loss_function=None,
        focal_gamma=None,
        train_sampler=None,
    )

    config = load_training_config(DEFAULT_CONFIG_PATH, args)

    assert config["loss_function"] == "focal"
    assert config["train_sampler"] == "weighted_random"

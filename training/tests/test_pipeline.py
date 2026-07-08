from argparse import Namespace
from pathlib import Path

from training.run_mvfoul_pipeline import (
    build_pipeline_commands,
    build_serving_env,
    resolve_pipeline_paths,
)


def test_build_serving_env_points_api_to_both_model_dirs():
    paths = resolve_pipeline_paths(Path("output/mvfoul-run"))

    env_text = build_serving_env(paths=paths, api_model_version="videomae-mvfoul-v1")

    assert "VARLENS_INFERENCE_BACKEND=videomae" in env_text
    assert "VARLENS_MODEL_VERSION=videomae-mvfoul-v1" in env_text
    assert str(paths.sanction_dir.resolve()) in env_text
    assert str(paths.action_dir.resolve()) in env_text


def test_build_pipeline_commands_for_dry_run_covers_manifest_and_both_tasks():
    args = Namespace(
        annotations=[Path("data/train.json"), Path("data/valid.json")],
        video_root=Path("dataset/videos"),
        default_split=None,
        verify_files=False,
        output_root=Path("output/mvfoul-run"),
        train_split="train",
        eval_split="valid",
        test_split="test",
        phase="dry-run",
        sanction_model_version="videomae-mvfoul-v1-sanction",
        action_model_version="videomae-mvfoul-v1-action",
        api_model_version="videomae-mvfoul-v1",
        python_executable="python",
    )
    paths = resolve_pipeline_paths(args.output_root)

    commands = build_pipeline_commands(args, paths)

    assert [command.name for command in commands] == [
        "build_manifest",
        "dry_run_sanction",
        "dry_run_action",
    ]
    assert "--output-dir" in commands[1].command
    assert str(paths.manifest_path) in commands[1].command
    assert "--dry-run" in commands[1].command
    assert "train" in commands[1].command
    assert "action" in commands[2].command

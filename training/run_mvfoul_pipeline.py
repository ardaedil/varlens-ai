from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class PipelinePaths:
    output_root: Path
    manifest_path: Path
    sanction_dir: Path
    action_dir: Path
    serving_env_path: Path
    pipeline_summary_path: Path


@dataclass(frozen=True)
class PlannedCommand:
    name: str
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the VARLens MVFoul manifest, train, evaluate, and serving workflow."
    )
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--default-split")
    parser.add_argument("--verify-files", action="store_true")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="valid")
    parser.add_argument("--test-split", default="test")
    parser.add_argument(
        "--phase",
        choices=["prepare", "dry-run", "train", "evaluate", "all"],
        default="dry-run",
    )
    parser.add_argument(
        "--sanction-model-version",
        default="videomae-mvfoul-v1-sanction",
    )
    parser.add_argument(
        "--action-model-version",
        default="videomae-mvfoul-v1-action",
    )
    parser.add_argument(
        "--api-model-version",
        default="videomae-mvfoul-v1",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used for child commands.",
    )
    return parser.parse_args()


def resolve_pipeline_paths(output_root: Path) -> PipelinePaths:
    return PipelinePaths(
        output_root=output_root,
        manifest_path=output_root / "mvfoul_manifest.json",
        sanction_dir=output_root / "sanction",
        action_dir=output_root / "action",
        serving_env_path=output_root / "api-serving.env",
        pipeline_summary_path=output_root / "pipeline-summary.json",
    )


def build_serving_env(*, paths: PipelinePaths, api_model_version: str) -> str:
    sanction_dir = paths.sanction_dir.resolve()
    action_dir = paths.action_dir.resolve()
    return (
        "VARLENS_INFERENCE_BACKEND=videomae\n"
        f"VARLENS_MODEL_VERSION={api_model_version}\n"
        f"VARLENS_SANCTION_MODEL_DIR={sanction_dir}\n"
        f"VARLENS_ACTION_MODEL_DIR={action_dir}\n"
    )


def build_pipeline_commands(args: argparse.Namespace, paths: PipelinePaths) -> list[PlannedCommand]:
    commands: list[PlannedCommand] = []
    if args.phase in {"prepare", "dry-run", "train", "all"}:
        commands.append(
            PlannedCommand(
                name="build_manifest",
                command=build_manifest_command(args, paths),
            )
        )

    if args.phase in {"dry-run", "train", "all"}:
        train_dry_run = args.phase == "dry-run"
        commands.append(
            PlannedCommand(
                name="train_sanction" if not train_dry_run else "dry_run_sanction",
                command=build_train_command(
                    python_executable=args.python_executable,
                    task="sanction",
                    model_version=args.sanction_model_version,
                    output_dir=paths.sanction_dir,
                    manifest_path=paths.manifest_path,
                    train_split=args.train_split,
                    eval_split=args.eval_split,
                    test_split=args.test_split,
                    dry_run=train_dry_run,
                ),
            )
        )
        commands.append(
            PlannedCommand(
                name="train_action" if not train_dry_run else "dry_run_action",
                command=build_train_command(
                    python_executable=args.python_executable,
                    task="action",
                    model_version=args.action_model_version,
                    output_dir=paths.action_dir,
                    manifest_path=paths.manifest_path,
                    train_split=args.train_split,
                    eval_split=args.eval_split,
                    test_split=args.test_split,
                    dry_run=train_dry_run,
                ),
            )
        )

    if args.phase in {"evaluate", "all"}:
        commands.extend(
            [
                PlannedCommand(
                    name="evaluate_sanction",
                    command=build_evaluate_command(
                        python_executable=args.python_executable,
                        model_dir=paths.sanction_dir,
                    ),
                ),
                PlannedCommand(
                    name="evaluate_action",
                    command=build_evaluate_command(
                        python_executable=args.python_executable,
                        model_dir=paths.action_dir,
                    ),
                ),
                PlannedCommand(
                    name="model_card_sanction",
                    command=build_model_card_command(
                        python_executable=args.python_executable,
                        task="sanction",
                        model_version=args.sanction_model_version,
                        model_dir=paths.sanction_dir,
                    ),
                ),
                PlannedCommand(
                    name="model_card_action",
                    command=build_model_card_command(
                        python_executable=args.python_executable,
                        task="action",
                        model_version=args.action_model_version,
                        model_dir=paths.action_dir,
                    ),
                ),
            ]
        )

    return commands


def build_manifest_command(args: argparse.Namespace, paths: PipelinePaths) -> list[str]:
    command = [
        args.python_executable,
        "-m",
        "data.build_mvfoul_views",
        "--annotations",
        *[str(path) for path in args.annotations],
        "--output",
        str(paths.manifest_path),
    ]
    if args.video_root is not None:
        command.extend(["--video-root", str(args.video_root)])
    if args.default_split is not None:
        command.extend(["--default-split", args.default_split])
    if args.verify_files:
        command.append("--verify-files")
    return command


def build_train_command(
    *,
    python_executable: str,
    task: str,
    model_version: str,
    output_dir: Path,
    manifest_path: Path,
    train_split: str,
    eval_split: str,
    test_split: str,
    dry_run: bool,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "training.train_videomae",
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(output_dir),
        "--task",
        task,
        "--model-version",
        model_version,
        "--train-split",
        train_split,
        "--eval-split",
        eval_split,
        "--test-split",
        test_split,
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def build_evaluate_command(*, python_executable: str, model_dir: Path) -> list[str]:
    return [
        python_executable,
        "-m",
        "training.evaluate",
        "--predictions",
        str(model_dir / "test_predictions.json"),
        "--output",
        str(model_dir / "eval_summary.json"),
    ]


def build_model_card_command(
    *,
    python_executable: str,
    task: str,
    model_version: str,
    model_dir: Path,
) -> list[str]:
    return [
        python_executable,
        "-m",
        "training.export_model_card",
        "--model-version",
        model_version,
        "--task",
        task,
        "--metrics-summary",
        str(model_dir / "eval_summary.json"),
        "--output",
        str(model_dir / "model_card.md"),
    ]


def write_pipeline_files(
    *,
    args: argparse.Namespace,
    paths: PipelinePaths,
    commands: list[PlannedCommand],
) -> None:
    paths.output_root.mkdir(parents=True, exist_ok=True)
    paths.serving_env_path.write_text(
        build_serving_env(paths=paths, api_model_version=args.api_model_version),
        encoding="utf-8",
    )
    payload = {
        "phase": args.phase,
        "annotations": [str(path) for path in args.annotations],
        "video_root": str(args.video_root) if args.video_root is not None else None,
        "default_split": args.default_split,
        "verify_files": args.verify_files,
        "manifest_path": str(paths.manifest_path),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "test_split": args.test_split,
        "sanction_model_version": args.sanction_model_version,
        "action_model_version": args.action_model_version,
        "api_model_version": args.api_model_version,
        "sanction_dir": str(paths.sanction_dir),
        "action_dir": str(paths.action_dir),
        "serving_env_path": str(paths.serving_env_path),
        "commands": [
            {"name": planned.name, "command": planned.command, "shell": subprocess.list2cmdline(planned.command)}
            for planned in commands
        ],
    }
    paths.pipeline_summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_commands(commands: list[PlannedCommand]) -> None:
    for planned in commands:
        print(f"[varlens] {planned.name}: {subprocess.list2cmdline(planned.command)}")
        subprocess.run(planned.command, check=True)


def main() -> None:
    args = parse_args()
    paths = resolve_pipeline_paths(args.output_root)
    commands = build_pipeline_commands(args, paths)
    write_pipeline_files(args=args, paths=paths, commands=commands)
    run_commands(commands)
    print(f"[varlens] Wrote serving env to {paths.serving_env_path}")
    print(f"[varlens] Wrote pipeline summary to {paths.pipeline_summary_path}")


if __name__ == "__main__":
    main()

# Model Training

This repository now includes a first real pass at the MVFoul model workflow. It is still gated on dataset access and ML dependencies, but the project is no longer limited to placeholder training scripts.

## Workflow

1. Build a manifest from authorized MVFoul annotations:

```bash
python -m data.build_mvfoul_views --annotations path/to/train.json path/to/valid.json path/to/test.json --video-root path/to/mvfoul/videos --verify-files --output output/mvfoul_manifest.json
```

2. Inspect the training plan without importing the ML stack:

```bash
python -m training.train_videomae --manifest output/mvfoul_manifest.json --output-dir output/videomae-run --dry-run
```

3. Install training dependencies and run a real pass:

```bash
python -m pip install -r training/requirements.txt
python -m training.train_videomae --manifest output/mvfoul_manifest.json --output-dir output/videomae-run
```

4. Compute a standalone evaluation summary from saved predictions:

```bash
python -m training.evaluate --predictions output/videomae-run/test_predictions.json --output output/videomae-run/eval_summary.json
```

5. Export a model card scaffold with reported metrics:

```bash
python -m training.export_model_card --model-version videomae-mvfoul-v1 --task sanction --metrics-summary output/videomae-run/eval_summary.json --output output/videomae-run/model_card.md
```

## Notes

- `data/build_mvfoul_views.py` normalizes sanction and action labels against `config/labels.json`.
- The dry-run path works without `torch`, `transformers`, or `pytorchvideo`.
- The training path follows the VideoMAE recipe: derive normalization and resize settings from the image processor, apply temporal subsampling, and fine-tune a classification head on top of the pretrained encoder.
- The current training script assumes action clips are already temporally localized. It does not yet implement multi-clip test-time aggregation or confidence calibration.

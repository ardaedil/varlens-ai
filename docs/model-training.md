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

3. Install training dependencies and run a real pass for the sanction model:

```bash
python -m pip install -r training/requirements.txt
python -m training.train_videomae --manifest output/mvfoul_manifest.json --task sanction --model-version videomae-mvfoul-v1-sanction --output-dir output/videomae-sanction
```

4. Train the action-type model into its own output directory:

```bash
python -m training.train_videomae --manifest output/mvfoul_manifest.json --task action --model-version videomae-mvfoul-v1-action --output-dir output/videomae-action
```

5. Compute a standalone evaluation summary from saved predictions:

```bash
python -m training.evaluate --predictions output/videomae-sanction/test_predictions.json --output output/videomae-sanction/eval_summary.json
```

6. Export a model card scaffold with reported metrics:

```bash
python -m training.export_model_card --model-version videomae-mvfoul-v1-sanction --task sanction --metrics-summary output/videomae-sanction/eval_summary.json --output output/videomae-sanction/model_card.md
```

7. Point the API at both trained directories when you are ready to serve real inference:

```bash
python -m pip install -r services/api/requirements-ml.txt
VARLENS_INFERENCE_BACKEND=videomae
VARLENS_MODEL_VERSION=videomae-mvfoul-v1
VARLENS_SANCTION_MODEL_DIR=output/videomae-sanction
VARLENS_ACTION_MODEL_DIR=output/videomae-action
```

## Notes

- `data/build_mvfoul_views.py` normalizes sanction and action labels against `config/labels.json`.
- The dry-run path works without `torch`, `transformers`, or `pytorchvideo`.
- The training path follows the VideoMAE recipe: derive normalization and resize settings from the image processor, apply temporal subsampling, and fine-tune a classification head on top of the pretrained encoder.
- The serving path expects two fine-tuned checkpoints: one for sanction labels and one for action-type labels.
- The current training script assumes action clips are already temporally localized. It does not yet implement multi-clip test-time aggregation or confidence calibration.

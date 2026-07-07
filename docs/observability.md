# Observability

VARLens should emit vendor-neutral telemetry suitable for OpenTelemetry.

## Required Signals

- Request latency.
- Upload rejection rate by error code.
- Unsupported-scope rate.
- Preprocessing duration.
- Inference duration.
- Model version.
- Confidence distribution by label.
- Delete-after-processing failure count.
- Model availability and degraded-state count.

## Current Implementation

The MVP includes structured request logging in `services/api/app/telemetry/logging.py`. It logs method, path, status, and latency without raw video content.

## Next Production Step

Add OpenTelemetry instrumentation around:

- `POST /api/v1/analyze`
- upload write/delete
- model inference
- explanation generation
- model-info/version checks

Alert when:

- p95 analysis latency exceeds the product target.
- delete-after-processing fails.
- unsupported-scope rate spikes after a UI or routing change.
- model confidence distribution drifts materially from validation baselines.

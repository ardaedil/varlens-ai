# VARLens AI MVP

VARLens AI is an educational soccer clip analysis MVP. It accepts one short uploaded clip, estimates likely foul sanction and action type, and returns an uncertainty-aware explanation of what a referee-style review would inspect.

The v1 scope is intentionally narrow:

- Supported: short-clip foul and sanction explanation.
- Unsupported in v1: offside adjudication, standalone handball determination, penalty/no-penalty decisions, official referee decisions.
- Privacy default: uploaded clips are processed transiently and deleted after inference.

## Repository Layout

```text
apps/web                 Next.js App Router frontend
services/api             FastAPI analysis service
packages/contracts       Shared JSON, TypeScript, and Python contracts
packages/prompts         Deterministic explanation templates
config/labels.json       Canonical label taxonomy
docs/                    Product, security, and observability specs
data/                    Dataset preparation scaffolding
training/                Model training/evaluation scaffolding
```

## Local Setup

Install JavaScript dependencies:

```bash
pnpm install
```

Install backend dependencies:

```bash
python -m pip install -r services/api/requirements.txt -r services/api/requirements-dev.txt
```

Run the API:

```bash
python -m uvicorn services.api.app.main:app --reload --host 0.0.0.0 --port 8000
```

Run the web app:

```bash
pnpm dev:web
```

The frontend proxies analysis requests to `API_BASE_URL`, defaulting to `http://localhost:8000`.

## Validation

```bash
pnpm contract:check
pnpm typecheck
python -m pytest services/api/tests
```

The current inference path is a deterministic stub that matches the response contract. Replace `services/api/app/inference/stub_model.py` with a trained VideoMAE MVFoul adapter once model artifacts are available.

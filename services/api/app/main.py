from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from services.api.app.core.settings import settings
from services.api.app.routes import analyze, health
from services.api.app.security.errors import error_payload
from services.api.app.telemetry.logging import request_logging_middleware

app = FastAPI(
    title="VARLens AI API",
    version=settings.spec_version,
    description="Educational soccer foul and sanction explanation API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.middleware("http")(request_logging_middleware)

app.include_router(analyze.router, prefix="/api/v1", tags=["analysis"])
app.include_router(health.router, prefix="/api/v1", tags=["system"])


@app.exception_handler(RequestValidationError)
async def request_validation_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    request_id = str(uuid4())
    payload = error_payload(
        code="invalid_request",
        message="Request validation failed.",
        request_id=request_id,
        details={"path": request.url.path, "errors": str(exc.errors())},
    )
    return JSONResponse(status_code=400, content=payload.model_dump(exclude_none=True))


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "varlens-api",
        "docs": "/docs",
        "health": "/api/v1/health",
    }

from pathlib import Path

from fastapi.testclient import TestClient

from packages.contracts.analyze import ActionTypePrediction, SanctionPrediction
import services.api.app.routes.analyze as analyze_route
from services.api.app.main import app


client = TestClient(app)
SAMPLE_CLIP = Path("tests/fixtures/sample_clip.mp4")


class _FakeBackend:
    def __init__(self, *, available: bool = True, reason: str = "unavailable") -> None:
        self.info = type(
            "BackendInfoLike",
            (),
            {
                "available": available,
                "details": {"reason": reason},
            },
        )()

    def analyze_clip(self, *, path, filename, content_type, size_bytes):
        del path, filename, content_type, size_bytes
        return (
            "videomae-mvfoul-v1-test",
            SanctionPrediction(
                label="offence_yellow",
                confidence=0.73,
                alternatives=[{"label": "offence_no_card", "confidence": 0.21}],
            ),
            ActionTypePrediction(
                label="holding",
                confidence=0.66,
                alternatives=[{"label": "pushing", "confidence": 0.18}],
            ),
        )


def test_valid_upload_returns_contract_response_and_deletes_temp_file(monkeypatch):
    deleted_paths = []

    async def fake_write_transient_upload(upload, settings):
        return SAMPLE_CLIP, SAMPLE_CLIP.stat().st_size

    def fake_delete_transient_upload(path, settings):
        deleted_paths.append(path)
        return True

    monkeypatch.setattr(analyze_route, "write_transient_upload", fake_write_transient_upload)
    monkeypatch.setattr(analyze_route, "delete_transient_upload", fake_delete_transient_upload)
    monkeypatch.setattr(analyze_route, "get_inference_backend", lambda: _FakeBackend())

    response = client.post(
        "/api/v1/analyze",
        data={"scope": "foul_review_context", "clip_duration_seconds": "8"},
        files={"file": ("clip.mp4", b"fake mp4 bytes", "video/mp4")},
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["review_context"]["official_decision_claimed"] is False
    assert payload["limitations"]
    assert deleted_paths == [SAMPLE_CLIP]


def test_model_unavailable_returns_typed_error(monkeypatch):
    monkeypatch.setattr(
        analyze_route,
        "get_inference_backend",
        lambda: _FakeBackend(available=False, reason="No checkpoints are configured."),
    )

    response = client.post(
        "/api/v1/analyze",
        data={"scope": "foul_review_context", "clip_duration_seconds": "8"},
        files={"file": ("clip.mp4", b"fake mp4 bytes", "video/mp4")},
    )

    payload = response.json()
    assert response.status_code == 503
    assert payload["error"]["code"] == "model_unavailable"
    assert "No checkpoints are configured." in payload["error"]["message"]


def test_unsupported_scope_returns_typed_error():
    response = client.post(
        "/api/v1/analyze",
        data={"scope": "offside", "clip_duration_seconds": "8"},
        files={"file": ("clip.mp4", b"fake mp4 bytes", "video/mp4")},
    )

    payload = response.json()
    assert response.status_code == 422
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "unsupported_scope"


def test_unsupported_media_type_returns_typed_error():
    response = client.post(
        "/api/v1/analyze",
        data={"scope": "foul_review_context", "clip_duration_seconds": "8"},
        files={"file": ("clip.txt", b"not video", "text/plain")},
    )

    payload = response.json()
    assert response.status_code == 415
    assert payload["error"]["code"] == "unsupported_media_type"


def test_clip_too_long_returns_typed_error():
    response = client.post(
        "/api/v1/analyze",
        data={"scope": "foul_review_context", "clip_duration_seconds": "30"},
        files={"file": ("clip.mp4", b"fake mp4 bytes", "video/mp4")},
    )

    payload = response.json()
    assert response.status_code == 422
    assert payload["error"]["code"] == "clip_too_long"

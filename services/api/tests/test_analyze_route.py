from pathlib import Path

from fastapi.testclient import TestClient

import services.api.app.routes.analyze as analyze_route
from services.api.app.main import app


client = TestClient(app)
SAMPLE_CLIP = Path("tests/fixtures/sample_clip.mp4")


def test_valid_upload_returns_contract_response_and_deletes_temp_file(monkeypatch):
    deleted_paths = []

    async def fake_write_transient_upload(upload, settings):
        return SAMPLE_CLIP, SAMPLE_CLIP.stat().st_size

    def fake_delete_transient_upload(path, settings):
        deleted_paths.append(path)
        return True

    monkeypatch.setattr(analyze_route, "write_transient_upload", fake_write_transient_upload)
    monkeypatch.setattr(analyze_route, "delete_transient_upload", fake_delete_transient_upload)

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

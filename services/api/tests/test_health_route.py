from fastapi.testclient import TestClient

import services.api.app.routes.health as health_route
from services.api.app.main import app


client = TestClient(app)


class _FakeBackend:
    def __init__(self, *, available: bool, backend_name: str, uses_stub: bool) -> None:
        self.info = type(
            "BackendInfoLike",
            (),
            {
                "available": available,
                "backend_name": backend_name,
                "model_version": "videomae-mvfoul-v1-sanction+action",
                "uses_stub": uses_stub,
                "details": {"sanction_model_version": "videomae-mvfoul-v1-sanction"},
            },
        )()


def test_health_reports_backend_details(monkeypatch):
    monkeypatch.setattr(
        health_route,
        "get_inference_backend",
        lambda: _FakeBackend(available=True, backend_name="videomae", uses_stub=False),
    )

    response = client.get("/api/v1/health")
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["inference_backend"] == "videomae"
    assert payload["uses_stub"] is False


def test_model_info_reports_stub_fallback(monkeypatch):
    monkeypatch.setattr(
        health_route,
        "get_inference_backend",
        lambda: _FakeBackend(available=True, backend_name="stub", uses_stub=True),
    )

    response = client.get("/api/v1/model-info")
    payload = response.json()

    assert response.status_code == 200
    assert payload["inference_backend"] == "stub"
    assert payload["uses_stub"] is True
    assert "holding" in payload["action_type_labels"]

"""
Tests for the CTV Promo Placement API.

Uses mocked XGBoost models so the test suite runs without model artifacts.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mock_cls_model():
    model = MagicMock()
    # Default: high probability of moving (should_move=1)
    model.predict_proba.return_value = np.array([[0.2, 0.8]])
    model.feature_names_in_ = ["hour", "weather_rain_mm", "break_position_pct", "channel"]
    return model


@pytest.fixture(scope="session")
def mock_reg_model():
    model = MagicMock()
    model.predict.return_value = np.array([0.35])
    return model


@pytest.fixture(scope="session")
def mock_schema():
    return {
        "feature_cols": ["hour", "weather_rain_mm", "break_position_pct", "channel"],
        "categorical_cols": ["channel"],
        "category_values": {"channel": ["ITV1", "ITV2", "ITVBe"]},
    }


@pytest.fixture(autouse=True)
def set_state(mock_cls_model, mock_reg_model, mock_schema):
    """Inject mock models into app state before every test."""
    from main import _state
    _state["cls_model"] = mock_cls_model
    _state["reg_model"] = mock_reg_model
    _state["schema"] = mock_schema
    _state["ready"] = True
    _state["error"] = None


@pytest.fixture(scope="session")
def client():
    from main import app
    # Use raise_server_exceptions=False so we can assert on 5xx responses
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_ok_when_ready(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["schema_loaded"] is True
        assert body["feature_count"] == 4

    def test_503_when_not_ready(self, client):
        from main import _state
        _state["ready"] = False
        _state["error"] = "disk full"
        try:
            r = client.get("/health")
            assert r.status_code == 503
            assert "disk full" in r.json()["detail"]
        finally:
            _state["ready"] = True
            _state["error"] = None

    def test_schema_not_loaded_flag(self, client):
        from main import _state
        original = _state["schema"]
        _state["schema"] = None
        try:
            r = client.get("/health")
            assert r.status_code == 200
            assert r.json()["schema_loaded"] is False
        finally:
            _state["schema"] = original


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_minimal_payload_accepted(self, client):
        r = client.post("/predict", json={})
        assert r.status_code == 200

    def test_response_shape(self, client):
        r = client.post("/predict", json={"promo_title": "Test Promo", "hour": 21})
        assert r.status_code == 200
        body = r.json()
        assert set(body.keys()) >= {"should_move", "move_probability", "predicted_uplift"}

    def test_passthrough_fields_returned(self, client):
        r = client.post("/predict", json={
            "promo_title": "My Show",
            "datetime": "2024-11-01T21:00:00",
        })
        body = r.json()
        assert body["promo_title"] == "My Show"
        assert body["datetime"] == "2024-11-01T21:00:00"

    def test_should_move_1_when_high_proba(self, client, mock_cls_model):
        mock_cls_model.predict_proba.return_value = np.array([[0.1, 0.9]])
        r = client.post("/predict", json={"hour": 20})
        body = r.json()
        assert body["should_move"] == 1
        assert body["move_probability"] == pytest.approx(0.9, abs=1e-3)

    def test_should_move_0_when_low_proba(self, client, mock_cls_model, mock_reg_model):
        mock_cls_model.predict_proba.return_value = np.array([[0.85, 0.15]])
        r = client.post("/predict", json={"hour": 10})
        body = r.json()
        assert body["should_move"] == 0
        assert body["predicted_uplift"] == 0.0
        # Regressor must NOT be called when should_move=0
        mock_reg_model.predict.assert_not_called()
        mock_reg_model.predict.reset_mock()

    def test_503_when_not_ready(self, client):
        from main import _state
        _state["ready"] = False
        try:
            r = client.post("/predict", json={})
            assert r.status_code == 503
        finally:
            _state["ready"] = True

    def test_field_validation_hour_out_of_range(self, client):
        r = client.post("/predict", json={"hour": 25})
        assert r.status_code == 422

    def test_field_validation_break_position_pct(self, client):
        r = client.post("/predict", json={"break_position_pct": 1.5})
        assert r.status_code == 422

    def test_extra_fields_forwarded(self, client):
        # Extra fields should not cause a validation error
        r = client.post("/predict", json={"unknown_feature_xyz": 99.9})
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# /predict/batch
# ---------------------------------------------------------------------------

class TestPredictBatch:
    def test_basic_batch(self, client, mock_cls_model, mock_reg_model):
        mock_cls_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])
        mock_reg_model.predict.return_value = np.array([0.4])
        r = client.post("/predict/batch", json=[
            {"promo_title": "Show A", "hour": 21},
            {"promo_title": "Show B", "hour": 14},
        ])
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 2
        assert body[0]["should_move"] == 1
        assert body[1]["should_move"] == 0
        assert body[1]["predicted_uplift"] == 0.0

    def test_empty_batch_rejected(self, client):
        r = client.post("/predict/batch", json=[])
        assert r.status_code == 400

    def test_batch_limit_enforced(self, client):
        payload = [{"hour": i % 24} for i in range(501)]
        r = client.post("/predict/batch", json=payload)
        assert r.status_code == 400
        assert "limit" in r.json()["detail"].lower()

    def test_passthrough_per_item(self, client, mock_cls_model, mock_reg_model):
        mock_cls_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.4, 0.6]])
        mock_reg_model.predict.return_value = np.array([0.2, 0.3])
        r = client.post("/predict/batch", json=[
            {"promo_title": "Alpha"},
            {"promo_title": "Beta"},
        ])
        body = r.json()
        assert body[0]["promo_title"] == "Alpha"
        assert body[1]["promo_title"] == "Beta"

    def test_single_item_batch(self, client, mock_cls_model, mock_reg_model):
        mock_cls_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        mock_reg_model.predict.return_value = np.array([0.5])
        r = client.post("/predict/batch", json=[{"hour": 20}])
        assert r.status_code == 200
        assert len(r.json()) == 1


# ---------------------------------------------------------------------------
# /reload
# ---------------------------------------------------------------------------

class TestReload:
    def test_reload_success(self, client):
        with patch("main._load_models") as mock_load:
            r = client.post("/reload")
            assert r.status_code == 200
            assert r.json()["status"] == "reloaded"
            mock_load.assert_called_once()

    def test_reload_failure_returns_500(self, client):
        with patch("main._load_models", side_effect=FileNotFoundError("pkl missing")):
            r = client.post("/reload")
            assert r.status_code == 500
            assert "pkl missing" in r.json()["detail"]

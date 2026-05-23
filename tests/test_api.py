"""
Tests for the CTV Promo Placement API.

Uses mocked XGBoost models so the test suite runs without model artifacts.
"""

import numpy as np
import pandas as pd
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
    mock_cls_model.reset_mock()
    mock_reg_model.reset_mock()
    from main import _state
    _state["cls_model"] = mock_cls_model
    _state["reg_model"] = mock_reg_model
    _state["schema"] = mock_schema
    _state["ready"] = True
    _state["error"] = None
    _state["training"] = {
        "status": "idle",
        "started_at": None,
        "finished_at": None,
        "metrics": None,
        "error": None,
    }


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


# ---------------------------------------------------------------------------
# /feedback
# ---------------------------------------------------------------------------

class TestFeedback:
    def test_submit_single_row(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("main.FEEDBACK_PATH", tmp_path / "feedback.csv")
        r = client.post("/feedback", json=[{
            "promo_title": "Test Show",
            "hour": 21,
            "should_move": 1,
            "uplift_if_optimised": 0.35,
        }])
        assert r.status_code == 200
        body = r.json()
        assert body["accepted"] == 1
        assert body["total_feedback_rows"] == 1

    def test_submit_appends_rows(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("main.FEEDBACK_PATH", tmp_path / "feedback.csv")
        client.post("/feedback", json=[{"should_move": 0}])
        r = client.post("/feedback", json=[{"should_move": 1}, {"should_move": 0}])
        body = r.json()
        assert body["accepted"] == 2
        assert body["total_feedback_rows"] == 3

    def test_missing_should_move_rejected(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("main.FEEDBACK_PATH", tmp_path / "feedback.csv")
        r = client.post("/feedback", json=[{"promo_title": "No Label"}])
        assert r.status_code == 422

    def test_should_move_out_of_range_rejected(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("main.FEEDBACK_PATH", tmp_path / "feedback.csv")
        r = client.post("/feedback", json=[{"should_move": 5}])
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# /retrain/status
# ---------------------------------------------------------------------------

class TestRetrainStatus:
    def test_returns_idle_by_default(self, client):
        r = client.get("/retrain/status")
        assert r.status_code == 200
        assert r.json()["status"] == "idle"

    def test_reflects_updated_state(self, client):
        from main import _state
        _state["training"]["status"] = "done"
        _state["training"]["metrics"] = {"pr_auc": 0.75}
        r = client.get("/retrain/status")
        assert r.json()["status"] == "done"
        assert r.json()["metrics"]["pr_auc"] == 0.75


# ---------------------------------------------------------------------------
# /retrain
# ---------------------------------------------------------------------------

class TestRetrain:
    def test_404_when_no_feedback_file(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("main.FEEDBACK_PATH", tmp_path / "nonexistent.csv")
        r = client.post("/retrain")
        assert r.status_code == 404

    def test_400_when_too_few_rows(self, client, tmp_path, monkeypatch):
        path = tmp_path / "feedback.csv"
        pd.DataFrame({"should_move": [0, 1]}).to_csv(path, index=False)
        monkeypatch.setattr("main.FEEDBACK_PATH", path)
        r = client.post("/retrain", params={"min_rows": 10})
        assert r.status_code == 400
        assert "10" in r.json()["detail"]

    def test_409_when_already_running(self, client):
        from main import _state
        _state["training"]["status"] = "running"
        r = client.post("/retrain")
        assert r.status_code == 409

    def test_starts_when_enough_rows(self, client, tmp_path, monkeypatch):
        path = tmp_path / "feedback.csv"
        df = pd.DataFrame({
            "should_move": [0, 1] * 10,
            "uplift_if_optimised": [0.0, 0.5] * 10,
        })
        df.to_csv(path, index=False)
        monkeypatch.setattr("main.FEEDBACK_PATH", path)

        with patch("main.threading.Thread") as mock_thread:
            mock_thread.return_value.start = lambda: None
            r = client.post("/retrain", params={"min_rows": 5})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "started"
        assert body["rows"] == 20


# ---------------------------------------------------------------------------
# /retrain/upload
# ---------------------------------------------------------------------------

class TestRetrainUpload:
    def _labeled_csv_bytes(self, n: int = 25) -> bytes:
        import io
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "should_move": rng.integers(0, 2, n),
            "uplift_if_optimised": rng.uniform(0, 1, n),
            "hour": rng.integers(0, 24, n),
        })
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        return buf.getvalue()

    def test_rejects_non_csv(self, client):
        r = client.post(
            "/retrain/upload",
            files={"file": ("data.xlsx", b"binary", "application/vnd.ms-excel")},
        )
        assert r.status_code == 400

    def test_rejects_missing_labels(self, client):
        import io
        df = pd.DataFrame({"hour": [20, 21], "channel": ["ITV1", "ITV2"]})
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        r = client.post(
            "/retrain/upload",
            files={"file": ("data.csv", buf.getvalue(), "text/csv")},
        )
        assert r.status_code == 400
        assert "should_move" in r.json()["detail"]

    def test_409_when_already_running(self, client):
        from main import _state
        _state["training"]["status"] = "running"
        r = client.post(
            "/retrain/upload",
            files={"file": ("x.csv", b"should_move\n1\n0", "text/csv")},
        )
        assert r.status_code == 409

    def test_starts_training_with_valid_csv(self, client, tmp_path, monkeypatch):
        monkeypatch.setattr("main.FEEDBACK_PATH", tmp_path / "feedback.csv")
        with patch("main.threading.Thread") as mock_thread:
            mock_thread.return_value.start = lambda: None
            r = client.post(
                "/retrain/upload",
                files={"file": ("labeled.csv", self._labeled_csv_bytes(), "text/csv")},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "started"
        assert body["rows"] == 25

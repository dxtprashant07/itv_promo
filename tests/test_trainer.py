"""
tests/test_trainer.py
=====================
End-to-end tests: synthetic data → train_models() → saved artifacts → predict.
Runs without network access or pre-existing model files.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from config import TARGET_CLS, TARGET_REG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 60) -> pd.DataFrame:
    """Minimal labeled DataFrame that satisfies train_models() requirements."""
    rng = np.random.default_rng(42)
    # Mix of numeric + categorical features, plus required label columns
    return pd.DataFrame({
        "hour":                  rng.integers(0, 24, n),
        "minute":                rng.integers(0, 60, n),
        "day_of_week":           rng.integers(0, 7, n),
        "is_weekend":            rng.integers(0, 2, n),
        "promo_length_seconds":  rng.uniform(10, 60, n),
        "break_position_pct":    rng.uniform(0, 1, n),
        "weather_rain_mm":       rng.uniform(0, 10, n),
        "promo_fatigue_index":   rng.uniform(0, 1, n),
        "channel":               rng.choice(["ITV1", "ITV2", "ITVBe"], n),
        TARGET_CLS:              rng.integers(0, 2, n),
        TARGET_REG:              rng.uniform(0, 1, n),
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrainModels:
    def test_full_train_returns_metrics(self, tmp_path):
        from trainer import train_models
        df = _make_df()
        metrics = train_models(df, tmp_path, incremental=False)

        assert metrics["rows_trained"] == len(df)
        assert metrics["features"] >= 1
        assert "trained_at" in metrics
        assert metrics["incremental"] is False
        if metrics["pr_auc"] is not None:
            assert 0.0 <= metrics["pr_auc"] <= 1.0

    def test_artifacts_written(self, tmp_path):
        from trainer import train_models
        train_models(_make_df(), tmp_path, incremental=False)

        assert (tmp_path / "promo_move_classifier.pkl").exists()
        assert (tmp_path / "promo_uplift_regressor.pkl").exists()
        assert (tmp_path / "inference_schema.json").exists()

    def test_schema_contents(self, tmp_path):
        from trainer import train_models
        train_models(_make_df(), tmp_path, incremental=False)

        schema = json.loads((tmp_path / "inference_schema.json").read_text())
        assert "feature_cols" in schema
        assert "channel" in schema["categorical_cols"]
        assert len(schema["category_values"]["channel"]) > 0

    def test_loaded_classifier_predicts(self, tmp_path):
        from trainer import train_models
        df = _make_df()
        train_models(df, tmp_path, incremental=False)

        cls_model = joblib.load(tmp_path / "promo_move_classifier.pkl")
        sample = df.drop(columns=[TARGET_CLS, TARGET_REG]).iloc[:5].copy()
        sample["channel"] = sample["channel"].astype("category")

        probas = cls_model.predict_proba(sample)
        assert probas.shape == (5, 2)
        assert np.all((probas >= 0) & (probas <= 1))
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-5)

    def test_loaded_regressor_predicts(self, tmp_path):
        from trainer import train_models
        df = _make_df()
        train_models(df, tmp_path, incremental=False)

        reg_model = joblib.load(tmp_path / "promo_uplift_regressor.pkl")
        # Regressor trained only on move=1 rows; sample a few
        move_rows = df[df[TARGET_CLS] == 1].drop(columns=[TARGET_CLS, TARGET_REG]).iloc[:3].copy()
        if len(move_rows) == 0:
            pytest.skip("No positive rows in synthetic data")
        move_rows["channel"] = move_rows["channel"].astype("category")
        preds = reg_model.predict(move_rows)
        assert preds.shape == (len(move_rows),)

    def test_incremental_adds_trees(self, tmp_path):
        from trainer import train_models
        df = _make_df()
        train_models(df, tmp_path, incremental=False)

        n_initial = len(
            joblib.load(tmp_path / "promo_move_classifier.pkl").get_booster().get_dump()
        )

        metrics = train_models(df, tmp_path, incremental=True, n_new_trees=20)
        assert metrics["incremental"] is True

        n_after = len(
            joblib.load(tmp_path / "promo_move_classifier.pkl").get_booster().get_dump()
        )
        assert n_after == n_initial + 20

    def test_incremental_falls_back_when_no_existing_model(self, tmp_path):
        from trainer import train_models
        # No existing pkl → incremental should silently fall back to full train
        df = _make_df()
        metrics = train_models(df, tmp_path, incremental=True, n_new_trees=50)
        # Fall-back full train: incremental flag becomes False
        assert metrics["incremental"] is False
        assert (tmp_path / "promo_move_classifier.pkl").exists()

    def test_missing_target_cls_raises(self, tmp_path):
        from trainer import train_models
        df = _make_df().drop(columns=[TARGET_CLS])
        with pytest.raises(ValueError, match=TARGET_CLS):
            train_models(df, tmp_path)

    def test_missing_target_reg_raises(self, tmp_path):
        from trainer import train_models
        df = _make_df().drop(columns=[TARGET_REG])
        with pytest.raises(ValueError, match=TARGET_REG):
            train_models(df, tmp_path)

    def test_no_feature_cols_raises(self, tmp_path):
        from trainer import train_models
        df = pd.DataFrame({
            TARGET_CLS: [0, 1, 0, 1, 0],
            TARGET_REG: [0.1, 0.5, 0.2, 0.8, 0.0],
        })
        with pytest.raises(ValueError, match="No feature columns"):
            train_models(df, tmp_path)

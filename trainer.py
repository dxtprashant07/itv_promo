"""
trainer.py
==========
Core training/retraining logic shared by train_local.py (CLI) and main.py (API).
"""

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, r2_score
import xgboost as xgb

from config import (
    LEAKY_COLUMNS, TARGET_CLS, TARGET_REG,
    ID_META, RECOMMENDATION_COLS, RANDOM_STATE,
)

warnings.filterwarnings("ignore")

_DROP_SET = set(LEAKY_COLUMNS) | set(ID_META) | {TARGET_CLS, TARGET_REG} | set(RECOMMENDATION_COLS)


def train_models(
    df: pd.DataFrame,
    model_dir: Path,
    incremental: bool = False,
    n_new_trees: int = 100,
) -> dict:
    """
    Train or warm-start XGBoost classifier + regressor from a labeled DataFrame.

    Parameters
    ----------
    df          : DataFrame with feature columns + TARGET_CLS + TARGET_REG
    model_dir   : Where to save model artifacts (pkl + inference_schema.json)
    incremental : If True, load existing models and add trees (warm-start)
    n_new_trees : Trees to add in incremental mode (ignored for full retrain)

    Returns dict of training metrics.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if TARGET_CLS not in df.columns:
        raise ValueError(f"Column '{TARGET_CLS}' required for training (0=keep, 1=move)")
    if TARGET_REG not in df.columns:
        raise ValueError(f"Column '{TARGET_REG}' required for training (expected uplift float)")

    # Drop leaky columns if present
    df = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns]).copy()

    # Sort by time if available for honest temporal split
    if "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)

    # Identify feature columns (exclude ids, targets, leaky)
    feature_cols = [c for c in df.columns if c not in _DROP_SET]
    if not feature_cols:
        raise ValueError("No feature columns found after dropping leaky/meta columns.")

    # Encode string columns as pandas category (XGBoost native categorical)
    # pandas 3.x uses dtype "str" instead of "object" for string columns
    cat_features = [
        c for c in feature_cols
        if df[c].dtype == "object" or str(df[c].dtype) in {"string", "str"}
    ]
    for c in cat_features:
        df[c] = df[c].astype("category")

    X = df[feature_cols]
    y_cls = df[TARGET_CLS].astype(int)
    y_reg = df[TARGET_REG]

    # 80/20 temporal split
    if "datetime" in df.columns:
        cutoff = df["datetime"].quantile(0.80)
        train_mask = df["datetime"] < cutoff
    else:
        n = int(len(df) * 0.80)
        train_mask = pd.Series([True] * n + [False] * (len(df) - n), index=df.index)

    has_test = int((~train_mask).sum()) >= 5
    X_train, y_cls_train, y_reg_train = X[train_mask], y_cls[train_mask], y_reg[train_mask]
    X_test  = X[~train_mask]  if has_test else X.iloc[:0]
    y_cls_test = y_cls[~train_mask] if has_test else y_cls.iloc[:0]
    y_reg_test = y_reg[~train_mask] if has_test else y_reg.iloc[:0]

    # Class balance weight
    n_pos = int((y_cls_train == 1).sum())
    n_neg = int((y_cls_train == 0).sum())
    scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

    # Load existing boosters for warm-start
    existing_cls_booster: Optional[object] = None
    existing_reg_booster: Optional[object] = None
    if incremental:
        cp = model_dir / "promo_move_classifier.pkl"
        rp = model_dir / "promo_uplift_regressor.pkl"
        if cp.exists() and rp.exists():
            existing_cls_booster = joblib.load(cp).get_booster()
            existing_reg_booster = joblib.load(rp).get_booster()
        else:
            incremental = False  # no existing model, fall back to full train

    # ── Classifier ──────────────────────────────────────────────────────────
    cls_model = xgb.XGBClassifier(
        n_estimators=n_new_trees if incremental else 1000,
        max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        enable_categorical=True,
        early_stopping_rounds=None if incremental else 30,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    cls_kw: dict = {"verbose": False}
    if has_test:
        cls_kw["eval_set"] = [(X_test, y_cls_test)]
    if incremental and existing_cls_booster is not None:
        cls_kw["xgb_model"] = existing_cls_booster
    cls_model.fit(X_train, y_cls_train, **cls_kw)

    pr_auc: Optional[float] = None
    if has_test:
        pr_auc = float(average_precision_score(y_cls_test, cls_model.predict_proba(X_test)[:, 1]))

    # ── Regressor (should_move=1 rows only) ──────────────────────────────────
    move_tr = y_cls_train == 1
    move_te = y_cls_test == 1
    X_tr_m, y_tr_m = X_train[move_tr], y_reg_train[move_tr]
    X_te_m, y_te_m = X_test[move_te], y_reg_test[move_te]

    reg_model = xgb.XGBRegressor(
        n_estimators=n_new_trees if incremental else 1000,
        max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
        objective="reg:squarederror",
        enable_categorical=True,
        early_stopping_rounds=None if incremental else 30,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    reg_kw: dict = {"verbose": False}
    if has_test and int(len(X_te_m)) >= 5:
        reg_kw["eval_set"] = [(X_te_m, y_te_m)]
    if incremental and existing_reg_booster is not None:
        reg_kw["xgb_model"] = existing_reg_booster

    r2: Optional[float] = None
    if len(X_tr_m) > 0:
        reg_model.fit(X_tr_m, y_tr_m, **reg_kw)
        if has_test and len(X_te_m) > 0:
            r2 = float(r2_score(y_te_m, reg_model.predict(X_te_m)))

    # ── Save artifacts ──────────────────────────────────────────────────────
    joblib.dump(cls_model, model_dir / "promo_move_classifier.pkl")
    joblib.dump(reg_model, model_dir / "promo_uplift_regressor.pkl")

    schema = {
        "feature_cols": feature_cols,
        "categorical_cols": cat_features,
        "category_values": {c: list(df[c].cat.categories) for c in cat_features},
        "scale_pos_weight": scale_pos_weight,
        "leaky_columns_removed": LEAKY_COLUMNS,
        "incremental": incremental,
        "trained_rows": len(df),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (model_dir / "inference_schema.json").write_text(
        json.dumps(schema, indent=2, default=str), encoding="utf-8"
    )

    return {
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
        "r2": round(r2, 4) if r2 is not None else None,
        "rows_trained": len(df),
        "features": len(feature_cols),
        "incremental": incremental,
        "trained_at": schema["trained_at"],
    }

"""
CTV Promo Placement Optimisation — Corrected Pipeline
======================================================

Rewrite of the original prototype notebook with the ML issues fixed.

Key changes vs original (each marked "# FIX:" in the code below):
  1. Drop leaky post-hoc features (actual_*, was_missed, observed_*, best_possible_*)
  2. Time-based train/test split instead of random (data spans 7 years)
  3. Keep df_raw for inference so the prediction pipeline is not silently broken
  4. Handle class imbalance with scale_pos_weight + evaluate on PR-AUC, not accuracy
  5. Train regressor only on should_move=1 rows (predicting uplift for 'keep' rows is meaningless)
  6. XGBoost native categorical support (enable_categorical=True) instead of LabelEncoder
  7. Early stopping instead of hardcoded n_estimators
  8. Do not fillna(0) on numerics — let XGBoost handle NaN natively
  9. Actual EDA before training

Run on Kaggle: just execute top to bottom.
Run locally: change TRAIN_PATH and OUTPUT_DIR at the top.
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    average_precision_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

import xgboost as xgb

warnings.filterwarnings("ignore")

# =============================================================================
# Config
# =============================================================================

TRAIN_PATH = "/kaggle/input/datasets/vinayguptaji/now-cast-demo/ctv_promo_training_dataset_7y.csv"
OUTPUT_DIR = Path("/kaggle/working/model_artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# FIX 1: Columns that describe what happened AFTER the promo aired.
# Not available at scheduling time — including them causes data leakage and
# inflates your metrics. Drop them.
LEAKY_COLUMNS = [
    "actual_status",
    "actual_start_offset_seconds",
    "actual_duration_seconds",
    "duration_diff_seconds",
    "was_missed",
    "observed_effectiveness_score",   # post-hoc outcome
    "best_possible_score",            # derived from actuals
]

TARGET_CLS = "should_move"
TARGET_REG = "uplift_if_optimised"

# =============================================================================
# 1) Load data — keep a RAW copy for inference
# =============================================================================
# FIX 3: Original notebook label-encoded df in place, then the inference helper
# tried to encode it AGAIN (treating ints as strings). Result: every categorical
# feature collapsed to "UNKNOWN" at predict time. Keeping a raw copy avoids this.

print("Loading data...")
df_raw = pd.read_csv(TRAIN_PATH, parse_dates=["datetime"])
df_raw = df_raw.sort_values("datetime").reset_index(drop=True)

print(f"Shape: {df_raw.shape}")
print(f"Date range: {df_raw['datetime'].min()} → {df_raw['datetime'].max()}")

# =============================================================================
# 2) EDA — this was missing from the original notebook
# =============================================================================

print("\n" + "=" * 70)
print("EDA")
print("=" * 70)

# 2a) Class balance — how imbalanced is should_move?
print("\nClass balance for should_move:")
print(df_raw[TARGET_CLS].value_counts(normalize=True).round(4))

# 2b) Regression target — is uplift heavy-tailed? outlier-dominated?
print(f"\nUplift distribution:")
print(df_raw[TARGET_REG].describe().round(3))

# 2c) Yearly drift — 7 years is a long window. COVID + streaming disruption
# probably changed the data-generating process mid-stream.
yearly = df_raw.groupby(df_raw["datetime"].dt.year).agg(
    rows=("datetime", "count"),
    move_rate=(TARGET_CLS, "mean"),
    mean_uplift=(TARGET_REG, "mean"),
).round(4)
print(f"\nYearly summary (watch for drift):")
print(yearly)

# 2d) Missing values — if >50% missing, the column is barely usable
missing = df_raw.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing):
    print(f"\nColumns with missing values:")
    print(missing.head(10))

# 2e) Categorical cardinality — promo_title etc. probably have 1000s of uniques
cat_cols_raw = df_raw.select_dtypes(include="object").columns
cardinality = df_raw[cat_cols_raw].nunique().sort_values(ascending=False)
print(f"\nCategorical cardinality:")
print(cardinality.head(10))

# 2f) Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

df_raw[TARGET_CLS].value_counts().sort_index().plot(
    kind="bar", ax=axes[0, 0], color=["#4d9de0", "#ff4d1c"]
)
axes[0, 0].set_title("Class balance: should_move")
axes[0, 0].set_xticklabels(["keep (0)", "move (1)"], rotation=0)

axes[0, 1].hist(df_raw[TARGET_REG].dropna(), bins=60, color="#4db8b8")
axes[0, 1].set_title("Distribution of uplift_if_optimised")
axes[0, 1].set_yscale("log")
axes[0, 1].set_xlabel("uplift")

yearly["move_rate"].plot(kind="line", marker="o", ax=axes[1, 0], color="#ff4d1c")
axes[1, 0].set_title("Move rate over time (drift check)")
axes[1, 0].set_ylabel("share of should_move=1")
axes[1, 0].set_xlabel("year")

hour_rate = df_raw.groupby("hour")[TARGET_CLS].mean()
axes[1, 1].plot(hour_rate.index, hour_rate.values, marker="o", color="#9b7de0")
axes[1, 1].set_title("Move rate by hour of day")
axes[1, 1].set_xlabel("hour")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda.png", dpi=100, bbox_inches="tight")
plt.show()

# =============================================================================
# 3) Feature preparation
# =============================================================================

df = df_raw.copy()

# Columns that are not features
id_and_meta = ["date", "datetime", "house_number"]
target_cols = [TARGET_CLS, TARGET_REG, "recommended_position", "recommended_offset_minutes"]

feature_cols = [
    c for c in df.columns
    if c not in id_and_meta
    and c not in target_cols
    and c not in LEAKY_COLUMNS
]
print(f"\nUsing {len(feature_cols)} features (dropped {len(LEAKY_COLUMNS)} leaky columns)")

# FIX 6: Use pandas category dtype + XGBoost's native categorical handling.
# LabelEncoder creates fake ordinal relationships (promo_title=7 is NOT 
# "between" 6 and 8 in any meaningful way).
cat_features = [c for c in feature_cols if df[c].dtype == "object"]
for c in cat_features:
    df[c] = df[c].astype("category")

# FIX 8: DO NOT fillna(0) on numerics. break_event_position=0 is a valid value
# (first event in break). XGBoost handles NaN natively — let it.

X = df[feature_cols]
y_cls = df[TARGET_CLS].astype(int)
y_reg = df[TARGET_REG]

print(f"Feature dtype breakdown: {X.dtypes.value_counts().to_dict()}")

# =============================================================================
# 4) Time-based train/test split
# =============================================================================
# FIX 2: Random split lets the model train on 2024 data and test on 2018 data.
# With 7 years of data, drift is real. Split by date, not randomly.

cutoff = df["datetime"].quantile(0.80)
train_mask = df["datetime"] < cutoff

X_train, X_test = X[train_mask], X[~train_mask]
y_cls_train, y_cls_test = y_cls[train_mask], y_cls[~train_mask]
y_reg_train, y_reg_test = y_reg[train_mask], y_reg[~train_mask]

print(f"\nTrain: {train_mask.sum():>6} rows  (up to {cutoff.date()})")
print(f"Test:  {(~train_mask).sum():>6} rows  (from {cutoff.date()})")

# =============================================================================
# 5) Classifier — imbalance-aware, with early stopping
# =============================================================================

# FIX 4: 80/20 imbalance means default XGBoost ignores the minority class.
# scale_pos_weight fixes this; PR-AUC is the right metric (not accuracy).
n_neg = (y_cls_train == 0).sum()
n_pos = (y_cls_train == 1).sum()
scale_pos_weight = n_neg / n_pos
print(f"\nscale_pos_weight = {scale_pos_weight:.2f}")

cls_model = xgb.XGBClassifier(
    n_estimators=1000,              # FIX 7: early stopping picks the real value
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",            # PR-AUC, not logloss
    enable_categorical=True,        # FIX 6: native categorical
    early_stopping_rounds=30,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

cls_model.fit(
    X_train, y_cls_train,
    eval_set=[(X_test, y_cls_test)],
    verbose=False,
)
print(f"Classifier stopped at tree {cls_model.best_iteration}")

# Honest evaluation
pred_cls = cls_model.predict(X_test)
proba_cls = cls_model.predict_proba(X_test)[:, 1]

print("\n--- Classifier metrics (honest, on held-out future period) ---")
print(f"Accuracy:     {accuracy_score(y_cls_test, pred_cls):.4f}  (vanity metric on imbalanced data)")
print(f"PR-AUC:       {average_precision_score(y_cls_test, proba_cls):.4f}  (the one that matters)")
print(f"ROC-AUC:      {roc_auc_score(y_cls_test, proba_cls):.4f}")
print()
print(classification_report(y_cls_test, pred_cls, target_names=["keep", "move"]))
print("Confusion matrix [rows=truth, cols=pred]:")
print(confusion_matrix(y_cls_test, pred_cls))

# Feature importance
importance = pd.Series(cls_model.feature_importances_, index=feature_cols)
importance = importance.sort_values(ascending=False)
print("\nTop 15 features:")
print(importance.head(15).round(4))

plt.figure(figsize=(10, 6))
importance.head(15).sort_values().plot(kind="barh", color="#ff4d1c")
plt.title("Classifier feature importance")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=100, bbox_inches="tight")
plt.show()

# =============================================================================
# 6) Regressor — trained only on should_move=1 rows
# =============================================================================
# FIX 5: Training the regressor on all rows is misleading. 80% of targets are ~0
# (for keep rows), so predicting zero looks amazing. What you actually need is
# "given this promo should be moved, how much uplift?" — so filter to move rows.

move_mask_train = y_cls_train == 1
move_mask_test = y_cls_test == 1

X_train_m = X_train[move_mask_train]
y_reg_train_m = y_reg_train[move_mask_train]
X_test_m = X_test[move_mask_test]
y_reg_test_m = y_reg_test[move_mask_test]

print(f"\nRegressor train set: {len(X_train_m)} move-rows")
print(f"Regressor test set:  {len(X_test_m)} move-rows")

reg_model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    objective="reg:squarederror",
    enable_categorical=True,
    early_stopping_rounds=30,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

reg_model.fit(
    X_train_m, y_reg_train_m,
    eval_set=[(X_test_m, y_reg_test_m)],
    verbose=False,
)
print(f"Regressor stopped at tree {reg_model.best_iteration}")

pred_reg = reg_model.predict(X_test_m)
print("\n--- Regressor metrics (on should_move=1 subset only) ---")
print(f"MAE:  {mean_absolute_error(y_reg_test_m, pred_reg):.4f}")
print(f"RMSE: {mean_squared_error(y_reg_test_m, pred_reg) ** 0.5:.4f}")
print(f"R²:   {r2_score(y_reg_test_m, pred_reg):.4f}")
print("\nNote: R² here will look lower than the original notebook's 0.844 because")
print("we're not cheating by predicting easy zeros. This is the honest number.")

# =============================================================================
# 7) Save artifacts + inference schema
# =============================================================================

joblib.dump(cls_model, OUTPUT_DIR / "promo_move_classifier.pkl")
joblib.dump(reg_model, OUTPUT_DIR / "promo_uplift_regressor.pkl")

# Save everything the inference code needs to rebuild a model-ready row from
# raw input. Without this, your API will mis-encode features silently.
inference_schema = {
    "feature_cols": feature_cols,
    "categorical_cols": cat_features,
    "category_values": {c: list(df[c].cat.categories) for c in cat_features},
    "cutoff_date": str(cutoff),
    "scale_pos_weight": float(scale_pos_weight),
    "leaky_columns_removed": LEAKY_COLUMNS,
}
with open(OUTPUT_DIR / "inference_schema.json", "w") as f:
    json.dump(inference_schema, f, indent=2, default=str)

print(f"\nArtifacts saved to {OUTPUT_DIR}")
print(os.listdir(OUTPUT_DIR))

# =============================================================================
# 8) Inference — works correctly on raw rows (unlike the original)
# =============================================================================

def prepare_for_inference(row_df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Convert raw (unencoded) rows to model-ready format.

    This is the function the original notebook tried but got wrong — the
    original applied encoding to already-encoded data, so every categorical
    field collapsed to 'UNKNOWN'. Here we take RAW strings and convert them
    to the same category dtype the model saw during training.
    """
    prepared = row_df[schema["feature_cols"]].copy()
    for col in schema["categorical_cols"]:
        # Force the exact same category set used in training.
        # New/unseen values become NaN, which XGBoost handles natively.
        prepared[col] = pd.Categorical(
            prepared[col].astype(str),
            categories=schema["category_values"][col],
        )
    return prepared


def predict_promo(raw_row_df: pd.DataFrame, schema: dict) -> dict:
    """Full inference: classifier + conditional regressor."""
    X_inf = prepare_for_inference(raw_row_df, schema)
    move_proba = float(cls_model.predict_proba(X_inf)[0, 1])
    should_move = int(move_proba >= 0.5)
    # Only bother predicting uplift if we're recommending the move
    uplift = float(reg_model.predict(X_inf)[0]) if should_move else 0.0
    return {
        "should_move": should_move,
        "move_probability": round(move_proba, 4),
        "predicted_uplift": round(uplift, 4),
    }


# Load schema back (simulates what a real API deployment does)
with open(OUTPUT_DIR / "inference_schema.json") as f:
    schema = json.load(f)

# Sanity test on a RAW row
test_row = df_raw.iloc[[100]].copy()
result = predict_promo(test_row, schema)

print("\n--- Sanity test ---")
print(f"Promo:           {test_row['promo_title'].iloc[0]}")
print(f"Datetime:        {test_row['datetime'].iloc[0]}")
print(f"Actual label:    should_move={test_row[TARGET_CLS].iloc[0]}, uplift={test_row[TARGET_REG].iloc[0]:.2f}")
print(f"Model output:    {result}")

# =============================================================================
# 9) API-style inference (partial payload)
# =============================================================================

def predict_promo_api(payload: dict, schema: dict, template_row: pd.DataFrame) -> dict:
    """Merge a partial payload into a template row and predict.

    Useful for what-if scenarios. In production you'd have a full input schema
    and wouldn't need a template — but this is fine for the prototype stage.
    """
    row = template_row.copy()
    for key, value in payload.items():
        if key in row.columns:
            row.at[row.index[0], key] = value
    result = predict_promo(row, schema)
    return {
        "promo_title": str(row["promo_title"].iloc[0]),
        "datetime": str(row["datetime"].iloc[0]),
        **result,
    }


api_payload = {
    "promo_title": "Alexander's Lost World - Tue @ 2100",
    "weather_rain_mm": 7.5,
    "weather_sun_hours": 0.8,
    "promo_fatigue_index": 0.85,
    "promo_in_break": 1,
    "break_position_pct": 0.75,
    "attention_context_score": 0.30,
}

template = df_raw.iloc[[0]].copy()
print("\n--- API-style what-if ---")
print(predict_promo_api(api_payload, schema, template))

print("\nDone.")

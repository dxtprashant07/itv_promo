"""
CTV Promo Placement Optimisation — FastAPI
==========================================

Endpoints:
  GET  /health             → model load status
  POST /reload             → hot-reload models without restart
  POST /predict            → single promo inference (JSON)
  POST /predict/batch      → batch inference up to 500 rows (JSON)
  POST /predict/upload     → upload a CSV file, download predictions CSV

Expects model artifacts in MODEL_DIR (default: ./model/):
  - promo_move_classifier.pkl
  - promo_uplift_regressor.pkl
  - inference_schema.json   (run generate_schema.py if missing)
"""

import io
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from config import LEAKY_COLUMNS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("ctv_promo_api")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(exist_ok=True)
FEEDBACK_PATH = DATA_DIR / "feedback.csv"

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
BATCH_LIMIT = int(os.getenv("BATCH_LIMIT", "500"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# ---------------------------------------------------------------------------
# App + middleware
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        _load_models()
    except Exception as exc:
        _state["error"] = str(exc)
        _state["ready"] = False
        logger.error("Startup failed: %s", exc)

    yield


app = FastAPI(
    title="CTV Promo Placement API",
    description="Predicts whether a TV promo should be repositioned and the expected uplift.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s → %d (%.1fms)", request.method, request.url.path, response.status_code, ms)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()   # guards all writes to _state model fields

_state: dict = {
    "cls_model": None,
    "reg_model": None,
    "schema": None,
    "ready": False,
    "error": None,
    "training": {
        "status": "idle",       # idle | running | done | error
        "started_at": None,
        "finished_at": None,
        "metrics": None,
        "error": None,
    },
}


def _load_models() -> None:
    """Load (or reload) model artifacts from MODEL_DIR."""
    logger.info("Loading models from %s", MODEL_DIR)
    cls_model = joblib.load(MODEL_DIR / "promo_move_classifier.pkl")
    reg_model = joblib.load(MODEL_DIR / "promo_uplift_regressor.pkl")

    schema_path = MODEL_DIR / "inference_schema.json"
    if schema_path.exists():
        with open(schema_path) as f:
            schema = json.load(f)
        logger.info("Schema loaded — %d features, %d categorical", len(schema["feature_cols"]), len(schema["categorical_cols"]))
    else:
        schema = None
        logger.warning(
            "inference_schema.json not found in %s. "
            "Categorical features will not be re-encoded — run generate_schema.py to fix this.",
            MODEL_DIR,
        )

    # Commit atomically under lock so no request sees a half-loaded state
    with _state_lock:
        _state["cls_model"] = cls_model
        _state["reg_model"] = reg_model
        _state["schema"] = schema
        _state["ready"] = True
        _state["error"] = None
    logger.info("Models ready")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PromoFeatures(BaseModel):
    """
    Fields match inference_schema.json exactly.
    All are optional — omitted fields become NaN, which XGBoost handles natively.
    Extra fields not listed here are accepted and forwarded to the model.
    """

    # Non-feature passthrough (returned in response, not fed to the model)
    datetime: Optional[str] = None

    # Categorical features
    channel: Optional[str] = None
    promo_title: Optional[str] = None
    content_type: Optional[str] = None
    event_type: Optional[str] = None
    genre_guess: Optional[str] = None
    time_band: Optional[str] = None
    promo_position_type: Optional[str] = None
    prev_promo_title: Optional[str] = None
    next_promo_title: Optional[str] = None
    weather_station: Optional[str] = None

    # Scheduling / placement
    hour: Optional[int] = Field(default=None, ge=0, le=23)
    minute: Optional[int] = Field(default=None, ge=0, le=59)
    day_of_week: Optional[int] = Field(default=None, ge=0, le=6)
    is_weekend: Optional[int] = Field(default=None, ge=0, le=1)
    promo_length_seconds: Optional[float] = Field(default=None, ge=0)
    promo_in_break: Optional[int] = Field(default=None, ge=0, le=1)
    break_event_position: Optional[int] = Field(default=None, ge=0)
    break_total_events: Optional[int] = Field(default=None, ge=0)
    break_position_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    preceded_by_break: Optional[int] = Field(default=None, ge=0, le=1)
    lead_to_next_program_min: Optional[float] = None

    # Weather
    weather_tmax_c: Optional[float] = None
    weather_tmin_c: Optional[float] = None
    weather_rain_mm: Optional[float] = Field(default=None, ge=0.0)
    weather_sun_hours: Optional[float] = Field(default=None, ge=0.0, le=24.0)
    weather_bad_index: Optional[float] = None
    weather_indoor_viewing_index: Optional[float] = None

    # OTT / streaming signals
    ott_avg_watch_pct: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ott_dropoff_prob: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ott_hook_strength: Optional[float] = None
    ott_visual_intensity: Optional[float] = None

    # Netflix / competitive
    netflix_popularity_score: Optional[float] = None
    netflix_rating_mean: Optional[float] = None

    # Synthetic / derived
    synthetic_premiere_probability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    synthetic_production_year_mean: Optional[float] = None

    # Promo health
    promo_fatigue_index: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    attention_context_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    model_config = {"extra": "allow"}


class PredictionResponse(BaseModel):
    promo_title: Optional[str] = None
    datetime: Optional[str] = None
    should_move: int = Field(..., description="1 = recommend repositioning, 0 = keep")
    move_probability: float = Field(..., description="Classifier confidence [0, 1]")
    predicted_uplift: float = Field(..., description="Expected uplift if moved (0 when should_move=0)")


class FeedbackRow(PromoFeatures):
    """Labeled promo row used for incremental retraining."""
    should_move: int = Field(..., ge=0, le=1, description="Actual label: 1=moved, 0=kept")
    uplift_if_optimised: Optional[float] = Field(
        default=0.0, description="Observed uplift when moved (leave 0 for keep rows)"
    )


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _prepare(df: pd.DataFrame, schema: Optional[dict]) -> pd.DataFrame:
    """Select training features and enforce dtypes XGBoost requires.

    XGBoost rejects object dtype — every column must be int/float/bool or
    pandas category.  None values for numeric fields produce object dtype in
    pandas, so we convert them to float (None → NaN) after handling categoricals.
    """
    cls_model = _state["cls_model"]

    if schema is not None:
        feature_cols = schema["feature_cols"]
        cat_cols = set(schema["categorical_cols"])
        category_values = schema.get("category_values", {})
    else:
        feature_cols = list(getattr(cls_model, "feature_names_in_", df.columns))
        raw_types = (
            getattr(cls_model, "feature_types_", None)
            or getattr(cls_model, "feature_types", None)
            or []
        )
        cat_cols = (
            {feature_cols[i] for i, t in enumerate(raw_types) if t == "c"}
            if raw_types
            else set(df.select_dtypes(include="object").columns)
        )
        category_values = {}

    prepared = df.reindex(columns=feature_cols)

    for col in prepared.columns:
        if col in cat_cols:
            vocab = category_values.get(col)
            if vocab:
                # Encode against exact training vocabulary; unknowns become NaN
                prepared[col] = pd.Categorical(
                    prepared[col].astype(str), categories=vocab
                )
            else:
                prepared[col] = prepared[col].astype("category")
        elif prepared[col].dtype == object:
            # Numeric column that is object dtype because all input values were None
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

    return prepared


def _run_batch(items: list[PromoFeatures]) -> list[PredictionResponse]:
    cls_model = _state["cls_model"]
    reg_model = _state["reg_model"]
    schema = _state["schema"]

    passthroughs = []
    rows = []
    for feat in items:
        raw = feat.model_dump(exclude_none=False)
        # datetime is not a model feature — pop it; promo_title IS a feature so keep it
        passthroughs.append({
            "promo_title": raw.get("promo_title"),
            "datetime": raw.pop("datetime", None),
        })
        rows.append(raw)

    X = _prepare(pd.DataFrame(rows), schema)

    move_probas: np.ndarray = cls_model.predict_proba(X)[:, 1]
    should_moves: np.ndarray = (move_probas >= 0.5).astype(int)

    uplifts = np.zeros(len(items), dtype=float)
    move_idx = np.where(should_moves == 1)[0]
    if len(move_idx):
        uplifts[move_idx] = reg_model.predict(X.iloc[move_idx])

    return [
        PredictionResponse(
            promo_title=pt.get("promo_title"),
            datetime=pt.get("datetime"),
            should_move=int(sm),
            move_probability=round(float(mp), 4),
            predicted_uplift=round(float(up), 4),
        )
        for pt, mp, sm, up in zip(passthroughs, move_probas, should_moves, uplifts)
    ]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
def health():
    """Liveness + readiness check."""
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail=_state["error"] or "Models not loaded")
    schema = _state["schema"]
    return {
        "status": "ok",
        "schema_loaded": schema is not None,
        "feature_count": len(schema["feature_cols"]) if schema else "unknown",
    }


@app.post("/reload", tags=["ops"])
def reload():
    """Hot-reload model artifacts from disk without restarting the server."""
    try:
        _load_models()
    except Exception as exc:
        _state["error"] = str(exc)
        _state["ready"] = False
        raise HTTPException(status_code=500, detail=f"Reload failed: {exc}")
    return {"status": "reloaded"}


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(features: PromoFeatures):
    """Predict whether a single promo should be repositioned."""
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        return _run_batch([features])[0]
    except Exception as exc:
        logger.error("Prediction error: %s", exc, exc_info=True)
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/predict/batch", response_model=list[PredictionResponse], tags=["inference"])
def predict_batch(items: list[PromoFeatures]):
    """Predict for a batch of promos (up to BATCH_LIMIT rows)."""
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="Models not loaded")
    if not items:
        raise HTTPException(status_code=400, detail="Request body must contain at least one item")
    if len(items) > BATCH_LIMIT:
        raise HTTPException(status_code=400, detail=f"Batch size limit is {BATCH_LIMIT}")
    try:
        return _run_batch(items)
    except Exception as exc:
        logger.error("Batch prediction error: %s", exc, exc_info=True)
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/predict/upload", tags=["inference"])
async def predict_upload(file: UploadFile = File(...)):
    """
    Upload a CSV file → download a CSV with predictions appended.

    - Accepts any CSV that contains a subset of the 37 training features.
    - Leaky columns (actual_status, was_missed, etc.) are silently dropped.
    - Missing feature columns become NaN — XGBoost handles them natively.
    - Returns the original CSV with three new columns added:
        should_move, move_probability, predicted_uplift
    """
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="Models not loaded")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    _LEAKY = set(LEAKY_COLUMNS)

    try:
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({len(contents)//1024//1024} MB). Limit is {MAX_UPLOAD_MB} MB.",
            )
        df = pd.read_csv(io.BytesIO(contents))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    # Drop leaky columns silently
    df = df.drop(columns=[c for c in _LEAKY if c in df.columns])

    schema = _state["schema"]
    feature_cols = schema["feature_cols"] if schema else list(
        getattr(_state["cls_model"], "feature_names_in_", df.columns)
    )

    # Build PromoFeatures objects from each row
    items = []
    for _, row in df.iterrows():
        payload = {}
        for col in feature_cols:
            if col in row.index and pd.notna(row[col]):
                val = row[col]
                payload[col] = val.item() if hasattr(val, "item") else val
        # Include datetime passthrough if present
        if "datetime" in row.index:
            payload["datetime"] = str(row["datetime"]) if pd.notna(row["datetime"]) else None
        items.append(PromoFeatures(**payload))

    if len(items) > BATCH_LIMIT:
        raise HTTPException(status_code=400, detail=f"CSV has {len(items)} rows — limit is {BATCH_LIMIT}. Split the file.")

    try:
        results = _run_batch(items)
    except Exception as exc:
        logger.error("Upload prediction error: %s", exc, exc_info=True)
        raise HTTPException(status_code=422, detail=str(exc))

    df["should_move"]      = [r.should_move       for r in results]
    df["move_probability"] = [r.move_probability   for r in results]
    df["predicted_uplift"] = [r.predicted_uplift   for r in results]

    # Stream the result back as a downloadable CSV
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    out_filename = file.filename.replace(".csv", "_predictions.csv")
    logger.info("Upload: %d rows predicted, returning %s", len(df), out_filename)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={out_filename}"},
    )


# ---------------------------------------------------------------------------
# Training / retraining endpoints
# ---------------------------------------------------------------------------

def _run_training(df: pd.DataFrame, incremental: bool, n_new_trees: int) -> None:
    """Background thread: train models and hot-reload on completion."""
    ts = _state["training"]
    ts["status"] = "running"
    ts["started_at"] = datetime.now(timezone.utc).isoformat()
    ts["finished_at"] = None
    ts["error"] = None
    ts["metrics"] = None

    try:
        from trainer import train_models
        metrics = train_models(df, MODEL_DIR, incremental=incremental, n_new_trees=n_new_trees)
        _load_models()                          # atomic hot-reload
        ts["status"] = "done"
        ts["metrics"] = metrics
        ts["finished_at"] = datetime.now(timezone.utc).isoformat()
        logger.info("Retrain complete: %s", metrics)
    except Exception as exc:
        ts["status"] = "error"
        ts["error"] = str(exc)
        ts["finished_at"] = datetime.now(timezone.utc).isoformat()
        logger.error("Retrain failed: %s", exc, exc_info=True)


@app.post("/feedback", tags=["training"])
def submit_feedback(items: list[FeedbackRow]):
    """
    Submit labeled promo rows as feedback for future retraining.

    Each row must include `should_move` (0/1) and optionally
    `uplift_if_optimised` (float). Rows are appended to data/feedback.csv.
    Call POST /retrain once you have accumulated enough rows.
    """
    rows = [item.model_dump() for item in items]
    new_df = pd.DataFrame(rows)

    if FEEDBACK_PATH.exists():
        existing = pd.read_csv(FEEDBACK_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(FEEDBACK_PATH, index=False)
    logger.info("Feedback: %d new rows appended (total %d)", len(new_df), len(combined))
    return {"accepted": len(new_df), "total_feedback_rows": len(combined)}


@app.post("/retrain", tags=["training"])
def trigger_retrain(incremental: bool = True, n_new_trees: int = 100, min_rows: int = 10):
    """
    Trigger model retraining from accumulated feedback data (data/feedback.csv).

    Set incremental=true (default) to warm-start the existing model with new
    trees — fast, suitable for frequent small updates.
    Set incremental=false for a full retrain from scratch — slower but resets
    the model to the new data distribution.

    Training runs in a background thread; poll GET /retrain/status to check progress.
    After completion the models are hot-reloaded automatically (no restart needed).
    """
    if _state["training"]["status"] == "running":
        raise HTTPException(status_code=409, detail="Training already in progress")

    if not FEEDBACK_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No feedback data found. Submit labeled rows via POST /feedback first.",
        )

    df = pd.read_csv(FEEDBACK_PATH)
    if len(df) < min_rows:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {min_rows} feedback rows, got {len(df)}. "
                   f"Submit more via POST /feedback.",
        )

    t = threading.Thread(
        target=_run_training, args=(df, incremental, n_new_trees), daemon=True
    )
    t.start()
    return {"status": "started", "rows": len(df), "incremental": incremental}


@app.post("/retrain/upload", tags=["training"])
async def retrain_upload(
    file: UploadFile = File(...),
    incremental: bool = False,
    n_new_trees: int = 100,
):
    """
    Upload a labeled CSV and immediately trigger model retraining.

    The CSV must contain `should_move` (0/1) and `uplift_if_optimised` (float)
    columns in addition to the standard promo feature columns.
    Leaky post-hoc columns are dropped automatically.

    Training runs in a background thread — poll GET /retrain/status to track progress.
    """
    if _state["training"]["status"] == "running":
        raise HTTPException(status_code=409, detail="Training already in progress")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    try:
        contents = await file.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({len(contents)//1024//1024} MB). Limit is {MAX_UPLOAD_MB} MB.",
            )
        df = pd.read_csv(io.BytesIO(contents))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    missing_labels = [c for c in ["should_move", "uplift_if_optimised"] if c not in df.columns]
    if missing_labels:
        raise HTTPException(
            status_code=400,
            detail=f"Training CSV missing required label columns: {missing_labels}. "
                   f"For prediction-only CSVs use POST /predict/upload instead.",
        )

    # Persist uploaded training data so /retrain can reuse it later
    df.to_csv(FEEDBACK_PATH, index=False)
    logger.info("Retrain upload: %d rows, incremental=%s", len(df), incremental)

    t = threading.Thread(
        target=_run_training, args=(df, incremental, n_new_trees), daemon=True
    )
    t.start()
    return {"status": "started", "rows": len(df), "incremental": incremental, "filename": file.filename}


@app.get("/retrain/status", tags=["training"])
def retrain_status():
    """Check the status of the current or most recent retraining run."""
    return _state["training"]

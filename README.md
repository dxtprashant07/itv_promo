# Ancast Nowcasting — CTV Promo Placement Optimisation

An end-to-end ML system that predicts whether a TV promo should be repositioned in the schedule and estimates the expected effectiveness uplift if moved.

Built with XGBoost, FastAPI, and Streamlit. Enriches schedule data with live weather (Open-Meteo) and content metadata (TMDB) automatically.

---

## What it does

For each promo slot in an upcoming schedule the system outputs:

| Output | Description |
|---|---|
| `should_move` | 1 = recommend repositioning, 0 = keep as-is |
| `move_probability` | Model confidence (0–1) |
| `predicted_uplift` | Expected effectiveness gain if moved (float) |

Decisions are driven by 37 features across four groups: schedule placement, weather context, content signals, and audience attention.

---

## Project structure

```
ITV_PROMO/
├── main.py                  # FastAPI backend — prediction + retraining endpoints
├── streamlit_app.py         # Streamlit dashboard UI
├── trainer.py               # Core XGBoost training logic (shared by CLI + API)
├── train_local.py           # CLI for training from a local CSV
├── fetch_live_data.py       # Live enrichment: weather (Open-Meteo) + TMDB
├── predict_from_csv.py      # Batch-predict an enriched CSV via the API
├── config.py                # Single source of truth: feature list, targets, constants
├── generate_schema.py       # Recover inference_schema.json from saved pkl files
├── ctv_promo_pipeline.py    # Original Kaggle training pipeline (reference only)
├── requirements.txt
├── .env.example
├── docker-compose.yml
├── Dockerfile
└── tests/
    ├── test_api.py          # FastAPI endpoint tests (33 tests)
    └── test_trainer.py      # End-to-end trainer tests (10 tests)
```

---

## Quick start

### 1. Install dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

### 2. Set up environment

```powershell
Copy-Item .env.example .env
```

Open `.env` and add your TMDB key (optional but recommended):

```
MODEL_DIR=model
CORS_ORIGINS=*
BATCH_LIMIT=500
PORT=8000
TMDB_API_KEY=your_key_here
```

Get a free TMDB key at [themoviedb.org/settings/api](https://www.themoviedb.org/settings/api).

### 3. Start the API

```powershell
.\.venv\Scripts\uvicorn main:app --reload
```

API live at `http://localhost:8000` — Swagger docs at `http://localhost:8000/docs`

### 4. Start the dashboard

```powershell
.\.venv\Scripts\streamlit run streamlit_app.py
```

Dashboard at `http://localhost:8501`

---

## Predicting from real-time data

### Minimal schedule CSV

Create a CSV from your playout system. Minimum required columns:

```csv
promo_title,datetime
Alexander's Lost World,2024-11-20 21:00
Europe's Last Warrior Kings,2024-11-20 21:20
Queen Victoria's Letters,2024-11-20 22:00
```

Recommended additional columns for better accuracy:

```csv
promo_title,datetime,channel,promo_length_seconds,promo_in_break,break_event_position,break_total_events,weather_station
Alexander's Lost World,2024-11-20 21:00,ITV1,20,1,2,4,heathrow
```

### Enrich + predict in one command

```powershell
# Weather only (no API key needed):
.\.venv\Scripts\python fetch_live_data.py --input my_schedule.csv

# Weather + content metadata (TMDB key required):
.\.venv\Scripts\python fetch_live_data.py --input my_schedule.csv --tmdb-key YOUR_KEY

# Skip the prediction step, just enrich:
.\.venv\Scripts\python fetch_live_data.py --input my_schedule.csv --no-predict
```

Output files:
- `my_schedule_enriched.csv` — all 37 features populated
- `my_schedule_enriched_predictions.csv` — enriched + model predictions

### What gets fetched automatically

| Feature group | Source | API key |
|---|---|---|
| Temperature, rain, sunshine | Open-Meteo | None — free |
| Content popularity & ratings | TMDB | Free (register once) |
| Promo fatigue index | Computed from schedule | — |
| Attention context score | Computed from time + break position | — |
| Prev / next promo title | Computed from schedule neighbours | — |

### Predict from an already-enriched CSV

```powershell
.\.venv\Scripts\python predict_from_csv.py --input my_schedule_enriched.csv
```

### Direct API call

```powershell
# Single prediction
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{"promo_title":"Alexander Lost World","hour":21,"promo_fatigue_index":0.3,"weather_rain_mm":2.5}'
```

```python
# Python batch prediction
import requests

rows = [
    {"promo_title": "Show A", "hour": 21, "promo_fatigue_index": 0.3},
    {"promo_title": "Show B", "hour": 14, "promo_fatigue_index": 0.7},
]
resp = requests.post("http://localhost:8000/predict/batch", json=rows)
print(resp.json())
```

---

## Training the model

### From a labeled CSV (CLI)

Your CSV must include label columns `should_move` (0 or 1) and `uplift_if_optimised` (float). All other columns are treated as features.

```powershell
# Full retrain from scratch:
.\.venv\Scripts\python train_local.py --data your_labeled_data.csv

# Incremental — add 100 trees to the existing model (fast):
.\.venv\Scripts\python train_local.py --data new_batch.csv --incremental

# Incremental with custom tree count:
.\.venv\Scripts\python train_local.py --data new_batch.csv --incremental --new-trees 50

# Custom model output directory:
.\.venv\Scripts\python train_local.py --data your_data.csv --model-dir /path/to/model
```

### Via the API (while the server is running)

```powershell
# Upload a labeled CSV and retrain in the background:
curl -X POST "http://localhost:8000/retrain/upload?incremental=true&n_new_trees=100" `
  -F "file=@labeled_data.csv"

# Check training progress:
curl http://localhost:8000/retrain/status

# Submit individual labeled rows for future retraining:
curl -X POST http://localhost:8000/feedback `
  -H "Content-Type: application/json" `
  -d '[{"promo_title":"Show A","hour":21,"should_move":1,"uplift_if_optimised":0.35}]'

# Trigger retrain from accumulated feedback:
curl -X POST "http://localhost:8000/retrain?incremental=true"
```

After training completes the model is hot-reloaded automatically — no server restart needed.

### Via the Streamlit UI

Open the **"🔁 Retrain model with labeled data"** section, upload your CSV, choose Incremental or Full retrain, and click Start.

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness + readiness check |
| POST | `/reload` | Hot-reload model artifacts from disk |
| POST | `/predict` | Single promo inference (JSON) |
| POST | `/predict/batch` | Batch inference up to 500 rows (JSON) |
| POST | `/predict/upload` | Upload CSV → download predictions CSV |
| POST | `/feedback` | Submit labeled rows for future retraining |
| POST | `/retrain` | Trigger retraining from accumulated feedback |
| POST | `/retrain/upload` | Upload labeled CSV and start retraining |
| GET | `/retrain/status` | Check status of current/last training run |

Full interactive docs: `http://localhost:8000/docs`

---

## The 37 model features

| Group | Features |
|---|---|
| Identity | `channel`, `promo_title`, `content_type`, `event_type`, `genre_guess` |
| Schedule | `promo_length_seconds`, `hour`, `minute`, `day_of_week`, `is_weekend`, `time_band` |
| Break placement | `promo_position_type`, `promo_in_break`, `break_event_position`, `break_total_events`, `break_position_pct`, `preceded_by_break`, `lead_to_next_program_min` |
| Neighbours | `prev_promo_title`, `next_promo_title` |
| Weather | `weather_station`, `weather_tmax_c`, `weather_tmin_c`, `weather_rain_mm`, `weather_sun_hours`, `weather_bad_index`, `weather_indoor_viewing_index` |
| OTT signals | `ott_avg_watch_pct`, `ott_dropoff_prob`, `ott_hook_strength`, `ott_visual_intensity` |
| Competitive | `netflix_popularity_score`, `netflix_rating_mean` |
| Derived | `synthetic_premiere_probability`, `synthetic_production_year_mean`, `promo_fatigue_index`, `attention_context_score` |

Missing features are handled natively by XGBoost (treated as NaN) — you don't need all 37 to get a prediction.

---

## Running tests

```powershell
.\.venv\Scripts\python -m pytest tests/ -v
```

43 tests covering:
- All API endpoints including `/feedback`, `/retrain`, `/retrain/upload`, `/retrain/status`
- End-to-end trainer: full train, incremental warm-start, artifact validation, error handling

---

## Docker

```powershell
# Start the API container:
docker-compose up --build

# With a custom port:
$env:PORT = "9000"
docker-compose up
```

The Streamlit dashboard is not included in the Docker image — run it locally or deploy to Streamlit Cloud.

---

## Deploying to Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set **Main file path** to `streamlit_app.py`
4. Add secrets in the Streamlit Cloud dashboard:
   ```toml
   TMDB_API_KEY = "your_key"
   API_URL = "https://your-api-host.com"
   ```

The API must be deployed separately (e.g. Railway, Render, or Docker on a VPS) and the `API_URL` environment variable set to point to it. If the API is unreachable, the dashboard falls back to local model inference automatically.

---

## Model artifacts

After training, three files are saved to `model/` (or your `--model-dir`):

| File | Description |
|---|---|
| `promo_move_classifier.pkl` | XGBoost classifier — predicts `should_move` |
| `promo_uplift_regressor.pkl` | XGBoost regressor — predicts `uplift_if_optimised` |
| `inference_schema.json` | Feature list, categorical vocabularies, training metadata |

If `inference_schema.json` is missing (e.g. after copying pkl files without it), regenerate it:

```powershell
.\.venv\Scripts\python generate_schema.py model/
```

---

## Supported UK weather stations

`heathrow` · `manchester` · `birmingham` · `edinburgh` · `cardiff` · `belfast` · `leeds` · `exeter` · `cambridge` · `norwich` · `newcastle` · `hurn`

Set the `weather_station` column in your schedule CSV to the nearest station. Defaults to `heathrow` if not specified.

---

## Requirements

- Python 3.11+
- See `requirements.txt` for package versions

| Package | Purpose |
|---|---|
| `xgboost>=2.0.0` | Core ML model |
| `fastapi>=0.111.0` | Prediction API |
| `uvicorn` | ASGI server |
| `streamlit>=1.35.0` | Dashboard UI |
| `pandas>=2.2.0` | Data processing |
| `scikit-learn>=1.4.0` | Evaluation metrics |
| `requests>=2.32.0` | Open-Meteo + TMDB API calls |

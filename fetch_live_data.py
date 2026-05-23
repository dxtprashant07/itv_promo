"""
fetch_live_data.py
==================
Enrich a minimal promo schedule CSV with real-time data from free APIs,
producing a fully-populated row ready for the prediction model.

Data sources
------------
1. Open-Meteo (https://open-meteo.com)  — FREE, no API key required
   Provides all 7 weather columns: tmax, tmin, rain, sun hours,
   plus computed bad_index and indoor_viewing_index.
   Works for historical dates, today, and up to 16 days ahead.

2. TMDB – The Movie Database (https://www.themoviedb.org/settings/api)
   FREE account gives an API key immediately.
   Provides: netflix_popularity_score, netflix_rating_mean,
   synthetic_premiere_probability, synthetic_production_year_mean,
   ott_hook_strength, ott_visual_intensity.
   Set TMDB_API_KEY env var or pass --tmdb-key.

3. Computed from the schedule itself
   promo_fatigue_index  — share of last-7-days slots occupied by this title
   attention_context_score — derived from time band + break position + weekend
   time_band, is_weekend, day_of_week, break_position_pct, preceded_by_break,
   genre_guess (keyword-inferred), lead_to_next_program_min,
   prev_promo_title, next_promo_title

Minimum required input columns
-------------------------------
  promo_title  — name of the promo (string)
  datetime     — ISO datetime of the scheduled slot, e.g. "2024-11-19 21:00"

Recommended additions
  channel, content_type, event_type, promo_length_seconds
  promo_in_break, break_event_position, break_total_events
  weather_station  — UK station name (default: heathrow)

Usage
-----
  python fetch_live_data.py --input schedule.csv
  python fetch_live_data.py --input schedule.csv --tmdb-key YOUR_KEY --output enriched.csv
  python fetch_live_data.py --input schedule.csv --no-predict   # just enrich, don't call API
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from config import FEATURE_COLS

# ---------------------------------------------------------------------------
# UK weather station → (latitude, longitude)
# ---------------------------------------------------------------------------

UK_STATIONS: dict[str, tuple[float, float]] = {
    "heathrow":   (51.48, -0.45),   # London
    "hurn":       (50.82, -1.83),   # Bournemouth
    "manchester": (53.35, -2.27),
    "birmingham": (52.45, -1.73),
    "edinburgh":  (55.95, -3.35),
    "cardiff":    (51.40, -3.35),
    "belfast":    (54.65, -6.22),
    "leeds":      (53.87, -1.66),
    "exeter":     (50.73, -3.41),
    "cambridge":  (52.20,  0.18),
    "norwich":    (52.62,  1.22),
    "newcastle":  (54.98, -1.62),
}

DEFAULT_STATION = "heathrow"

# ---------------------------------------------------------------------------
# Default OTT/Netflix values used when TMDB is unavailable
# Tune these to match your platform's baseline metrics.
# ---------------------------------------------------------------------------

OTT_DEFAULTS: dict[str, float] = {
    "ott_avg_watch_pct":              0.65,
    "ott_dropoff_prob":               0.25,
    "ott_hook_strength":              0.60,
    "ott_visual_intensity":           0.50,
    "netflix_popularity_score":       40.0,
    "netflix_rating_mean":            6.5,
    "synthetic_premiere_probability": 0.30,
    "synthetic_production_year_mean": 2019.0,
}

# ---------------------------------------------------------------------------
# Genre inference from title keywords
# ---------------------------------------------------------------------------

_GENRE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("history_doc", ["history", "historical", "ancient", "war", "empire", "warrior", "world war"]),
    ("drama",       ["drama", "thriller", "crime", "mystery", "detective", "murder"]),
    ("comedy",      ["comedy", "laugh", "funny", "sitcom"]),
    ("reality",     ["reality", "bake off", "love island", "celebrity", "talent"]),
    ("sport",       ["sport", "football", "cricket", "tennis", "match", "championship"]),
    ("news",        ["news", "tonight", "report", "investigation", "documentary"]),
    ("factual",     ["factual", "nature", "wildlife", "science", "planet", "earth"]),
]


def infer_genre(title: str) -> str:
    lower = title.lower()
    for genre, keywords in _GENRE_KEYWORDS:
        if any(k in lower for k in keywords):
            return genre
    return "general"


# ---------------------------------------------------------------------------
# Weather — Open-Meteo
# ---------------------------------------------------------------------------

def _open_meteo_url(lat: float, lon: float, target_date: date) -> str:
    """
    Return the correct Open-Meteo URL for a given date:
      past dates  → archive API  (https://archive-api.open-meteo.com/v1/archive)
      today/future (≤16 days) → forecast API with explicit date range
      far future (>16 days)   → forecast API with forecast_days=1 (best available)
    """
    today = date.today()
    base_params = (
        f"latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,sunshine_duration"
        f"&timezone=Europe%2FLondon"
    )

    if target_date < today:
        # Historical — use archive endpoint
        return (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?{base_params}&start_date={target_date}&end_date={target_date}"
        )
    elif target_date <= today + timedelta(days=15):
        # Today or near future — forecast endpoint with explicit date range
        return (
            f"https://api.open-meteo.com/v1/forecast"
            f"?{base_params}&start_date={target_date}&end_date={target_date}"
        )
    else:
        # Far future — climatological best-guess (forecast_days=1 returns tomorrow)
        return (
            f"https://api.open-meteo.com/v1/forecast"
            f"?{base_params}&forecast_days=1"
        )


def fetch_weather(station: str, target_date: date, timeout: int = 10) -> dict:
    """
    Return a weather dict for one UK station on one date.
    Falls back to conservative seasonal defaults on any API failure.
    """
    station_key = (station or DEFAULT_STATION).lower()
    lat, lon = UK_STATIONS.get(station_key, UK_STATIONS[DEFAULT_STATION])
    url = _open_meteo_url(lat, lon, target_date)

    tmax = tmin = rain = sun_h = None
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        daily = r.json().get("daily", {})

        tmax  = (daily.get("temperature_2m_max")  or [None])[0]
        tmin  = (daily.get("temperature_2m_min")  or [None])[0]
        rain  = (daily.get("precipitation_sum")   or [None])[0]
        sun_s = (daily.get("sunshine_duration")   or [None])[0]
        sun_h = round(float(sun_s) / 3600, 2) if sun_s is not None else None

    except Exception as exc:
        print(f"  [weather] {station_key} / {target_date}: {exc} — using defaults")

    # Apply defaults for any missing values
    tmax  = round(float(tmax),  1) if tmax  is not None else 12.0
    tmin  = round(float(tmin),  1) if tmin  is not None else 6.0
    rain  = round(float(rain),  2) if rain  is not None else 0.0
    sun_h = round(float(sun_h), 2) if sun_h is not None else 4.0

    # Derived indices (0–1 scale)
    rain_norm  = min(rain / 20.0, 1.0)
    sun_norm   = 1.0 - min(sun_h / 12.0, 1.0)
    temp_norm  = max(0.0, 1.0 - (tmax - 10.0) / 20.0)
    bad_index  = round(rain_norm * 0.45 + sun_norm * 0.35 + temp_norm * 0.20, 3)
    indoor_idx = round(min(1.0, bad_index * 1.15), 3)

    return {
        "weather_station":              station_key,
        "weather_tmax_c":               tmax,
        "weather_tmin_c":               tmin,
        "weather_rain_mm":              rain,
        "weather_sun_hours":            sun_h,
        "weather_bad_index":            bad_index,
        "weather_indoor_viewing_index": indoor_idx,
    }


# ---------------------------------------------------------------------------
# Content metadata — TMDB
# ---------------------------------------------------------------------------

# Circuit breaker: skip TMDB after this many consecutive failures
_TMDB_MAX_FAILURES = 3
_tmdb_failure_count = 0


def fetch_tmdb(title: str, api_key: str, timeout: int = 8) -> dict:
    """
    Search TMDB for a TV show and return model-ready metadata.
    Retries once on 429 (rate limit). Returns {} on any other failure.
    Activates circuit breaker after _TMDB_MAX_FAILURES consecutive misses.
    """
    global _tmdb_failure_count

    if _tmdb_failure_count >= _TMDB_MAX_FAILURES:
        return {}   # circuit open — skip silently

    # Strip scheduling suffix like " - Tue @ 2100"
    clean = title.split(" - ")[0].strip()
    url = (
        f"https://api.themoviedb.org/3/search/tv"
        f"?api_key={api_key}&query={requests.utils.quote(clean)}&language=en-GB"
    )

    for attempt in range(2):
        try:
            r = requests.get(url, timeout=timeout)

            if r.status_code == 429:
                # Respect Retry-After header, then try once more
                wait = int(r.headers.get("Retry-After", 2 ** (attempt + 1)))
                print(f"  [tmdb] rate-limited — waiting {wait}s")
                time.sleep(wait)
                continue

            r.raise_for_status()
            results = r.json().get("results", [])

            if not results:
                _tmdb_failure_count += 1
                return {}

            hit = results[0]
            popularity = round(min(float(hit.get("popularity", 0)) / 1000 * 100, 100), 1)
            rating     = round(float(hit.get("vote_average", 0)), 2)
            first_air  = hit.get("first_air_date", "")
            year       = int(first_air[:4]) if first_air and len(first_air) >= 4 else None
            years_old  = (date.today().year - year) if year else None
            premiere   = round(max(0.0, 1.0 - (years_old or 5) / 10.0), 3)

            _tmdb_failure_count = 0   # reset circuit breaker on success
            return {
                "netflix_popularity_score":       popularity,
                "netflix_rating_mean":            rating,
                "synthetic_premiere_probability": premiere,
                "synthetic_production_year_mean": float(year) if year else None,
                "ott_hook_strength":              round(min(rating / 10.0, 1.0), 3),
                "ott_visual_intensity":           round(popularity / 100.0, 3),
            }

        except Exception as exc:
            print(f"  [tmdb] '{clean}': {exc}")
            _tmdb_failure_count += 1
            return {}

    _tmdb_failure_count += 1
    return {}


# ---------------------------------------------------------------------------
# Derived feature computation
# ---------------------------------------------------------------------------

def compute_derived(row: pd.Series, schedule_df: pd.DataFrame, pos: int) -> dict:
    """
    Compute features derivable from the schedule row + its neighbours.

    Assumes enrich_schedule() has already parsed datetime and added
    hour / minute / day_of_week columns to schedule_df.
    """
    promo_title = str(row.get("promo_title", ""))
    dt          = row.get("datetime")

    # These are guaranteed to exist because enrich_schedule() added them
    hour        = int(row.get("hour", 12))
    minute      = int(row.get("minute", 0))
    dow         = int(row.get("day_of_week", 0))
    is_weekend  = int(dow >= 5)

    promo_len   = float(row.get("promo_length_seconds", 20))
    in_break    = int(row.get("promo_in_break", 1))
    break_pos   = int(row.get("break_event_position", 1))
    break_total = int(row.get("break_total_events", 3))
    break_pct   = break_pos / break_total if break_total > 0 else 0.5

    # Time band
    if   6  <= hour < 10: time_band = "morning"
    elif 10 <= hour < 18: time_band = "daytime"
    elif 18 <= hour < 23: time_band = "peak"
    else:                 time_band = "late"

    # ── Promo fatigue index ───────────────────────────────────────────────
    # Primary: time-windowed (last 7 days) if datetime is available
    # Fallback: simple frequency within the uploaded schedule
    if (
        "datetime" in schedule_df.columns
        and "promo_title" in schedule_df.columns
        and hasattr(dt, "date")
    ):
        cutoff     = dt - timedelta(days=7)
        past_all   = schedule_df[
            (schedule_df["datetime"] < dt) & (schedule_df["datetime"] >= cutoff)
        ]
        past_title = past_all[past_all["promo_title"] == promo_title]
        fatigue    = len(past_title) / max(len(past_all), 1)
    elif "promo_title" in schedule_df.columns:
        # No datetime — use fraction of schedule slots this title occupies
        title_count = (schedule_df["promo_title"] == promo_title).sum()
        fatigue     = title_count / max(len(schedule_df), 1)
    else:
        fatigue = 0.3

    # ── Attention context score ───────────────────────────────────────────
    attention = (
        (0.35 if time_band == "peak" else 0.20 if time_band == "morning" else 0.15)
        + 0.25 * break_pct        # end-of-break = higher attention
        + 0.20 * is_weekend
        + (0.10 if promo_len >= 20 else 0.05)
    )
    attention = round(min(1.0, attention), 3)

    # ── Neighbour promos (use iloc for positional safety) ─────────────────
    prev_title = next_title = None
    if "promo_title" in schedule_df.columns:
        if pos > 0:
            try:
                prev_title = str(schedule_df.iloc[pos - 1]["promo_title"])
            except (IndexError, KeyError):
                pass
        if pos < len(schedule_df) - 1:
            try:
                next_title = str(schedule_df.iloc[pos + 1]["promo_title"])
            except (IndexError, KeyError):
                pass

    # ── Lead to next programme ────────────────────────────────────────────
    if pd.notna(row.get("lead_to_next_program_min")):
        lead_min = float(row["lead_to_next_program_min"])
    else:
        lead_min = 30.0 if time_band == "peak" else 45.0

    return {
        "hour":                    hour,
        "minute":                  minute,
        "day_of_week":             dow,
        "is_weekend":              is_weekend,
        "time_band":               time_band,
        "promo_in_break":          in_break,
        "break_event_position":    break_pos,
        "break_total_events":      break_total,
        "break_position_pct":      round(break_pct, 4),
        "preceded_by_break":       in_break,
        "lead_to_next_program_min": lead_min,
        "promo_fatigue_index":     round(min(fatigue, 1.0), 4),
        "attention_context_score": attention,
        "prev_promo_title":        prev_title,
        "next_promo_title":        next_title,
        "genre_guess":             infer_genre(promo_title),
    }


# ---------------------------------------------------------------------------
# Main enrichment pipeline
# ---------------------------------------------------------------------------

def enrich_schedule(
    df: pd.DataFrame,
    tmdb_api_key: str | None = None,
    weather_cache: dict | None = None,
    tmdb_cache: dict | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Enrich a minimal schedule DataFrame with weather + content metadata
    + derived features, producing rows ready for the prediction model.

    Parameters
    ----------
    df            : DataFrame — at minimum needs a promo_title column
    tmdb_api_key  : TMDB API key; if None, OTT/Netflix columns use OTT_DEFAULTS
    weather_cache : Pass a dict to reuse weather fetches across multiple calls
    tmdb_cache    : Pass a dict to reuse TMDB lookups across multiple calls
    verbose       : Print progress to stdout

    Returns the enriched DataFrame.
    """
    global _tmdb_failure_count
    _tmdb_failure_count = 0   # reset circuit breaker for each new schedule

    if weather_cache is None:
        weather_cache = {}
    if tmdb_cache is None:
        tmdb_cache = {}

    df = df.copy()

    # ── Parse datetime first so derived features can use it ──────────────
    if "datetime" in df.columns and df["datetime"].dtype == object:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # ── Derive time columns from datetime if not supplied ─────────────────
    if "datetime" in df.columns:
        if "hour"        not in df.columns: df["hour"]        = df["datetime"].dt.hour
        if "minute"      not in df.columns: df["minute"]      = df["datetime"].dt.minute
        if "day_of_week" not in df.columns: df["day_of_week"] = df["datetime"].dt.weekday

    # ── Column defaults (only if completely absent) ───────────────────────
    _col_defaults = {
        "weather_station":      DEFAULT_STATION,
        "content_type":         "Promo",
        "event_type":           "Primary",
        "promo_length_seconds": 20.0,
        "promo_in_break":       1,
        "break_event_position": 1,
        "break_total_events":   3,
        "hour":                 12,
        "minute":               0,
        "day_of_week":          0,
    }
    for col, default in _col_defaults.items():
        if col not in df.columns:
            df[col] = default

    # Reset index so iloc lookups in compute_derived are correct
    df = df.reset_index(drop=True)

    enriched_rows: list[dict] = []
    total = len(df)

    for pos, row in df.iterrows():
        if verbose:
            title_short = str(row.get("promo_title", "?"))[:45]
            print(f"  [{pos + 1:>4}/{total}]  {title_short:<45}", end="\r", flush=True)

        # ── Weather (one API call per station/date, cached) ───────────────
        dt_val      = row.get("datetime")
        target_date = dt_val.date() if hasattr(dt_val, "date") else date.today()
        station     = str(row.get("weather_station", DEFAULT_STATION)).lower()
        wx_key      = (station, str(target_date))

        if wx_key not in weather_cache:
            weather_cache[wx_key] = fetch_weather(station, target_date)
        weather_data = weather_cache[wx_key]

        # ── TMDB content metadata (one lookup per unique clean title) ──────
        tmdb_data: dict = dict(OTT_DEFAULTS)   # always start with defaults
        if tmdb_api_key:
            clean_title = str(row.get("promo_title", "")).split(" - ")[0].strip()
            if clean_title not in tmdb_cache:
                tmdb_cache[clean_title] = fetch_tmdb(clean_title, tmdb_api_key)
                time.sleep(0.25)          # stay under TMDB's 40-req/10s limit
            tmdb_data.update(tmdb_cache[clean_title])   # override defaults with real data

        # ── Derived features ──────────────────────────────────────────────
        derived = compute_derived(row, df, int(pos))

        # ── Merge order: CSV row → derived → weather → TMDB ──────────────
        # Later sources override earlier ones for the same key.
        enriched = {**row.to_dict(), **derived, **weather_data, **tmdb_data}
        enriched_rows.append(enriched)

    if verbose:
        print(f"\n  Enriched {total} rows")

    result = pd.DataFrame(enriched_rows)

    # ── Validate all 37 model features are present ────────────────────────
    missing_features = [c for c in FEATURE_COLS if c not in result.columns]
    if missing_features and verbose:
        print(f"  [warn] {len(missing_features)} model features still missing "
              f"(will be NaN — model handles them): {missing_features}")

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Enrich a minimal promo schedule CSV with real-time weather + content data"
    )
    p.add_argument("--input",     required=True, help="Path to minimal schedule CSV")
    p.add_argument("--output",    default="",    help="Output path (default: <input>_enriched.csv)")
    p.add_argument(
        "--tmdb-key", default="",
        help="TMDB API key (or set TMDB_API_KEY env var). "
             "Free at https://www.themoviedb.org/settings/api",
    )
    p.add_argument(
        "--no-predict", action="store_true",
        help="Stop after enrichment — do not call the prediction API",
    )
    p.add_argument("--api-url", default="http://localhost:8000",
                   help="Prediction API base URL (default: http://localhost:8000)")
    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"ERROR: {input_path} not found")

    output_path = (
        Path(args.output) if args.output
        else input_path.with_name(input_path.stem + "_enriched.csv")
    )

    tmdb_key = args.tmdb_key or os.getenv("TMDB_API_KEY", "")

    # ── Load ─────────────────────────────────────────────────────────────
    print(f"\nLoading {input_path} ...")
    peek = pd.read_csv(input_path, nrows=0)
    df = pd.read_csv(
        input_path,
        parse_dates=["datetime"] if "datetime" in peek.columns else [],
    )
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    if "promo_title" not in df.columns:
        sys.exit(
            "ERROR: Input CSV must have a 'promo_title' column.\n"
            "Minimum columns: promo_title, datetime"
        )

    if tmdb_key:
        print(f"  TMDB key found — content metadata enabled")
    else:
        print(
            "  No TMDB key — OTT/Netflix columns will use built-in defaults.\n"
            "  Get a free key at https://www.themoviedb.org/settings/api\n"
            "  Then re-run with:  --tmdb-key YOUR_KEY  or  TMDB_API_KEY=YOUR_KEY"
        )

    # ── Enrich ────────────────────────────────────────────────────────────
    print(f"\nEnriching with live weather + derived features ...")
    enriched_df = enrich_schedule(df, tmdb_api_key=tmdb_key or None)
    enriched_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # ── Predict ───────────────────────────────────────────────────────────
    if not args.no_predict:
        print(f"\nChecking prediction API at {args.api_url} ...")
        try:
            r = requests.get(f"{args.api_url}/health", timeout=3)
            api_ready = r.status_code == 200
        except Exception:
            api_ready = False

        if api_ready:
            pred_output = output_path.with_name(output_path.stem + "_predictions.csv")
            subprocess.run(
                [
                    sys.executable, "predict_from_csv.py",
                    "--input",  str(output_path),
                    "--output", str(pred_output),
                    "--url",    args.api_url,
                ],
                check=True,
            )
        else:
            print(
                "  API not reachable — start it first:\n"
                "    uvicorn main:app --reload\n"
                f"  Then predict:\n"
                f"    python predict_from_csv.py --input {output_path}"
            )

    print("\nDone.\n")


if __name__ == "__main__":
    main()

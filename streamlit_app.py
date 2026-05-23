"""
Ancast Nowcasting — Streamlit Prototype
========================================
Connects to the FastAPI backend for live predictions.
Falls back to pre-scored mock data when the API is unreachable.

Run:
    streamlit run streamlit_app.py
"""

import io
import json
import os
import random
import time
from datetime import datetime, date
from pathlib import Path

import joblib
import numpy as np

from config import FEATURE_COLS as _FEATURE_COLS_LIST, LEAKY_COLUMNS as _LEAKY_COLS
import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL = os.getenv("API_URL", "http://localhost:8000")
REFRESH_INTERVAL_SEC = 30
CHANNEL = "Viasat History CEE"

# ---------------------------------------------------------------------------
# Page setup (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ancast Nowcasting",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Global text colour — force black everywhere ────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] * {
    color: #111 !important;
}
/* Restore white text on dark backgrounds */
.ancast-header, .ancast-header *,
.ancast-logo, .ancast-logo *,
.channel-pill, .date-nav, .stat-block,
.stat-label, .stat-value, .stat-sub,
.avatar,
.daypart-tab.active, .daypart-tab.active *,
.density-btn.active, .density-btn.active * {
    color: white !important;
}
/* Streamlit widget labels, help text, captions */
label, .stSelectbox label, .stTextInput label,
.stSlider label, .stNumberInput label,
[data-testid="stMarkdownContainer"] p,
[data-testid="stWidgetLabel"] { color: #111 !important; }
/* Expander headers */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p { color: #111 !important; }
/* Metric labels/values */
[data-testid="stMetricLabel"], [data-testid="stMetricValue"],
[data-testid="stMetricDelta"] { color: #111 !important; }
/* Selectbox + input text */
[data-baseweb="select"] span, [data-baseweb="input"] input { color: #111 !important; }

/* ── Layout ─────────────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] { background: #f0ebe0; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 0 0 2rem 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }

/* ── Header bar ─────────────────────────────────────────────────────────── */
.ancast-header {
    background: #1a1f38;
    color: white;
    padding: 0 28px;
    display: flex;
    align-items: center;
    gap: 0;
    height: 72px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.ancast-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: -0.3px;
    white-space: nowrap;
    padding-right: 20px;
    border-right: 1px solid rgba(255,255,255,0.15);
    margin-right: 18px;
}
.ancast-logo .logo-icon {
    width: 32px; height: 32px;
    background: #3a8ef6;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 900; font-size: 16px;
}
.channel-pill {
    display: flex; align-items: center; gap: 7px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 13px;
    cursor: pointer;
    white-space: nowrap;
    margin-right: 14px;
}
.channel-pill .dot { width: 8px; height: 8px; background: #22dd88; border-radius: 50%; }
.date-nav {
    display: flex; align-items: center; gap: 8px;
    font-size: 13px;
    margin-right: 20px;
    padding-right: 20px;
    border-right: 1px solid rgba(255,255,255,0.15);
    white-space: nowrap;
}
.date-nav .arrow {
    width: 24px; height: 24px;
    background: rgba(255,255,255,0.08);
    border-radius: 5px;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; font-size: 12px;
}
.date-label { font-weight: 600; }
.stat-block {
    display: flex; flex-direction: column; justify-content: center;
    padding: 0 20px;
    border-right: 1px solid rgba(255,255,255,0.1);
    min-width: 140px;
}
.stat-block:last-of-type { border-right: none; }
.stat-label { font-size: 9px; letter-spacing: 0.8px; text-transform: uppercase; color: rgba(255,255,255,0.5); margin-bottom: 1px; }
.stat-value { font-size: 22px; font-weight: 700; line-height: 1.1; }
.stat-sub { font-size: 10px; color: rgba(255,255,255,0.4); margin-top: 2px; display: flex; align-items: center; gap: 4px; }
.stat-sub .live-dot { width: 6px; height: 6px; background: #22dd88; border-radius: 50%; }
.header-spacer { flex: 1; }
.avatar {
    width: 36px; height: 36px;
    background: #3a8ef6;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 13px;
}

/* ── Advisory banner ─────────────────────────────────────────────────────── */
.advisory-banner {
    background: #e8f0fe;
    border-left: 3px solid #3a8ef6;
    padding: 10px 28px;
    font-size: 12.5px;
    color: #2c4a8a;
    display: flex; align-items: center; gap: 8px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* ── Filter bar ──────────────────────────────────────────────────────────── */
.filter-bar {
    background: #f0ebe0;
    padding: 10px 28px;
    display: flex; align-items: center; gap: 8px;
    border-bottom: 1px solid #ddd8cc;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    flex-wrap: wrap;
}
.daypart-tab {
    padding: 5px 14px;
    border-radius: 6px;
    font-size: 13px;
    cursor: pointer;
    border: none;
    background: transparent;
    color: #555;
    font-weight: 500;
}
.daypart-tab.active {
    background: #1a1f38;
    color: white;
}
.filter-divider { width: 1px; height: 22px; background: #ccc; margin: 0 6px; }
.filter-pill {
    padding: 5px 12px;
    border-radius: 6px;
    font-size: 12.5px;
    border: 1px solid #ccc;
    background: white;
    color: #333;
    cursor: pointer;
    display: inline-flex; align-items: center; gap: 4px;
}
.search-box {
    margin-left: auto;
    border: 1px solid #ccc;
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 12.5px;
    background: white;
    color: #888;
    min-width: 160px;
}

/* ── Table ───────────────────────────────────────────────────────────────── */
.sched-wrap { padding: 0 28px; margin-top: 4px; }
.sched-table {
    width: 100%;
    border-collapse: collapse;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 13px;
}
.sched-table thead tr {
    border-bottom: 2px solid #ccc;
}
.sched-table thead th {
    text-align: left;
    font-size: 10px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: #888;
    padding: 8px 10px;
    font-weight: 600;
}
.sched-table tbody tr {
    border-bottom: 1px solid #e0dbd0;
    transition: background 0.1s;
}
.sched-table tbody tr:hover { background: rgba(0,0,0,0.025); }
.sched-table tbody td {
    padding: 12px 10px;
    vertical-align: middle;
    color: #1a1a2a;
}
.tx-time { font-weight: 600; font-size: 14px; }
.title-cell { font-weight: 500; }
.genre-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 10.5px;
    font-weight: 600;
    margin-left: 6px;
    vertical-align: middle;
}
.tag-news     { background: #ffe0e6; color: #aa2244; }
.tag-drama    { background: #ede0ff; color: #6622aa; }
.tag-factual  { background: #d8f5e8; color: #116633; }
.tag-comedy   { background: #fff3cc; color: #997700; }
.tag-reality  { background: #ffe0f0; color: #aa1166; }
.tag-sport    { background: #d8eeff; color: #115588; }
.tag-default  { background: #ebebeb; color: #555555; }
.duration-cell { color: #555; font-variant-numeric: tabular-nums; }
.type-cell { color: #444; }

/* Flag badges */
.flag-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 11.5px;
    font-weight: 600;
    white-space: nowrap;
}
.flag-swap { background: #ffebeb; color: #cc2200; border: 1px solid #ffbbaa; }
.flag-move { background: #fff5e0; color: #cc7700; border: 1px solid #ffcc88; }
.flag-ok   { background: #e8faf0; color: #116633; border: 1px solid #aaddc0; }
.flag-dot-red    { width: 7px; height: 7px; background: #dd3300; border-radius: 50%; }
.flag-dot-orange { width: 7px; height: 7px; background: #cc7700; border-radius: 50%; }
.flag-dot-green  { width: 7px; height: 7px; background: #22aa66; border-radius: 50%; }

.action-cell { color: #333; max-width: 200px; }
.uplift-cell { font-weight: 700; color: #117733; }
.rationale-cell { color: #666; font-size: 12px; }

/* ── Footer ──────────────────────────────────────────────────────────────── */
.sched-footer {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 28px;
    font-size: 12px;
    color: #888;
    border-top: 1px solid #ddd8cc;
    margin-top: 8px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.density-btn {
    padding: 4px 12px;
    border-radius: 5px;
    border: 1px solid #ccc;
    background: white;
    font-size: 12px;
    cursor: pointer;
    color: #333;
}
.density-btn.active { background: #1a1f38; color: white; border-color: #1a1f38; }
.footer-spacer { flex: 1; }
.refresh-info { color: #999; font-size: 11px; }

/* ── Streamlit widget overrides ─────────────────────────────────────────── */
div[data-testid="stHorizontalBlock"] { gap: 8px; }
.stSelectbox > div { font-size: 13px; }
div[data-baseweb="select"] > div { min-height: 32px; border-radius: 6px; background: white; border: 1px solid #ccc; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "daypart" not in st.session_state:
    st.session_state.daypart = "Daytime"
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if "predictions_cache" not in st.session_state:
    st.session_state.predictions_cache = {}
if "real_schedule" not in st.session_state:
    st.session_state.real_schedule = None

# ---------------------------------------------------------------------------
# Mock schedule data
# ---------------------------------------------------------------------------

# Feature names match inference_schema.json exactly.
# "title" and "genre_display" are display-only — not model features.
MOCK_SCHEDULE = [
    {"tx_time": "06:00", "title": "Alexander's Lost World",          "genre_display": "History Doc", "duration": "00:20", "promo_type_display": "Promo · break",     "daypart": "Breakfast",
     "channel": "Viasat History CEE", "promo_title": "Alexander's Lost World - Tue @ 2100",     "content_type": "Promo", "event_type": "Primary",            "genre_guess": "history_doc", "promo_length_seconds": 20,  "hour": 6,  "minute": 0,  "day_of_week": 5, "is_weekend": 1, "time_band": "morning", "promo_position_type": "break_middle",    "promo_in_break": 1, "break_event_position": 2, "break_total_events": 4, "break_position_pct": 0.50, "preceded_by_break": 1, "lead_to_next_program_min": 45.0, "weather_station": "hurn", "weather_tmax_c": 14.2, "weather_tmin_c": 7.1, "weather_rain_mm": 1.2, "weather_sun_hours": 3.5, "weather_bad_index": 0.35, "weather_indoor_viewing_index": 0.62, "ott_avg_watch_pct": 0.68, "ott_dropoff_prob": 0.22, "ott_hook_strength": 0.71, "ott_visual_intensity": 0.55, "netflix_popularity_score": 42.0, "netflix_rating_mean": 7.1, "synthetic_premiere_probability": 0.30, "synthetic_production_year_mean": 2019.5, "promo_fatigue_index": 0.30, "attention_context_score": 0.45},
    {"tx_time": "07:15", "title": "Europe's Last Warrior Kings",     "genre_display": "History Doc", "duration": "00:20", "promo_type_display": "Promo · mid-roll",  "daypart": "Breakfast",
     "channel": "Viasat History CEE", "promo_title": "Europe's Last Warrior Kings - Wed @ 2100", "content_type": "Promo", "event_type": "Primary",            "genre_guess": "history_doc", "promo_length_seconds": 20,  "hour": 7,  "minute": 15, "day_of_week": 5, "is_weekend": 1, "time_band": "morning", "promo_position_type": "break_end",       "promo_in_break": 1, "break_event_position": 3, "break_total_events": 3, "break_position_pct": 1.0,  "preceded_by_break": 0, "lead_to_next_program_min": 60.0, "weather_station": "hurn", "weather_tmax_c": 14.2, "weather_tmin_c": 7.1, "weather_rain_mm": 0.0, "weather_sun_hours": 4.2, "weather_bad_index": 0.20, "weather_indoor_viewing_index": 0.48, "ott_avg_watch_pct": 0.72, "ott_dropoff_prob": 0.18, "ott_hook_strength": 0.80, "ott_visual_intensity": 0.60, "netflix_popularity_score": 38.0, "netflix_rating_mean": 7.4, "synthetic_premiere_probability": 0.40, "synthetic_production_year_mean": 2018.0, "promo_fatigue_index": 0.20, "attention_context_score": 0.50},
    {"tx_time": "08:45", "title": "Animated Ident 1",               "genre_display": "Promo",       "duration": "00:10", "promo_type_display": "Jingle",            "daypart": "Breakfast",
     "channel": "Viasat History CEE", "promo_title": "Animated Ident 1 - Napoleon HD cutdown 10","content_type": "Jingle","event_type": "Primary-BreakHeader","genre_guess": "promo",       "promo_length_seconds": 10,  "hour": 8,  "minute": 45, "day_of_week": 5, "is_weekend": 1, "time_band": "morning", "promo_position_type": "programme_lead_in","promo_in_break": 0, "break_event_position": 1, "break_total_events": 1, "break_position_pct": 1.0,  "preceded_by_break": 0, "lead_to_next_program_min": 2.0,  "weather_station": "hurn", "weather_tmax_c": 14.2, "weather_tmin_c": 7.1, "weather_rain_mm": 3.0, "weather_sun_hours": 1.5, "weather_bad_index": 0.60, "weather_indoor_viewing_index": 0.75, "ott_avg_watch_pct": 0.55, "ott_dropoff_prob": 0.30, "ott_hook_strength": 0.50, "ott_visual_intensity": 0.80, "netflix_popularity_score": 55.0, "netflix_rating_mean": 6.8, "synthetic_premiere_probability": 0.10, "synthetic_production_year_mean": 2021.0, "promo_fatigue_index": 0.55, "attention_context_score": 0.38},
    {"tx_time": "10:15", "title": "Alexander's Lost World",          "genre_display": "History Doc", "duration": "00:20", "promo_type_display": "Promo · mid-roll",  "daypart": "Daytime",
     "channel": "Viasat History CEE", "promo_title": "Alexander's Lost World - Tue @ 2100",     "content_type": "Promo", "event_type": "Primary",            "genre_guess": "history_doc", "promo_length_seconds": 20,  "hour": 10, "minute": 15, "day_of_week": 5, "is_weekend": 1, "time_band": "morning", "promo_position_type": "break_middle",    "promo_in_break": 1, "break_event_position": 2, "break_total_events": 3, "break_position_pct": 0.67, "preceded_by_break": 1, "lead_to_next_program_min": 30.0, "weather_station": "hurn", "weather_tmax_c": 14.2, "weather_tmin_c": 7.1, "weather_rain_mm": 7.5, "weather_sun_hours": 0.8, "weather_bad_index": 0.80, "weather_indoor_viewing_index": 0.88, "ott_avg_watch_pct": 0.60, "ott_dropoff_prob": 0.28, "ott_hook_strength": 0.65, "ott_visual_intensity": 0.52, "netflix_popularity_score": 42.0, "netflix_rating_mean": 7.1, "synthetic_premiere_probability": 0.30, "synthetic_production_year_mean": 2019.5, "promo_fatigue_index": 0.85, "attention_context_score": 0.30},
    {"tx_time": "11:45", "title": "Queen Victoria's Letters",        "genre_display": "History Doc", "duration": "00:20", "promo_type_display": "Promo · break",     "daypart": "Daytime",
     "channel": "Viasat History CEE", "promo_title": "Queen Victoria's Letters: A Monarch - Thurs @ 2150","content_type": "Promo","event_type": "Primary",   "genre_guess": "history_doc", "promo_length_seconds": 20,  "hour": 11, "minute": 45, "day_of_week": 5, "is_weekend": 1, "time_band": "morning", "promo_position_type": "break_end",       "promo_in_break": 1, "break_event_position": 4, "break_total_events": 4, "break_position_pct": 1.0,  "preceded_by_break": 1, "lead_to_next_program_min": 20.0, "weather_station": "hurn", "weather_tmax_c": 14.2, "weather_tmin_c": 7.1, "weather_rain_mm": 4.2, "weather_sun_hours": 2.0, "weather_bad_index": 0.55, "weather_indoor_viewing_index": 0.70, "ott_avg_watch_pct": 0.75, "ott_dropoff_prob": 0.15, "ott_hook_strength": 0.88, "ott_visual_intensity": 0.45, "netflix_popularity_score": 35.0, "netflix_rating_mean": 7.8, "synthetic_premiere_probability": 0.55, "synthetic_production_year_mean": 2017.0, "promo_fatigue_index": 0.40, "attention_context_score": 0.55},
    {"tx_time": "13:15", "title": "Annihilation: The Destruction",   "genre_display": "History Doc", "duration": "00:20", "promo_type_display": "Promo · mid-roll",  "daypart": "Daytime",
     "channel": "Viasat History CEE", "promo_title": "Annihilation: The Destruction - Tonight @ 2250","content_type": "Promo","event_type": "Primary",      "genre_guess": "history_doc", "promo_length_seconds": 20,  "hour": 13, "minute": 15, "day_of_week": 5, "is_weekend": 1, "time_band": "morning", "promo_position_type": "break_middle",    "promo_in_break": 1, "break_event_position": 1, "break_total_events": 3, "break_position_pct": 0.33, "preceded_by_break": 0, "lead_to_next_program_min": 55.0, "weather_station": "hurn", "weather_tmax_c": 14.2, "weather_tmin_c": 7.1, "weather_rain_mm": 0.5, "weather_sun_hours": 5.5, "weather_bad_index": 0.25, "weather_indoor_viewing_index": 0.42, "ott_avg_watch_pct": 0.82, "ott_dropoff_prob": 0.12, "ott_hook_strength": 0.92, "ott_visual_intensity": 0.78, "netflix_popularity_score": 62.0, "netflix_rating_mean": 8.2, "synthetic_premiere_probability": 0.70, "synthetic_production_year_mean": 2022.0, "promo_fatigue_index": 0.22, "attention_context_score": 0.65},
    {"tx_time": "14:45", "title": "Europe's Last Warrior Kings",     "genre_display": "History Doc", "duration": "00:20", "promo_type_display": "Promo · mid-roll",  "daypart": "Daytime",
     "channel": "Viasat History CEE", "promo_title": "Europe's Last Warrior Kings - Wed @ 2100", "content_type": "Promo", "event_type": "Primary",            "genre_guess": "history_doc", "promo_length_seconds": 20,  "hour": 14, "minute": 45, "day_of_week": 5, "is_weekend": 1, "time_band": "morning", "promo_position_type": "break_middle",    "promo_in_break": 1, "break_event_position": 2, "break_total_events": 4, "break_position_pct": 0.50, "preceded_by_break": 1, "lead_to_next_program_min": 35.0, "weather_station": "hurn", "weather_tmax_c": 14.2, "weather_tmin_c": 7.1, "weather_rain_mm": 0.0, "weather_sun_hours": 6.2, "weather_bad_index": 0.15, "weather_indoor_viewing_index": 0.35, "ott_avg_watch_pct": 0.70, "ott_dropoff_prob": 0.20, "ott_hook_strength": 0.78, "ott_visual_intensity": 0.62, "netflix_popularity_score": 38.0, "netflix_rating_mean": 7.4, "synthetic_premiere_probability": 0.40, "synthetic_production_year_mean": 2018.0, "promo_fatigue_index": 0.60, "attention_context_score": 0.48},
    {"tx_time": "15:30", "title": "Alexander's Lost World",          "genre_display": "History Doc", "duration": "00:20", "promo_type_display": "Promo · break",     "daypart": "Daytime",
     "channel": "Viasat History CEE", "promo_title": "Alexander's Lost World - Tue @ 2100",     "content_type": "Promo", "event_type": "Primary",            "genre_guess": "history_doc", "promo_length_seconds": 20,  "hour": 15, "minute": 30, "day_of_week": 5, "is_weekend": 1, "time_band": "morning", "promo_position_type": "break_end",       "promo_in_break": 1, "break_event_position": 3, "break_total_events": 3, "break_position_pct": 1.0,  "preceded_by_break": 1, "lead_to_next_program_min": 40.0, "weather_station": "hurn", "weather_tmax_c": 14.2, "weather_tmin_c": 7.1, "weather_rain_mm": 0.0, "weather_sun_hours": 7.0, "weather_bad_index": 0.10, "weather_indoor_viewing_index": 0.28, "ott_avg_watch_pct": 0.65, "ott_dropoff_prob": 0.25, "ott_hook_strength": 0.72, "ott_visual_intensity": 0.55, "netflix_popularity_score": 42.0, "netflix_rating_mean": 7.1, "synthetic_premiere_probability": 0.35, "synthetic_production_year_mean": 2019.5, "promo_fatigue_index": 0.35, "attention_context_score": 0.72},
    {"tx_time": "17:00", "title": "Promotion",                       "genre_display": "General",     "duration": "00:20", "promo_type_display": "Promo · lead-in",   "daypart": "Daytime",
     "channel": "Viasat History CEE", "promo_title": "Promotion",                               "content_type": "Promo", "event_type": "Primary",            "genre_guess": "general",     "promo_length_seconds": 20,  "hour": 17, "minute": 0,  "day_of_week": 5, "is_weekend": 1, "time_band": "morning", "promo_position_type": "programme_lead_in","promo_in_break": 0, "break_event_position": 1, "break_total_events": 1, "break_position_pct": 1.0,  "preceded_by_break": 0, "lead_to_next_program_min": 5.0,  "weather_station": "hurn", "weather_tmax_c": 14.2, "weather_tmin_c": 7.1, "weather_rain_mm": 0.0, "weather_sun_hours": 5.8, "weather_bad_index": 0.20, "weather_indoor_viewing_index": 0.38, "ott_avg_watch_pct": 0.58, "ott_dropoff_prob": 0.32, "ott_hook_strength": 0.55, "ott_visual_intensity": 0.40, "netflix_popularity_score": 30.0, "netflix_rating_mean": 6.5, "synthetic_premiere_probability": 0.20, "synthetic_production_year_mean": 2020.0, "promo_fatigue_index": 0.45, "attention_context_score": 0.58},
    {"tx_time": "19:00", "title": "Queen Victoria's Letters",        "genre_display": "History Doc", "duration": "00:20", "promo_type_display": "Promo · break",     "daypart": "Peak",
     "channel": "Viasat History CEE", "promo_title": "Queen Victoria's Letters: A Monarch - Thurs @ 2150","content_type": "Promo","event_type": "Primary",   "genre_guess": "history_doc", "promo_length_seconds": 20,  "hour": 19, "minute": 0,  "day_of_week": 5, "is_weekend": 1, "time_band": "morning", "promo_position_type": "break_middle",    "promo_in_break": 1, "break_event_position": 2, "break_total_events": 4, "break_position_pct": 0.50, "preceded_by_break": 1, "lead_to_next_program_min": 65.0, "weather_station": "hurn", "weather_tmax_c": 14.2, "weather_tmin_c": 7.1, "weather_rain_mm": 0.0, "weather_sun_hours": 2.0, "weather_bad_index": 0.30, "weather_indoor_viewing_index": 0.55, "ott_avg_watch_pct": 0.78, "ott_dropoff_prob": 0.14, "ott_hook_strength": 0.85, "ott_visual_intensity": 0.48, "netflix_popularity_score": 35.0, "netflix_rating_mean": 7.8, "synthetic_premiere_probability": 0.55, "synthetic_production_year_mean": 2017.0, "promo_fatigue_index": 0.28, "attention_context_score": 0.80},
    {"tx_time": "20:00", "title": "Annihilation: The Destruction",   "genre_display": "History Doc", "duration": "00:20", "promo_type_display": "Promo · mid-roll",  "daypart": "Peak",
     "channel": "Viasat History CEE", "promo_title": "Annihilation: The Destruction - Tonight @ 2250","content_type": "Promo","event_type": "Primary",      "genre_guess": "history_doc", "promo_length_seconds": 20,  "hour": 20, "minute": 0,  "day_of_week": 5, "is_weekend": 1, "time_band": "morning", "promo_position_type": "break_end",       "promo_in_break": 1, "break_event_position": 4, "break_total_events": 4, "break_position_pct": 1.0,  "preceded_by_break": 1, "lead_to_next_program_min": 10.0, "weather_station": "hurn", "weather_tmax_c": 14.2, "weather_tmin_c": 7.1, "weather_rain_mm": 0.0, "weather_sun_hours": 0.5, "weather_bad_index": 0.35, "weather_indoor_viewing_index": 0.60, "ott_avg_watch_pct": 0.88, "ott_dropoff_prob": 0.10, "ott_hook_strength": 0.95, "ott_visual_intensity": 0.82, "netflix_popularity_score": 62.0, "netflix_rating_mean": 8.2, "synthetic_premiere_probability": 0.70, "synthetic_production_year_mean": 2022.0, "promo_fatigue_index": 0.15, "attention_context_score": 0.90},
    {"tx_time": "20:45", "title": "Animated Ident 2",               "genre_display": "Promo",       "duration": "00:10", "promo_type_display": "Jingle",            "daypart": "Peak",
     "channel": "Viasat History CEE", "promo_title": "Animated Ident 2 - Pushkin HD cutdown 10","content_type": "Jingle","event_type": "Primary-BreakHeader","genre_guess": "promo",      "promo_length_seconds": 10,  "hour": 20, "minute": 45, "day_of_week": 5, "is_weekend": 1, "time_band": "morning", "promo_position_type": "programme_lead_in","promo_in_break": 0, "break_event_position": 1, "break_total_events": 1, "break_position_pct": 1.0,  "preceded_by_break": 0, "lead_to_next_program_min": 3.0,  "weather_station": "hurn", "weather_tmax_c": 14.2, "weather_tmin_c": 7.1, "weather_rain_mm": 0.0, "weather_sun_hours": 0.2, "weather_bad_index": 0.40, "weather_indoor_viewing_index": 0.65, "ott_avg_watch_pct": 0.82, "ott_dropoff_prob": 0.12, "ott_hook_strength": 0.90, "ott_visual_intensity": 0.85, "netflix_popularity_score": 58.0, "netflix_rating_mean": 7.9, "synthetic_premiere_probability": 0.60, "synthetic_production_year_mean": 2021.5, "promo_fatigue_index": 0.20, "attention_context_score": 0.85},
    {"tx_time": "21:30", "title": "Europe's Last Warrior Kings",     "genre_display": "History Doc", "duration": "00:20", "promo_type_display": "Promo · mid-roll",  "daypart": "Peak",
     "channel": "Viasat History CEE", "promo_title": "Europe's Last Warrior Kings - Wed @ 2100", "content_type": "Promo", "event_type": "Primary",            "genre_guess": "history_doc", "promo_length_seconds": 20,  "hour": 21, "minute": 30, "day_of_week": 5, "is_weekend": 1, "time_band": "morning", "promo_position_type": "break_middle",    "promo_in_break": 1, "break_event_position": 2, "break_total_events": 3, "break_position_pct": 0.67, "preceded_by_break": 1, "lead_to_next_program_min": 25.0, "weather_station": "hurn", "weather_tmax_c": 14.2, "weather_tmin_c": 7.1, "weather_rain_mm": 0.0, "weather_sun_hours": 0.0, "weather_bad_index": 0.42, "weather_indoor_viewing_index": 0.68, "ott_avg_watch_pct": 0.75, "ott_dropoff_prob": 0.18, "ott_hook_strength": 0.82, "ott_visual_intensity": 0.65, "netflix_popularity_score": 38.0, "netflix_rating_mean": 7.4, "synthetic_premiere_probability": 0.45, "synthetic_production_year_mean": 2018.0, "promo_fatigue_index": 0.30, "attention_context_score": 0.75},
    {"tx_time": "23:00", "title": "Alexander's Lost World",          "genre_display": "History Doc", "duration": "00:20", "promo_type_display": "Promo · break",     "daypart": "Late",
     "channel": "Viasat History CEE", "promo_title": "Alexander's Lost World - Tue @ 2100",     "content_type": "Promo", "event_type": "Primary",            "genre_guess": "history_doc", "promo_length_seconds": 20,  "hour": 23, "minute": 0,  "day_of_week": 5, "is_weekend": 1, "time_band": "morning", "promo_position_type": "break_middle",    "promo_in_break": 1, "break_event_position": 1, "break_total_events": 3, "break_position_pct": 0.33, "preceded_by_break": 0, "lead_to_next_program_min": 50.0, "weather_station": "hurn", "weather_tmax_c": 14.2, "weather_tmin_c": 7.1, "weather_rain_mm": 0.0, "weather_sun_hours": 0.0, "weather_bad_index": 0.45, "weather_indoor_viewing_index": 0.70, "ott_avg_watch_pct": 0.55, "ott_dropoff_prob": 0.35, "ott_hook_strength": 0.60, "ott_visual_intensity": 0.50, "netflix_popularity_score": 42.0, "netflix_rating_mean": 7.1, "synthetic_premiere_probability": 0.30, "synthetic_production_year_mean": 2019.5, "promo_fatigue_index": 0.65, "attention_context_score": 0.42},
    {"tx_time": "23:45", "title": "Queen Victoria's Letters",        "genre_display": "History Doc", "duration": "00:20", "promo_type_display": "Promo · mid-roll",  "daypart": "Late",
     "channel": "Viasat History CEE", "promo_title": "Queen Victoria's Letters: A Monarch - Thurs @ 2150","content_type": "Promo","event_type": "Primary",   "genre_guess": "history_doc", "promo_length_seconds": 20,  "hour": 23, "minute": 45, "day_of_week": 5, "is_weekend": 1, "time_band": "morning", "promo_position_type": "break_end",       "promo_in_break": 1, "break_event_position": 3, "break_total_events": 3, "break_position_pct": 1.0,  "preceded_by_break": 1, "lead_to_next_program_min": 15.0, "weather_station": "hurn", "weather_tmax_c": 14.2, "weather_tmin_c": 7.1, "weather_rain_mm": 0.0, "weather_sun_hours": 0.0, "weather_bad_index": 0.48, "weather_indoor_viewing_index": 0.72, "ott_avg_watch_pct": 0.48, "ott_dropoff_prob": 0.40, "ott_hook_strength": 0.52, "ott_visual_intensity": 0.42, "netflix_popularity_score": 35.0, "netflix_rating_mean": 7.8, "synthetic_premiere_probability": 0.55, "synthetic_production_year_mean": 2017.0, "promo_fatigue_index": 0.75, "attention_context_score": 0.25},
]

ACTIONS = {
    ("history_doc", True):  "Consider moving to peak slot",
    ("history_doc", False): "Strong historical slot — keep",
    ("promo",       True):  "Consider alternative ident position",
    ("promo",       False): "Ident position optimal",
    ("general",     True):  "Consider repositioning promo",
    ("general",     False): "Placement performing well",
}
RATIONALES = {
    ("history_doc", True):  "Fatigue index above threshold",
    ("history_doc", False): "High OTT hook strength",
    ("promo",       True):  "Low attention context score",
    ("promo",       False): "Lead-in position confirmed",
    ("general",     True):  "Weather index driving indoor viewing",
    ("general",     False): "Audience context score strong",
}

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


_FEATURE_KEYS = set(_FEATURE_COLS_LIST)

def fetch_predictions(schedule: list[dict]) -> list[dict]:
    """Call /predict/batch with only model feature fields."""
    payload = [
        {k: row[k] for k in _FEATURE_KEYS if k in row}
        for row in schedule
    ]
    try:
        r = requests.post(f"{API_URL}/predict/batch", json=payload, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


@st.cache_resource(show_spinner=False)
def _load_local_models():
    model_dir = Path("model")
    cls_path = model_dir / "promo_move_classifier.pkl"
    if not cls_path.exists():
        return None, None, None
    cls = joblib.load(cls_path)
    reg = joblib.load(model_dir / "promo_uplift_regressor.pkl")
    schema_path = model_dir / "inference_schema.json"
    schema = json.loads(schema_path.read_text()) if schema_path.exists() else None
    return cls, reg, schema


def _prepare_df(df: pd.DataFrame, schema) -> pd.DataFrame:
    if schema is None:
        return df
    prepared = df.reindex(columns=schema["feature_cols"])
    for col in schema.get("categorical_cols", []):
        cats = schema.get("category_values", {}).get(col) or []
        if cats:
            prepared[col] = pd.Categorical(prepared[col].astype(str), categories=cats)
        else:
            prepared[col] = prepared[col].astype("category")
    return prepared


def local_predictions(rows: list[dict]) -> list[dict]:
    """Run real ML inference locally — used when the FastAPI backend is offline."""
    cls_model, reg_model, schema = _load_local_models()
    if cls_model is None:
        return mock_predictions(rows)

    df = pd.DataFrame([{k: row.get(k) for k in _FEATURE_KEYS} for row in rows])
    X = _prepare_df(df, schema)

    probas = cls_model.predict_proba(X)[:, 1]
    moves = (probas >= 0.5).astype(int)
    uplifts = np.zeros(len(rows))
    move_idx = np.where(moves == 1)[0]
    if len(move_idx):
        uplifts[move_idx] = reg_model.predict(X.iloc[move_idx])

    return [
        {
            "should_move": int(m),
            "move_probability": round(float(p), 4),
            "predicted_uplift": round(float(u), 4),
        }
        for m, p, u in zip(moves, probas, uplifts)
    ]


def csv_to_schedule_rows(df: pd.DataFrame) -> list[dict]:
    """Convert a real promo schedule CSV into MOCK_SCHEDULE-compatible row dicts."""
    GENRE_DISPLAY = {
        "history_doc": "History Doc", "general": "General", "promo": "Promo",
        "drama": "Drama", "comedy": "Comedy", "reality": "Reality",
        "sport": "Sport", "news": "News", "factual": "Factual",
    }

    def _hour_to_daypart(h: int) -> str:
        if 6 <= h < 10:  return "Breakfast"
        if 10 <= h < 18: return "Daytime"
        if 18 <= h < 23: return "Peak"
        return "Late"

    rows = []
    for _, r in df.iterrows():
        hour   = int(r["hour"])   if "hour"   in r.index and pd.notna(r["hour"])   else 12
        minute = int(r["minute"]) if "minute" in r.index and pd.notna(r["minute"]) else 0

        raw_title   = str(r["promo_title"]) if "promo_title" in r.index and pd.notna(r.get("promo_title")) else "Unknown"
        genre_guess = str(r["genre_guess"]) if "genre_guess" in r.index and pd.notna(r.get("genre_guess")) else "general"
        promo_len   = float(r["promo_length_seconds"]) if "promo_length_seconds" in r.index and pd.notna(r.get("promo_length_seconds")) else 20.0
        content_t   = str(r["content_type"])         if "content_type"         in r.index and pd.notna(r.get("content_type"))         else "Promo"
        pos_type    = str(r["promo_position_type"])  if "promo_position_type"  in r.index and pd.notna(r.get("promo_position_type"))  else "break_middle"

        title        = raw_title.split(" - ")[0] if " - " in raw_title else raw_title
        dur_m, dur_s = int(promo_len // 60), int(promo_len % 60)

        row_dict = {
            "tx_time":          f"{hour:02d}:{minute:02d}",
            "title":            title,
            "genre_display":    GENRE_DISPLAY.get(genre_guess, genre_guess.replace("_", " ").title()),
            "duration":         f"{dur_m:02d}:{dur_s:02d}",
            "promo_type_display": f"{content_t} · {pos_type.replace('_', ' ').title()}",
            "daypart":          _hour_to_daypart(hour),
        }
        # Carry all feature columns through (NaN allowed — model handles it natively)
        for col in _FEATURE_KEYS:
            if col in r.index:
                val = r[col]
                row_dict[col] = (val.item() if hasattr(val, "item") else val) if pd.notna(val) else None
        rows.append(row_dict)
    return rows


def mock_predictions(schedule: list[dict]) -> list[dict]:
    """Deterministic fallback predictions when API is unreachable."""
    random.seed(42)
    results = []
    for row in schedule:
        score = row["promo_fatigue_index"] * 0.6 + (1 - row["attention_context_score"]) * 0.4
        score += row["weather_rain_mm"] * 0.02
        proba = min(max(score + random.gauss(0, 0.05), 0.0), 1.0)
        should_move = int(proba >= 0.5)
        uplift = round(random.uniform(1.5, 4.8), 1) if should_move else 0.0
        results.append({
            "should_move": should_move,
            "move_probability": round(proba, 4),
            "predicted_uplift": uplift,
        })
    return results


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

GENRE_CLASSES = {
    "History Doc": "tag-drama", "Promo": "tag-default",
    "General": "tag-factual",
}

def genre_tag(genre: str) -> str:
    cls = GENRE_CLASSES.get(genre, "tag-default")
    return f'<span class="genre-tag {cls}">{genre}</span>'

def flag_badge(should_move: int, proba: float) -> str:
    if should_move == 1 and proba >= 0.70:
        return '<span class="flag-badge flag-swap"><span class="flag-dot-red"></span>Review — swap</span>'
    elif should_move == 1:
        return '<span class="flag-badge flag-move"><span class="flag-dot-orange"></span>Review — move</span>'
    else:
        return '<span class="flag-badge flag-ok"><span class="flag-dot-green"></span>On track</span>'

def build_table_html(rows: list[dict]) -> str:
    if not rows:
        return "<p style='color:#888;padding:20px'>No items match the current filters.</p>"

    cols = ["TX TIME", "TITLE", "DURATION", "TYPE", "FLAG", "ACTION", "UPLIFT", "RATIONALE"]
    header = "".join(f"<th>{c}</th>" for c in cols)

    body_rows = []
    for row in rows:
        sm = row["should_move"]
        proba = row["move_probability"]
        uplift = row["predicted_uplift"]
        genre_key = row["genre_guess"]           # model feature
        genre_display = row["genre_display"]     # human-readable display label

        action    = ACTIONS.get((genre_key, bool(sm)),    "Review placement")
        rationale = RATIONALES.get((genre_key, bool(sm)), "Model confidence high")
        uplift_str = f'<span class="uplift-cell">+{uplift:.1f}%</span>' if sm else '<span style="color:#aaa">—</span>'

        body_rows.append(f"""
        <tr>
            <td class="tx-time">{row['tx_time']}</td>
            <td class="title-cell">{row['title']}{genre_tag(genre_display)}</td>
            <td class="duration-cell">{row['duration']}</td>
            <td class="type-cell">{row['promo_type_display']}</td>
            <td>{flag_badge(sm, proba)}</td>
            <td class="action-cell">{action}</td>
            <td>{uplift_str}</td>
            <td class="rationale-cell">{rationale}</td>
        </tr>""")

    return f"""
    <div class="sched-wrap">
        <table class="sched-table">
            <thead><tr>{header}</tr></thead>
            <tbody>{"".join(body_rows)}</tbody>
        </table>
    </div>"""


# ---------------------------------------------------------------------------
# Data loading (cached per refresh cycle)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=REFRESH_INTERVAL_SEC, show_spinner=False)
def load_data(refresh_key: int):
    api_live = check_api_health()
    preds = fetch_predictions(MOCK_SCHEDULE) if api_live else local_predictions(MOCK_SCHEDULE)
    if not preds:
        preds = local_predictions(MOCK_SCHEDULE)

    enriched = []
    for row, pred in zip(MOCK_SCHEDULE, preds):
        enriched.append({**row, **pred})
    return enriched, api_live


refresh_key = int(time.time() // REFRESH_INTERVAL_SEC)

if st.session_state.real_schedule:
    api_live = check_api_health()
    schedule_src = st.session_state.real_schedule
    preds = fetch_predictions(schedule_src) if api_live else local_predictions(schedule_src)
    if not preds:
        preds = local_predictions(schedule_src)
    data = [{**row, **pred} for row, pred in zip(schedule_src, preds)]
else:
    data, api_live = load_data(refresh_key)

# ---------------------------------------------------------------------------
# Derived stats
# ---------------------------------------------------------------------------

flags_today   = sum(1 for r in data if r["should_move"] == 1)
avg_conf      = sum(r["move_probability"] for r in data) / len(data) * 100
coverage_pct  = round(len(data) / (len(data) + 5) * 100)
last_flag_min = 23
today_str     = datetime.now().strftime("%a %d %b")

# Live weather (cached for 30 min to avoid hammering the API on every rerun)
@st.cache_data(ttl=1800, show_spinner=False)
def _live_weather_now(station: str) -> dict:
    try:
        from fetch_live_data import fetch_weather
        from datetime import date as _date
        return fetch_weather(station, _date.today())
    except Exception:
        return {}

_wx = _live_weather_now("heathrow")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(f"""
<div class="ancast-header">
    <div class="ancast-logo">
        <div class="logo-icon">A</div>
        Ancast Nowcasting™
    </div>
    <div class="channel-pill">
        <div class="dot"></div>
        {CHANNEL} &nbsp;▾
    </div>
    <div class="date-nav">
        <div class="arrow">◂</div>
        <span class="date-label">{today_str}</span>
        <div class="arrow">▸</div>
    </div>
    <div class="stat-block">
        <div class="stat-label">Flags Raised Today</div>
        <div class="stat-value">{flags_today}</div>
        <div class="stat-sub">Last 24h · 7-day trend</div>
    </div>
    <div class="stat-block">
        <div class="stat-label">Time Since Last Flag</div>
        <div class="stat-value">{last_flag_min} min</div>
        <div class="stat-sub"><span class="live-dot"></span> Recent · system {"live" if api_live else "offline"}</div>
    </div>
    <div class="stat-block">
        <div class="stat-label">Avg. Confidence Today</div>
        <div class="stat-value">{avg_conf:.0f}%</div>
        <div class="stat-sub">Across all current recs</div>
    </div>
    <div class="stat-block">
        <div class="stat-label">Coverage</div>
        <div class="stat-value">{coverage_pct}%</div>
        <div class="stat-sub">Of upcoming slots</div>
    </div>
    <div class="stat-block">
        <div class="stat-label">Live Weather (London)</div>
        <div class="stat-value" style="font-size:13px;line-height:1.4">{_wx.get('weather_tmax_c','?')}°C</div>
        <div class="stat-sub">{_wx.get('weather_rain_mm','?')}mm · {_wx.get('weather_sun_hours','?')}h sun</div>
    </div>
    <div class="header-spacer"></div>
    <div class="avatar">BA</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Real schedule upload (replaces mock data when a CSV is provided)
# ---------------------------------------------------------------------------

with st.expander("📂  Load real schedule data (with live enrichment)", expanded=False):
    st.markdown(
        "Upload your promo schedule CSV and optionally auto-enrich it with **live weather** "
        "(Open-Meteo, free) and **content metadata** (TMDB, free key). "
        "The only required column is `promo_title`. Add `datetime` and break columns for best results."
    )

    enrich_col1, enrich_col2 = st.columns([3, 2])
    with enrich_col1:
        sched_file = st.file_uploader(
            "Schedule CSV", type=["csv"], key="schedule_uploader",
            label_visibility="collapsed",
        )
    with enrich_col2:
        tmdb_key_input = st.text_input(
            "TMDB API key (optional)",
            type="password",
            placeholder="Get free at themoviedb.org/settings/api",
            help="Enables real content popularity & ratings data (netflix_popularity_score, netflix_rating_mean)",
        )
        enrich_weather = st.checkbox("Auto-enrich with live weather", value=True,
                                     help="Fetches real temperature, rain, and sun hours from Open-Meteo (no key needed)")

    col_load, col_clear = st.columns([3, 1])
    with col_load:
        if sched_file is not None:
            btn_label = "Enrich with live data + load as schedule" if enrich_weather else "Load as schedule"
            if st.button(btn_label, use_container_width=True, type="primary"):
                try:
                    sched_df = pd.read_csv(io.BytesIO(sched_file.getvalue()))
                    sched_df = sched_df.drop(columns=[c for c in _LEAKY_COLS if c in sched_df.columns])

                    if enrich_weather:
                        from fetch_live_data import enrich_schedule
                        with st.spinner("Fetching live weather + computing derived features..."):
                            sched_df = enrich_schedule(
                                sched_df,
                                tmdb_api_key=tmdb_key_input.strip() or None,
                                verbose=False,
                            )
                        st.success(f"Enriched with live weather data!")

                    st.session_state.real_schedule = csv_to_schedule_rows(sched_df)
                    st.success(f"Loaded {len(st.session_state.real_schedule):,} rows from {sched_file.name}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not load schedule: {exc}")
    with col_clear:
        if st.session_state.real_schedule:
            if st.button("Reset to sample data", use_container_width=True):
                st.session_state.real_schedule = None
                st.rerun()

    if st.session_state.real_schedule:
        st.info(f"Currently using real schedule: **{len(st.session_state.real_schedule):,} rows**")

# ---------------------------------------------------------------------------
# Advisory banner
# ---------------------------------------------------------------------------

st.markdown("""
<div class="advisory-banner">
    ℹ️ &nbsp;<strong>Advisory output.</strong>
    Ancast surfaces recommendations from live, context and historical signals.
    Schedule changes are executed by the broadcaster's playout system — operators decide whether to act.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Filter bar
# ---------------------------------------------------------------------------

dayparts = ["Breakfast", "Daytime", "Peak", "Late", "All day"]

col_tabs, col_div, col_ctype, col_genre, col_flag, col_sort, col_more, col_search = st.columns(
    [3.2, 0.15, 1.4, 1.2, 1.3, 1.2, 1.1, 1.8]
)

with col_tabs:
    # Render daypart tabs as styled buttons
    tab_html = '<div class="filter-bar" style="padding:8px 0;border:none;gap:4px;">'
    for dp in dayparts:
        active_cls = "active" if dp == st.session_state.daypart else ""
        tab_html += f'<span class="daypart-tab {active_cls}">{dp}</span>'
    tab_html += "</div>"
    st.markdown(tab_html, unsafe_allow_html=True)

    # Invisible selectbox to capture the chosen daypart
    chosen = st.selectbox("daypart_select", dayparts,
                          index=dayparts.index(st.session_state.daypart),
                          label_visibility="collapsed", key="daypart_sel")
    if chosen != st.session_state.daypart:
        st.session_state.daypart = chosen
        st.rerun()

with col_ctype:
    content_type = st.selectbox("Content type", ["Promos", "All content", "Programmes"],
                                label_visibility="collapsed")
with col_genre:
    genre_filter = st.selectbox("Genre", ["All", "History Doc", "General", "Promo"],
                                label_visibility="collapsed")
with col_flag:
    flag_filter = st.selectbox("Flag", ["Open only", "All", "On track"],
                               label_visibility="collapsed")
with col_sort:
    sort_by = st.selectbox("Sort", ["Time ↑", "Uplift ↓", "Confidence ↓"],
                           label_visibility="collapsed")
with col_more:
    st.markdown('<button class="filter-pill" style="margin-top:2px">More filters ▾</button>',
                unsafe_allow_html=True)
with col_search:
    search_q = st.text_input("Search", placeholder="Search title...", label_visibility="collapsed")

# ---------------------------------------------------------------------------
# Filter + sort data
# ---------------------------------------------------------------------------

filtered = data

if st.session_state.daypart != "All day":
    filtered = [r for r in filtered if r["daypart"] == st.session_state.daypart]

if genre_filter != "All":
    filtered = [r for r in filtered if r["genre_display"] == genre_filter]

if flag_filter == "Open only":
    filtered = [r for r in filtered if r["should_move"] == 1]
elif flag_filter == "On track":
    filtered = [r for r in filtered if r["should_move"] == 0]

if search_q:
    q = search_q.lower()
    filtered = [r for r in filtered if q in r["title"].lower()]

if sort_by == "Uplift ↓":
    filtered = sorted(filtered, key=lambda r: r["predicted_uplift"], reverse=True)
elif sort_by == "Confidence ↓":
    filtered = sorted(filtered, key=lambda r: r["move_probability"], reverse=True)

# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

st.markdown(build_table_html(filtered), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

elapsed = int(time.time() - (refresh_key * REFRESH_INTERVAL_SEC))
refresh_in = REFRESH_INTERVAL_SEC - elapsed
api_badge = "🟢 API live" if api_live else "🔴 API offline (mock data)"
flags_shown = sum(1 for r in filtered if r["should_move"] == 1)
total_flags = flags_today

st.markdown(f"""
<div class="sched-footer">
    <button class="density-btn active">Comfortable</button>
    <button class="density-btn">Compact</button>
    &nbsp;Rich detail &nbsp;<span style="font-size:15px">●</span>
    <div class="footer-spacer"></div>
    <span>{api_badge}</span>
    &nbsp;·&nbsp;
    <span class="refresh-info">
        {flags_shown} of {total_flags} flags shown · refreshes in {refresh_in}s
    </span>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Single prediction form
# ---------------------------------------------------------------------------

st.markdown("""
<div style="margin:24px 28px 0 28px; border-top:2px solid #ddd8cc; padding-top:20px;
     font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
  <div style="font-size:16px;font-weight:700;color:#1a1f38;margin-bottom:2px;">
    Try a single prediction
  </div>
  <div style="font-size:12px;color:#888;margin-bottom:16px;">
    Fill in the promo details below and click Predict to get an instant recommendation.
  </div>
</div>
""", unsafe_allow_html=True)

KNOWN_TITLES = [
    "Alexander's Lost World - Tue @ 2100",
    "Europe's Last Warrior Kings - Wed @ 2100",
    "Queen Victoria's Letters: A Monarch - Thurs @ 2150",
    "Annihilation: The Destruction - Tonight @ 2250",
    "Promotion",
    "Animated Ident 1 - Napoleon HD cutdown 10",
    "Animated Ident 2 - Pushkin HD cutdown 10",
]

with st.form("single_predict_form", clear_on_submit=False):

    st.markdown("#### Promo identity")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        f_title = st.selectbox("Promo title", KNOWN_TITLES)
    with fc2:
        f_content_type = st.selectbox("Content type", ["Promo", "Jingle"])
    with fc3:
        f_genre = st.selectbox("Genre", ["history_doc", "general", "promo"])

    st.markdown("#### Schedule")
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        f_hour = st.slider("Hour of day", 0, 23, 21)
    with sc2:
        f_dow = st.selectbox("Day of week", [
            "0 — Monday", "1 — Tuesday", "2 — Wednesday",
            "3 — Thursday", "4 — Friday", "5 — Saturday", "6 — Sunday"
        ])
        f_day_of_week = int(f_dow[0])
    with sc3:
        f_is_weekend = st.selectbox("Weekend?", ["No (0)", "Yes (1)"])
        f_is_weekend_val = 1 if "Yes" in f_is_weekend else 0
    with sc4:
        f_promo_len = st.number_input("Promo length (sec)", min_value=5, max_value=120, value=20, step=5)

    st.markdown("#### Break placement")
    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1:
        f_in_break = st.selectbox("In break?", ["Yes (1)", "No (0)"])
        f_in_break_val = 1 if "Yes" in f_in_break else 0
    with bc2:
        f_pos_type = st.selectbox("Position type", ["break_middle", "break_end", "programme_lead_in"])
    with bc3:
        f_break_pos = st.slider("Break position %", 0.0, 1.0, 0.5, step=0.05,
                                help="0 = first slot, 1 = last slot")
    with bc4:
        f_break_ev = st.number_input("Event position", 1, 10, 2)
        f_break_tot = st.number_input("Total events", 1, 10, 4)

    st.markdown("#### Promo health signals")
    hc1, hc2 = st.columns(2)
    with hc1:
        f_fatigue = st.slider("Promo fatigue index", 0.0, 1.0, 0.5, step=0.05,
                              help="How worn-out the audience is with this promo. Higher = more fatigued.")
    with hc2:
        f_attention = st.slider("Attention context score", 0.0, 1.0, 0.5, step=0.05,
                                help="How receptive the slot audience is. Lower = less attention.")

    st.markdown("#### Weather")
    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        f_rain = st.number_input("Rain (mm)", 0.0, 50.0, 0.0, step=0.5)
    with wc2:
        f_sun = st.number_input("Sun hours", 0.0, 16.0, 4.0, step=0.5)
    with wc3:
        f_indoor = st.slider("Indoor viewing index", 0.0, 1.0, 0.4, step=0.05,
                             help="Higher = more people likely to be indoors watching TV.")

    submitted = st.form_submit_button(
        "Predict",
        use_container_width=True,
        type="primary",
    )

if submitted:
    payload = {
        "channel":               "Viasat History CEE",
        "promo_title":           f_title,
        "content_type":          f_content_type,
        "genre_guess":           f_genre,
        "promo_length_seconds":  f_promo_len,
        "hour":                  f_hour,
        "day_of_week":           f_day_of_week,
        "is_weekend":            f_is_weekend_val,
        "promo_position_type":   f_pos_type,
        "promo_in_break":        f_in_break_val,
        "break_event_position":  int(f_break_ev),
        "break_total_events":    int(f_break_tot),
        "break_position_pct":    f_break_pos,
        "weather_station":       "hurn",
        "weather_rain_mm":       f_rain,
        "weather_sun_hours":     f_sun,
        "weather_indoor_viewing_index": f_indoor,
        "promo_fatigue_index":   f_fatigue,
        "attention_context_score": f_attention,
    }
    try:
        if api_live:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()
        else:
            result = local_predictions([payload])[0]

        sm     = result["should_move"]
        prob   = result["move_probability"]
        uplift = result["predicted_uplift"]

        if sm == 1 and prob >= 0.70:
            banner_bg, banner_border, banner_text = "#ffebeb", "#cc2200", "#cc2200"
            flag_label, icon = "Review — swap", "🔴"
        elif sm == 1:
            banner_bg, banner_border, banner_text = "#fff5e0", "#cc7700", "#cc7700"
            flag_label, icon = "Review — move", "🟡"
        else:
            banner_bg, banner_border, banner_text = "#e8faf0", "#117733", "#117733"
            flag_label, icon = "On track — keep", "🟢"

        st.markdown(f"""
        <div style="background:{banner_bg};border:2px solid {banner_border};border-radius:10px;
             padding:20px 28px;margin:12px 0;
             font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
          <div style="font-size:20px;font-weight:800;color:{banner_text};margin-bottom:8px;">
            {icon} &nbsp; {flag_label}
          </div>
          <div style="font-size:13px;color:#333;">
            <b>Promo:</b> {f_title}
            &nbsp;&nbsp;|&nbsp;&nbsp;
            <b>Slot:</b> {f_hour:02d}:00 &nbsp; · &nbsp; Break position {f_break_pos*100:.0f}%
          </div>
        </div>
        """, unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        r1.metric("Should Move",      "Yes" if sm else "No")
        r2.metric("Move Probability", f"{prob*100:.1f}%")
        r3.metric("Predicted Uplift", f"+{uplift:.4f}" if sm else "—")

        with st.expander("Show payload"):
            st.json(payload)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------------------------------------------------------------------
# CSV Upload section
# ---------------------------------------------------------------------------

st.markdown("""
<div style="margin:24px 28px 0 28px; border-top:2px solid #ddd8cc; padding-top:20px;
     font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
  <div style="font-size:14px;font-weight:700;color:#1a1f38;margin-bottom:4px;">
    Upload your own schedule CSV
  </div>
  <div style="font-size:12px;color:#888;margin-bottom:12px;">
    Upload any CSV that contains promo scheduling data — predictions are appended
    and the result downloads automatically.
  </div>
</div>
""", unsafe_allow_html=True)

up_col1, up_col2 = st.columns([3, 2])

with up_col1:
    uploaded_file = st.file_uploader(
        "Drop your CSV here",
        type=["csv"],
        label_visibility="collapsed",
        help="CSV must contain at least some of the 37 model feature columns. "
             "Missing columns are treated as NaN automatically.",
    )

with up_col2:
    st.markdown("""
    <div style="font-size:11px;color:#666;padding-top:8px;line-height:1.7">
    <b>Required columns (at least one):</b><br>
    hour &nbsp;·&nbsp; promo_in_break &nbsp;·&nbsp; break_position_pct<br>
    promo_fatigue_index &nbsp;·&nbsp; attention_context_score<br><br>
    <b>Optional (improves accuracy):</b><br>
    channel &nbsp;·&nbsp; promo_title &nbsp;·&nbsp; genre_guess<br>
    weather_rain_mm &nbsp;·&nbsp; ott_hook_strength
    </div>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    with st.spinner(f"Processing {uploaded_file.name}…"):
        try:
            df_up = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
            df_up = df_up.drop(columns=[c for c in _LEAKY_COLS if c in df_up.columns])

            if api_live:
                response = requests.post(
                    f"{API_URL}/predict/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                    timeout=60,
                )
                response.raise_for_status()
                preview_df = pd.read_csv(io.BytesIO(response.content))
                result_bytes = response.content
            else:
                rows = df_up.to_dict(orient="records")
                preds = local_predictions(rows)
                preds_df = pd.DataFrame(preds)
                preview_df = pd.concat([df_up.reset_index(drop=True), preds_df], axis=1)
                result_bytes = preview_df.to_csv(index=False).encode()

            total   = len(preview_df)
            flagged = int(preview_df["should_move"].sum())
            out_name = uploaded_file.name.replace(".csv", "_predictions.csv")

            st.success(f"Done — {total:,} rows processed, **{flagged}** flagged ({flagged/total*100:.1f}%)")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total rows",     f"{total:,}")
            m2.metric("Should move",    f"{flagged:,}")
            m3.metric("Avg confidence", f"{preview_df['move_probability'].mean()*100:.1f}%")

            display_cols = [c for c in
                ["datetime", "promo_title", "hour", "should_move", "move_probability", "predicted_uplift"]
                if c in preview_df.columns]
            st.dataframe(preview_df[display_cols].head(20), use_container_width=True, hide_index=True)

            st.download_button(
                label="Download full predictions CSV",
                data=result_bytes,
                file_name=out_name,
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Upload failed: {e}")

# ---------------------------------------------------------------------------
# Retrain Model
# ---------------------------------------------------------------------------

st.markdown("""
<div style="margin:24px 28px 0 28px; border-top:2px solid #ddd8cc; padding-top:20px;
     font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
  <div style="font-size:16px;font-weight:700;color:#1a1f38;margin-bottom:2px;">
    Retrain / Update the Model
  </div>
  <div style="font-size:12px;color:#888;margin-bottom:16px;">
    Upload a <strong>labeled</strong> CSV (must include <code>should_move</code> and
    <code>uplift_if_optimised</code> columns) to update the model with real data.
    Choose <em>Incremental</em> to add trees to the existing model (fast, good for
    small new batches) or <em>Full retrain</em> to rebuild from scratch.
  </div>
</div>
""", unsafe_allow_html=True)

with st.expander("🔁  Retrain model with labeled data", expanded=False):
    rt_col1, rt_col2 = st.columns([3, 2])

    with rt_col1:
        train_file = st.file_uploader(
            "Labeled training CSV",
            type=["csv"], key="train_uploader",
            label_visibility="collapsed",
            help="Must include should_move (0/1) and uplift_if_optimised (float) columns.",
        )

    with rt_col2:
        retrain_mode = st.radio(
            "Training mode",
            ["Incremental (add trees)", "Full retrain from scratch"],
            index=0,
            help=(
                "Incremental: warm-start the existing model — fast, good for small updates.\n"
                "Full retrain: rebuild the model from this data only — slower but resets distribution."
            ),
        )
        n_trees = st.number_input(
            "New trees (incremental only)", min_value=10, max_value=500, value=100, step=10,
            help="Ignored for full retrain.",
        )

    if train_file is not None:
        is_incremental = "Incremental" in retrain_mode
        btn_label = "Start incremental update" if is_incremental else "Start full retrain"
        if st.button(btn_label, use_container_width=True, type="primary"):
            if api_live:
                try:
                    resp = requests.post(
                        f"{API_URL}/retrain/upload",
                        files={"file": (train_file.name, train_file.getvalue(), "text/csv")},
                        params={"incremental": str(is_incremental).lower(), "n_new_trees": int(n_trees)},
                        timeout=15,
                    )
                    resp.raise_for_status()
                    info = resp.json()
                    st.success(
                        f"Training started in background — {info['rows']:,} rows, "
                        f"{'incremental' if info['incremental'] else 'full retrain'}. "
                        f"Check status below."
                    )
                except Exception as exc:
                    st.error(f"Could not start retraining: {exc}")
            else:
                # API is offline — run training locally in-process
                try:
                    from trainer import train_models, LEAKY_COLUMNS as _LEAKY
                    from pathlib import Path as _Path
                    import pandas as _pd

                    _df = _pd.read_csv(io.BytesIO(train_file.getvalue()))
                    missing_lbl = [c for c in ["should_move", "uplift_if_optimised"] if c not in _df.columns]
                    if missing_lbl:
                        st.error(f"CSV missing required label columns: {missing_lbl}")
                    else:
                        with st.spinner("Training in progress (API offline — running locally)..."):
                            metrics = train_models(
                                _df, _Path("model"),
                                incremental=is_incremental,
                                n_new_trees=int(n_trees),
                            )
                        st.success(
                            f"Training complete!  PR-AUC={metrics['pr_auc']}  "
                            f"R²={metrics['r2']}  rows={metrics['rows_trained']:,}. "
                            "Reload the page to use the updated model."
                        )
                        st.json(metrics)
                except Exception as exc:
                    st.error(f"Local training failed: {exc}")

    # Status check
    if api_live:
        if st.button("Refresh training status", use_container_width=False):
            try:
                status_resp = requests.get(f"{API_URL}/retrain/status", timeout=5)
                status_resp.raise_for_status()
                s = status_resp.json()
                if s["status"] == "running":
                    st.info(f"Training in progress... (started {s['started_at']})")
                elif s["status"] == "done":
                    st.success(f"Last training completed at {s['finished_at']}")
                    if s["metrics"]:
                        st.json(s["metrics"])
                elif s["status"] == "error":
                    st.error(f"Last training failed: {s['error']}")
                else:
                    st.write("No training job has been run yet in this session.")
            except Exception as exc:
                st.warning(f"Could not fetch status: {exc}")

    st.markdown("""
    <div style="font-size:11px;color:#888;margin-top:8px;line-height:1.7">
    <b>Running without the API?</b> You can also train from the command line:<br>
    <code>python train_local.py --data your_labeled_data.csv</code><br>
    <code>python train_local.py --data new_batch.csv --incremental</code>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Manual refresh
# ---------------------------------------------------------------------------

st.markdown(
    "<div style='margin:8px 28px;font-size:11px;color:#aaa'>"
    f"Last refreshed: {datetime.now().strftime('%H:%M:%S')} &nbsp;·&nbsp; "
    "Predictions refresh automatically when you interact with any filter above."
    "</div>",
    unsafe_allow_html=True,
)
if st.button("↻  Refresh predictions now", use_container_width=False):
    st.cache_data.clear()
    st.rerun()

"""
config.py
=========
Single source of truth for constants shared across the whole project.

Every other file imports from here. Never duplicate these lists.
"""

# Columns that describe what happened AFTER the promo aired.
# Available at reporting time, NOT at scheduling time — including them causes
# data leakage. Drop these before training and before sending to the model.
LEAKY_COLUMNS: list[str] = [
    "actual_status",
    "actual_start_offset_seconds",
    "actual_duration_seconds",
    "duration_diff_seconds",
    "was_missed",
    "observed_effectiveness_score",
    "best_possible_score",
]

# Classification target: 1 = promo should be repositioned, 0 = keep as-is
TARGET_CLS: str = "should_move"

# Regression target: expected effectiveness uplift if the promo is moved
TARGET_REG: str = "uplift_if_optimised"

# Columns that identify a row but are not predictive features
ID_META: list[str] = ["date", "datetime", "house_number"]

# Post-prediction recommendation columns (not features)
RECOMMENDATION_COLS: list[str] = [
    "recommended_position",
    "recommended_offset_minutes",
]

# The 37 feature columns the model was trained on.
# This list is the ground truth — all other files load from here or from
# inference_schema.json (which is generated from this list during training).
FEATURE_COLS: list[str] = [
    # Identity / categorical
    "channel",
    "promo_title",
    "content_type",
    "event_type",
    "genre_guess",
    # Scheduling numerics
    "promo_length_seconds",
    "hour",
    "minute",
    "day_of_week",
    "is_weekend",
    "time_band",
    # Break placement
    "promo_position_type",
    "promo_in_break",
    "break_event_position",
    "break_total_events",
    "break_position_pct",
    "preceded_by_break",
    "lead_to_next_program_min",
    # Neighbour promos
    "prev_promo_title",
    "next_promo_title",
    # Weather
    "weather_station",
    "weather_tmax_c",
    "weather_tmin_c",
    "weather_rain_mm",
    "weather_sun_hours",
    "weather_bad_index",
    "weather_indoor_viewing_index",
    # OTT / streaming signals
    "ott_avg_watch_pct",
    "ott_dropoff_prob",
    "ott_hook_strength",
    "ott_visual_intensity",
    # Competitive
    "netflix_popularity_score",
    "netflix_rating_mean",
    # Synthetic / derived
    "synthetic_premiere_probability",
    "synthetic_production_year_mean",
    "promo_fatigue_index",
    "attention_context_score",
]

# Context columns kept in batch prediction output but NOT sent to the model
PASSTHROUGH_COLS: list[str] = ["datetime", "house_number", TARGET_CLS, TARGET_REG]

RANDOM_STATE: int = 42

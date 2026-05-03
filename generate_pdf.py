"""
generate_pdf.py
===============
Generates a comprehensive project documentation PDF.

Run:
    python generate_pdf.py
Output:
    CTV_Promo_Project_Documentation.pdf
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import date

OUTPUT = "CTV_Promo_Project_Documentation.pdf"
PAGE_W, PAGE_H = A4

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY      = colors.HexColor("#1a1f38")
BLUE      = colors.HexColor("#3a8ef6")
LIGHT_BG  = colors.HexColor("#f0ebe0")
CARD_BG   = colors.HexColor("#e8f0fe")
GREEN     = colors.HexColor("#117733")
RED       = colors.HexColor("#cc2200")
GREY      = colors.HexColor("#555555")
LGREY     = colors.HexColor("#dddddd")
WHITE     = colors.white
BLACK     = colors.HexColor("#1a1a2a")
ORANGE    = colors.HexColor("#cc7700")

# ── Styles ────────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

styles = {
    "cover_title": S("cover_title",
        fontSize=28, fontName="Helvetica-Bold",
        textColor=WHITE, alignment=TA_CENTER, leading=34),
    "cover_sub": S("cover_sub",
        fontSize=13, fontName="Helvetica",
        textColor=colors.HexColor("#aabbff"), alignment=TA_CENTER, leading=18),
    "cover_meta": S("cover_meta",
        fontSize=10, fontName="Helvetica",
        textColor=colors.HexColor("#8899cc"), alignment=TA_CENTER),
    "h1": S("h1",
        fontSize=18, fontName="Helvetica-Bold",
        textColor=NAVY, spaceBefore=18, spaceAfter=6, leading=22),
    "h2": S("h2",
        fontSize=13, fontName="Helvetica-Bold",
        textColor=BLUE, spaceBefore=12, spaceAfter=4, leading=17),
    "h3": S("h3",
        fontSize=11, fontName="Helvetica-Bold",
        textColor=GREY, spaceBefore=8, spaceAfter=3, leading=14),
    "body": S("body",
        fontSize=10, fontName="Helvetica",
        textColor=BLACK, leading=15, spaceAfter=5, alignment=TA_JUSTIFY),
    "bullet": S("bullet",
        fontSize=10, fontName="Helvetica",
        textColor=BLACK, leading=14, leftIndent=16, spaceAfter=3,
        bulletIndent=6),
    "code": S("code",
        fontSize=8.5, fontName="Courier",
        textColor=colors.HexColor("#1a3a6a"),
        backColor=colors.HexColor("#eef2ff"),
        leading=12, leftIndent=10, rightIndent=10,
        spaceBefore=4, spaceAfter=4),
    "note": S("note",
        fontSize=9, fontName="Helvetica-Oblique",
        textColor=colors.HexColor("#3355aa"),
        backColor=CARD_BG,
        leading=13, leftIndent=10, rightIndent=10,
        spaceBefore=4, spaceAfter=6),
    "caption": S("caption",
        fontSize=9, fontName="Helvetica-Oblique",
        textColor=GREY, alignment=TA_CENTER),
}

def HR():
    return HRFlowable(width="100%", thickness=1, color=LGREY, spaceAfter=6, spaceBefore=4)

def GAP(h=6):
    return Spacer(1, h)

def H1(text):  return Paragraph(text, styles["h1"])
def H2(text):  return Paragraph(text, styles["h2"])
def H3(text):  return Paragraph(text, styles["h3"])
def P(text):   return Paragraph(text, styles["body"])
def B(text):   return Paragraph(f"• &nbsp; {text}", styles["bullet"])
def Code(text): return Paragraph(text.replace("\n","<br/>").replace(" ","&nbsp;"), styles["code"])
def Note(text): return Paragraph(f"ℹ &nbsp; {text}", styles["note"])

def table(data, col_widths, header=True):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    cmds = [
        ("FONTNAME",  (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",  (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [WHITE, colors.HexColor("#f7f5f0")]),
        ("GRID",      (0,0), (-1,-1), 0.4, LGREY),
        ("LEFTPADDING",  (0,0), (-1,-1), 7),
        ("RIGHTPADDING", (0,0), (-1,-1), 7),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("VALIGN",    (0,0), (-1,-1), "TOP"),
    ]
    if header:
        cmds += [
            ("BACKGROUND",  (0,0), (-1,0), NAVY),
            ("TEXTCOLOR",   (0,0), (-1,0), WHITE),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,0), 9),
        ]
    t.setStyle(TableStyle(cmds))
    return t


# ── Cover page builder ────────────────────────────────────────────────────────
def cover_page():
    """Returns a list of flowables for the cover page."""
    from reportlab.platypus import Flowable

    class ColorRect(Flowable):
        def __init__(self, w, h, fill):
            Flowable.__init__(self)
            self.width, self.height = w, h
            self.fill = fill
        def draw(self):
            self.canv.setFillColor(self.fill)
            self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=0)

    story = []
    story.append(ColorRect(PAGE_W - 4*cm, 4.5*cm, NAVY))
    story.append(GAP(8))
    story.append(Paragraph("CTV Promo Placement<br/>Optimisation", styles["cover_title"]))
    story.append(GAP(10))
    story.append(Paragraph("End-to-End ML System — Project Documentation", styles["cover_sub"]))
    story.append(GAP(20))
    story.append(HR())
    story.append(GAP(8))
    story.append(Paragraph(
        f"Viasat History CEE &nbsp;|&nbsp; XGBoost · FastAPI · Streamlit &nbsp;|&nbsp; {date.today().strftime('%d %B %Y')}",
        styles["cover_meta"]))
    story.append(GAP(30))
    return story


# ── Document sections ─────────────────────────────────────────────────────────
def build_story():
    s = []

    # ── Cover ─────────────────────────────────────────────────────────────────
    s += cover_page()
    s.append(PageBreak())

    # ── 1. Overview ───────────────────────────────────────────────────────────
    s.append(H1("1. Project Overview"))
    s.append(HR())
    s.append(P(
        "This project builds an intelligent TV promo scheduling system for Viasat History CEE. "
        "It uses seven years of broadcast scheduling data to train two machine learning models: "
        "a <b>classifier</b> that predicts whether a promo should be repositioned in its break, "
        "and a <b>regressor</b> that estimates how much effectiveness uplift that move would deliver. "
        "The models are served through a production-ready <b>FastAPI</b> REST service and visualised "
        "in a <b>Streamlit</b> dashboard that replicates the Ancast Nowcasting operator interface."
    ))
    s.append(GAP(8))
    s.append(Note(
        "Advisory output only. Recommendations are surfaced to operators who decide whether to act. "
        "No schedule changes are made automatically."
    ))
    s.append(GAP(10))

    arch_data = [
        ["Layer", "File", "Purpose"],
        ["Training pipeline", "ctv_promo_pipeline.py", "Loads data, trains models, saves artifacts"],
        ["ML artifacts",      "model/",                "promo_move_classifier.pkl, promo_uplift_regressor.pkl, inference_schema.json"],
        ["REST API",          "main.py",               "FastAPI service — /predict, /predict/batch, /reload, /health"],
        ["Frontend",          "streamlit_app.py",      "Ancast Nowcasting operator UI"],
        ["Schema recovery",   "generate_schema.py",    "Extracts inference_schema.json from pkl files when missing"],
        ["Tests",             "tests/test_api.py",     "20 pytest unit tests with mocked models"],
        ["Container",         "Dockerfile / docker-compose.yml", "Containerised deployment"],
    ]
    s.append(table(arch_data, [3.5*cm, 5*cm, 8.5*cm]))
    s.append(GAP(12))

    # ── 2. Problem Statement ──────────────────────────────────────────────────
    s.append(H1("2. Problem Statement"))
    s.append(HR())
    s.append(P(
        "TV promos compete for limited break slots. A promo placed at the wrong position in a break "
        "(e.g., third of four when first would perform better) loses audience impact. "
        "Schedulers currently make these decisions manually, often without access to real-time "
        "signals like weather, streaming competition, or audience fatigue."
    ))
    s.append(GAP(6))
    s.append(H2("Two prediction tasks"))
    s.append(B("<b>Classification</b> — should_move: should this promo be repositioned? (0 = keep, 1 = move)"))
    s.append(B("<b>Regression</b> — uplift_if_optimised: if moved, how much effectiveness gain is expected?"))
    s.append(GAP(8))
    s.append(H2("Why this is hard"))
    s.append(B("Class imbalance: roughly 80% of promos should be kept, only 20% should move."))
    s.append(B("Data leakage risk: post-broadcast outcome columns (actual position, was_missed) must never be used as features."))
    s.append(B("Temporal drift: 7 years of data spans COVID disruption and a streaming revolution — a random train/test split would cheat."))
    s.append(B("Categorical cardinality: promo_title and neighbour titles have hundreds of unique values with no natural ordering."))
    s.append(GAP(12))

    # ── 3. Dataset ────────────────────────────────────────────────────────────
    s.append(H1("3. Dataset"))
    s.append(HR())
    s.append(P(
        "The training dataset covers 7 years of Viasat History CEE broadcast scheduling. "
        "Each row represents one promo event at a specific time in a specific break."
    ))
    s.append(GAP(6))

    feat_data = [
        ["Feature Group", "Feature Names", "Type"],
        ["Identity / title",    "channel, promo_title, content_type, event_type, genre_guess", "Categorical"],
        ["Scheduling",          "hour, minute, day_of_week, is_weekend, time_band", "Numeric / Cat"],
        ["Break placement",     "promo_position_type, promo_in_break, break_event_position,\nbreak_total_events, break_position_pct, preceded_by_break,\nlead_to_next_program_min", "Numeric / Cat"],
        ["Neighbour context",   "prev_promo_title, next_promo_title", "Categorical"],
        ["Weather",             "weather_station, weather_tmax_c, weather_tmin_c,\nweather_rain_mm, weather_sun_hours,\nweather_bad_index, weather_indoor_viewing_index", "Numeric / Cat"],
        ["OTT / streaming",     "ott_avg_watch_pct, ott_dropoff_prob,\nott_hook_strength, ott_visual_intensity", "Numeric"],
        ["Competitive",         "netflix_popularity_score, netflix_rating_mean", "Numeric"],
        ["Synthetic / derived", "synthetic_premiere_probability, synthetic_production_year_mean", "Numeric"],
        ["Promo health",        "promo_fatigue_index, attention_context_score", "Numeric"],
    ]
    s.append(table(feat_data, [3.5*cm, 9.5*cm, 4*cm]))
    s.append(GAP(6))
    s.append(P(
        "<b>Leaky columns removed (7):</b> actual_status, actual_start_offset_seconds, "
        "actual_duration_seconds, duration_diff_seconds, was_missed, "
        "observed_effectiveness_score, best_possible_score. "
        "These describe what happened after the promo aired and are not available at scheduling time."
    ))
    s.append(GAP(12))

    # ── 4. ML Pipeline ────────────────────────────────────────────────────────
    s.append(H1("4. ML Pipeline  (ctv_promo_pipeline.py)"))
    s.append(HR())

    s.append(H2("4.1  Eight critical fixes over the original prototype"))
    fixes = [
        ("FIX 1 — Drop leaky features",
         "7 post-hoc outcome columns removed. Using them would give perfect training accuracy "
         "but zero real-world performance because they don't exist at scheduling time."),
        ("FIX 2 — Time-based train/test split",
         "Data is sorted by datetime and split at the 80th percentile. The model trains on "
         "older data and is tested on the most recent 20%. A random split lets the model see "
         "future data during training — a form of leakage."),
        ("FIX 3 — Keep df_raw for inference",
         "The raw DataFrame (before encoding) is preserved. The original notebook encoded "
         "df in place, then re-encoded it at inference time, turning every categorical to UNKNOWN."),
        ("FIX 4 — Handle class imbalance",
         "scale_pos_weight = n_negative / n_positive (~4.18) is passed to XGBoost. "
         "Evaluation uses PR-AUC (area under the precision-recall curve) not accuracy, "
         "because accuracy is meaningless on imbalanced data."),
        ("FIX 5 — Regressor trained only on should_move=1 rows",
         "Training the regressor on all rows is misleading — 80% of rows have uplift ≈ 0 "
         "(the 'keep' rows), so predicting zero looks impressive but is useless. "
         "The regressor only sees rows where a move is actually recommended."),
        ("FIX 6 — XGBoost native categorical support",
         "Columns are cast to pandas category dtype and enable_categorical=True is set. "
         "LabelEncoder creates false ordinal relationships (title_id=7 is not 'between' 6 and 8)."),
        ("FIX 7 — Early stopping",
         "n_estimators=1000 with early_stopping_rounds=30. Training stops when the validation "
         "PR-AUC stops improving for 30 consecutive rounds, preventing overfitting."),
        ("FIX 8 — No fillna(0) on numerics",
         "NaN values are left as-is. XGBoost learns a 'default direction' for each tree split "
         "when a feature is missing, which is statistically correct. Filling with 0 would "
         "conflate 'missing' with 'zero rain' or 'zero position'."),
    ]
    for title, desc in fixes:
        s.append(KeepTogether([
            Paragraph(f"<b>{title}</b>", styles["h3"]),
            P(desc),
            GAP(4),
        ]))

    s.append(H2("4.2  Classifier details"))
    s.append(P(
        "The classifier predicts <b>should_move</b> (0 or 1). It outputs a probability; "
        "the default decision threshold is 0.50, but this can be adjusted for precision/recall trade-offs."
    ))
    clf_params = [
        ["Parameter", "Value", "Why"],
        ["n_estimators", "1000 (capped by early stopping)", "Generous upper bound; early stopping finds the real optimal"],
        ["max_depth", "6", "Deep enough for interactions; shallow enough to generalise"],
        ["learning_rate", "0.05", "Slow learning = better generalisation than 0.1 or 0.3"],
        ["subsample", "0.9", "Stochastic boosting reduces overfitting"],
        ["colsample_bytree", "0.9", "Random feature subsets per tree"],
        ["min_child_weight", "3", "Prevents splits on tiny leaf groups"],
        ["scale_pos_weight", "~4.18", "Compensates for 80/20 class imbalance"],
        ["eval_metric", "aucpr", "PR-AUC is the correct metric for imbalanced classification"],
        ["enable_categorical", "True", "Native XGBoost categorical handling — no label encoding needed"],
        ["early_stopping_rounds", "30", "Stop if PR-AUC doesn't improve for 30 rounds"],
    ]
    s.append(table(clf_params, [4.5*cm, 5.5*cm, 7*cm]))
    s.append(GAP(8))

    s.append(H2("4.3  Regressor details"))
    s.append(P(
        "The regressor predicts <b>uplift_if_optimised</b> — the expected effectiveness gain "
        "if this promo is repositioned. It is trained <i>only</i> on rows where should_move=1 "
        "and is only called at inference time when the classifier also predicts should_move=1. "
        "Objective is reg:squarederror (MSE minimisation)."
    ))
    s.append(GAP(8))

    s.append(H2("4.4  Inference pipeline"))
    s.append(P("At prediction time, the following steps run for every incoming row:"))
    steps = [
        "1. Build a single-row (or batch) DataFrame from the JSON payload.",
        "2. Call prepare_for_inference(): select exactly the 37 training features, encode "
        "   categoricals against the training vocabulary (unseen values become NaN).",
        "3. Convert any object-dtype numeric columns to float (None → NaN).",
        "4. Classifier predicts move_probability.",
        "5. If move_probability ≥ 0.5 → should_move = 1 → regressor predicts uplift.",
        "6. If move_probability < 0.5 → should_move = 0 → uplift = 0.0 (regressor not called).",
        "7. Return should_move, move_probability, predicted_uplift.",
    ]
    for step in steps:
        s.append(B(step))
    s.append(GAP(12))
    s.append(PageBreak())

    # ── 5. FastAPI ────────────────────────────────────────────────────────────
    s.append(H1("5. FastAPI Service  (main.py)"))
    s.append(HR())
    s.append(P(
        "The API serves predictions over HTTP. It loads model artifacts once at startup "
        "and keeps them in memory for low-latency inference. All endpoints are documented "
        "automatically at http://localhost:8000/docs (Swagger UI)."
    ))
    s.append(GAP(8))

    s.append(H2("5.1  Endpoints"))
    ep_data = [
        ["Method", "Path", "Description"],
        ["GET",  "/health",         "Returns model load status, schema loaded flag, feature count. Returns 503 if models failed to load."],
        ["POST", "/predict",        "Single promo prediction. Accepts a JSON object with any subset of the 37 feature fields."],
        ["POST", "/predict/batch",  "Batch prediction for up to 500 promos in one request. Single DataFrame pass — much faster than looping /predict."],
        ["POST", "/reload",         "Hot-reload model artifacts from disk without restarting the server. Use after updating pkl files."],
    ]
    s.append(table(ep_data, [1.8*cm, 4.5*cm, 10.7*cm]))
    s.append(GAP(8))

    s.append(H2("5.2  Request / Response"))
    s.append(P("<b>Request body (/predict):</b>"))
    s.append(Code(
        '{\n'
        '  "channel": "Viasat History CEE",\n'
        '  "promo_title": "Alexander\'s Lost World - Tue @ 2100",\n'
        '  "genre_guess": "history_doc",\n'
        '  "hour": 21,\n'
        '  "promo_in_break": 1,\n'
        '  "break_position_pct": 0.5,\n'
        '  "promo_fatigue_index": 0.85,\n'
        '  "attention_context_score": 0.30\n'
        '}'
    ))
    s.append(P("<b>Response:</b>"))
    s.append(Code(
        '{\n'
        '  "promo_title": "Alexander\'s Lost World - Tue @ 2100",\n'
        '  "datetime": null,\n'
        '  "should_move": 1,\n'
        '  "move_probability": 0.7812,\n'
        '  "predicted_uplift": 0.3245\n'
        '}'
    ))
    s.append(GAP(8))

    s.append(H2("5.3  Key design decisions"))
    s.append(B("<b>All fields optional.</b> Missing fields become NaN; XGBoost handles them via learned default directions."))
    s.append(B("<b>Batch uses a single model call.</b> predict_proba() is called once on the full DataFrame, not once per row."))
    s.append(B("<b>Regressor gating.</b> The regressor is never called unless should_move=1, matching the training setup exactly."))
    s.append(B("<b>Atomic reload.</b> /reload only updates _state after both models load successfully, so a failed reload never breaks live traffic."))
    s.append(B("<b>Structured logging.</b> Every request logs method, path, status code, and latency in milliseconds."))
    s.append(B("<b>CORS enabled.</b> Configurable via CORS_ORIGINS environment variable."))
    s.append(GAP(8))

    s.append(H2("5.4  Environment variables"))
    env_data = [
        ["Variable", "Default", "Description"],
        ["MODEL_DIR",     "model",  "Path to directory containing pkl files and inference_schema.json"],
        ["CORS_ORIGINS",  "*",      "Comma-separated list of allowed CORS origins"],
        ["BATCH_LIMIT",   "500",    "Maximum number of items in a single /predict/batch request"],
    ]
    s.append(table(env_data, [4*cm, 2.5*cm, 10.5*cm]))
    s.append(GAP(12))

    # ── 6. Streamlit ──────────────────────────────────────────────────────────
    s.append(H1("6. Streamlit Frontend  (streamlit_app.py)"))
    s.append(HR())
    s.append(P(
        "The Streamlit app implements the Ancast Nowcasting operator interface — "
        "a scheduling dashboard that shows today's promo schedule enriched with "
        "live ML predictions. It calls /predict/batch on the FastAPI service and "
        "falls back to deterministic mock predictions when the API is unreachable."
    ))
    s.append(GAP(6))

    s.append(H2("6.1  UI components"))
    ui_data = [
        ["Component", "Description"],
        ["Header bar",       "Dark navy bar with channel selector, date navigation, and four live stat cards (flags raised, time since last flag, avg confidence, coverage %)."],
        ["Advisory banner",  "Blue info strip reminding operators that all output is advisory — no automatic schedule changes."],
        ["Daypart tabs",     "Breakfast / Daytime / Peak / Late / All day — filters the table to the selected part of the day."],
        ["Filter dropdowns", "Content type, genre, flag status (Open only / On track / All), sort order."],
        ["Search box",       "Live title search filtering rows as you type."],
        ["Schedule table",   "TX TIME, TITLE (with genre tag), DURATION, TYPE, FLAG badge, ACTION, UPLIFT, RATIONALE columns."],
        ["Flag badges",      "Review — swap (red, probability ≥ 0.70), Review — move (orange, probability 0.50–0.69), On track (green)."],
        ["Footer",           "API live/offline indicator, count of flags shown, auto-refresh countdown."],
    ]
    s.append(table(ui_data, [4*cm, 13*cm]))
    s.append(GAP(8))

    s.append(H2("6.2  Data flow"))
    s.append(B("1. On load, check /health to determine if the API is reachable."))
    s.append(B("2. If API live → POST /predict/batch with all 15 schedule items' real feature values."))
    s.append(B("3. If API offline → compute deterministic mock predictions locally (same formula every time so the UI is stable)."))
    s.append(B("4. Merge predictions back into the schedule rows."))
    s.append(B("5. Apply user-selected filters (daypart, genre, flag, search)."))
    s.append(B("6. Render the HTML table with coloured badges."))
    s.append(B("7. Auto-rerun every 30 seconds (st.rerun() + st.cache_data TTL)."))
    s.append(GAP(12))

    # ── 7. Supporting files ───────────────────────────────────────────────────
    s.append(H1("7. Supporting Files"))
    s.append(HR())

    s.append(H2("7.1  generate_schema.py"))
    s.append(P(
        "When inference_schema.json is missing (e.g., after copying pkl files from Kaggle without "
        "the schema), this script recovers it directly from the saved XGBoost model. "
        "XGBoost stores feature_names_in_ and feature_types_ on the model object. "
        "Run: <b>python generate_schema.py</b>. "
        "Limitation: category vocabularies cannot be recovered from pkl files alone — "
        "the schema will have empty category_values, meaning unseen strings pass through as NaN."
    ))
    s.append(GAP(6))

    s.append(H2("7.2  tests/test_api.py"))
    s.append(P(
        "20 pytest unit tests covering all endpoints. Models are mocked using unittest.mock.MagicMock "
        "so the test suite runs without any pkl files. Tests cover:"
    ))
    s.append(B("Health endpoint — OK response, 503 when not ready, schema-not-loaded flag."))
    s.append(B("Predict — minimal payload, response shape, passthrough fields, should_move=0/1 logic, regressor not called when should_move=0, field validation (hour out of range, break_position_pct > 1)."))
    s.append(B("Predict/batch — basic batch, empty batch rejected, batch limit enforced, per-item passthrough."))
    s.append(B("Reload — success path, failure returns 500."))
    s.append(P("Run: <b>pytest tests/</b>"))
    s.append(GAP(6))

    s.append(H2("7.3  Dockerfile + docker-compose.yml"))
    s.append(P(
        "The Dockerfile builds a Python 3.11-slim image. The model directory is volume-mounted "
        "in docker-compose.yml so pkl files can be updated without rebuilding the image. "
        "A built-in Docker healthcheck polls /health every 30 seconds. "
        "Two uvicorn workers are configured for better CPU utilisation with XGBoost inference."
    ))
    s.append(Code("docker compose up --build"))
    s.append(GAP(12))

    # ── 8. Missing values behaviour ───────────────────────────────────────────
    s.append(H1("8. How Missing Input Values Are Handled"))
    s.append(HR())
    s.append(P(
        "All 37 input fields are optional. When a field is not provided it becomes NaN "
        "in the inference DataFrame. XGBoost handles this through a mechanism called "
        "<b>learned default directions</b>."
    ))
    s.append(GAP(6))
    s.append(H2("How it works"))
    s.append(P(
        "During training, for every split in every tree XGBoost evaluates two choices: "
        "when this feature is NaN, send the row left or send it right? "
        "It picks whichever direction reduces training loss more. "
        "That choice is stored permanently in the model. At inference time, "
        "any missing feature follows that stored path through all 100+ trees automatically."
    ))
    s.append(GAP(6))
    s.append(H2("Feature priority tiers"))
    tier_data = [
        ["Tier", "Fields", "Impact if missing"],
        ["1 — Always provide",
         "promo_fatigue_index, attention_context_score,\nbreak_position_pct, hour, promo_in_break",
         "High — these drive most tree splits"],
        ["2 — Provide if available",
         "promo_title, genre_guess, break_event_position,\nweather_rain_mm, weather_indoor_viewing_index,\nott_hook_strength",
         "Medium — meaningful improvement in borderline cases"],
        ["3 — Nice to have",
         "netflix_popularity_score, synthetic_premiere_probability,\nweather_tmax_c, ott_dropoff_prob",
         "Low — marginal gain at the edges"],
    ]
    s.append(table(tier_data, [3.5*cm, 8*cm, 5.5*cm]))
    s.append(GAP(12))

    # ── 9. How to run ─────────────────────────────────────────────────────────
    s.append(H1("9. How to Run"))
    s.append(HR())

    s.append(H2("9.1  Local development"))
    s.append(Code(
        "# 1. Create and activate virtual environment\n"
        "python -m venv .venv\n"
        ".venv\\Scripts\\activate          # Windows\n"
        "source .venv/bin/activate        # Mac / Linux\n\n"
        "# 2. Install dependencies\n"
        "pip install -r requirements.txt\n\n"
        "# 3. (If inference_schema.json is missing) recover it\n"
        "python generate_schema.py\n\n"
        "# 4. Start the API\n"
        "uvicorn main:app --reload\n\n"
        "# 5. In a second terminal, start the UI\n"
        "streamlit run streamlit_app.py\n\n"
        "# 6. Run tests\n"
        "pytest tests/"
    ))
    s.append(GAP(8))

    s.append(H2("9.2  Docker"))
    s.append(Code(
        "# Build and start\n"
        "docker compose up --build\n\n"
        "# Update model without rebuild\n"
        "# 1. Replace pkl files in ./model/\n"
        "# 2. POST http://localhost:8000/reload"
    ))
    s.append(GAP(8))

    s.append(H2("9.3  Access points"))
    access_data = [
        ["Service",    "URL"],
        ["API",        "http://localhost:8000"],
        ["Swagger UI", "http://localhost:8000/docs"],
        ["Streamlit",  "http://localhost:8501"],
    ]
    s.append(table(access_data, [4*cm, 13*cm]))
    s.append(GAP(12))

    # ── 10. File structure ────────────────────────────────────────────────────
    s.append(H1("10. File Structure"))
    s.append(HR())
    s.append(Code(
        "ITV_PROMO/\n"
        "├── ctv_promo_pipeline.py       # Training pipeline\n"
        "├── main.py                     # FastAPI service\n"
        "├── streamlit_app.py            # Streamlit frontend\n"
        "├── generate_schema.py          # Schema recovery tool\n"
        "├── generate_pdf.py             # This documentation generator\n"
        "├── requirements.txt            # Python dependencies\n"
        "├── Dockerfile\n"
        "├── docker-compose.yml\n"
        "├── .env.example\n"
        "├── model/\n"
        "│   ├── promo_move_classifier.pkl\n"
        "│   ├── promo_uplift_regressor.pkl\n"
        "│   ├── inference_schema.json\n"
        "│   ├── eda.png\n"
        "│   └── feature_importance.png\n"
        "├── tests/\n"
        "│   ├── __init__.py\n"
        "│   └── test_api.py\n"
        "└── .venv/                      # Virtual environment\n"
    ))
    s.append(GAP(12))

    # ── 11. Glossary ──────────────────────────────────────────────────────────
    s.append(H1("11. Glossary"))
    s.append(HR())
    gloss = [
        ["Term", "Definition"],
        ["should_move",           "Binary label: 1 = this promo would perform better if repositioned in its break."],
        ["uplift_if_optimised",   "Continuous label: expected effectiveness score improvement if the promo is repositioned."],
        ["PR-AUC",                "Area under the Precision-Recall curve. The preferred metric for imbalanced binary classification."],
        ["scale_pos_weight",      "XGBoost hyperparameter = n_negative / n_positive. Compensates for class imbalance by up-weighting the minority class during training."],
        ["enable_categorical",    "XGBoost flag that enables native handling of pandas category dtype columns, avoiding the need for label encoding."],
        ["Default direction",     "The branch (left or right) that XGBoost takes at a tree split when the feature value is NaN. Learned from training data."],
        ["inference_schema.json", "JSON file storing the training feature list, categorical column names, and per-category vocabulary. Required for correct inference encoding."],
        ["promo_fatigue_index",   "Score (0–1) measuring how 'worn out' the audience is with a particular promo. Higher = more fatigued = more likely should_move=1."],
        ["attention_context_score","Score (0–1) measuring how receptive the audience in that slot is likely to be. Lower = less attention = more likely should_move=1."],
        ["break_position_pct",    "Float 0–1 representing the promo's relative position within its break. 0=first, 1=last."],
        ["OTT",                   "Over-the-top streaming (Netflix, Amazon, etc.). OTT signals reflect streaming competition that can affect linear TV viewing."],
    ]
    s.append(table(gloss, [5*cm, 12*cm]))
    s.append(GAP(20))

    s.append(HR())
    s.append(Paragraph(
        f"Generated {date.today().strftime('%d %B %Y')} · CTV Promo Placement Optimisation · Viasat History CEE",
        styles["caption"]
    ))

    return s


# ── Build PDF ─────────────────────────────────────────────────────────────────
def main():
    doc = SimpleDocTemplate(
        OUTPUT,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="CTV Promo Placement Optimisation — Project Documentation",
        author="Ancast Nowcasting",
    )

    def add_page_number(canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(GREY)
        canvas.drawString(2*cm, 1.2*cm, "CTV Promo Placement Optimisation")
        canvas.drawRightString(PAGE_W - 2*cm, 1.2*cm, f"Page {doc.page}")
        canvas.restoreState()

    story = build_story()
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"PDF saved: {OUTPUT}")


if __name__ == "__main__":
    main()

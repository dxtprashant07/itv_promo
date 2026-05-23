"""
predict_from_csv.py
===================
Reads the training CSV (or any CSV in the same format), calls the FastAPI
/predict/batch endpoint in chunks, and saves the results to a new CSV.

Usage:
    python predict_from_csv.py --input path/to/data.csv
    python predict_from_csv.py --input path/to/data.csv --output results.csv
    python predict_from_csv.py --input path/to/data.csv --chunk 200 --url http://localhost:8000
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests

from config import FEATURE_COLS, LEAKY_COLUMNS as LEAKY_COLS, PASSTHROUGH_COLS as KEEP_COLS


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_api(url: str) -> bool:
    try:
        r = requests.get(f"{url}/health", timeout=3)
        if r.status_code == 200:
            info = r.json()
            print(f"  API live — schema_loaded={info['schema_loaded']}, features={info['feature_count']}")
            return True
        print(f"  API returned {r.status_code}")
        return False
    except Exception as e:
        print(f"  Cannot reach API: {e}")
        return False


def row_to_payload(row: pd.Series) -> dict:
    """Convert one DataFrame row to a clean API payload dict."""
    payload = {}
    for col in FEATURE_COLS:
        if col in row.index:
            val = row[col]
            # pandas NA / NaN → omit (API treats missing as NaN natively)
            if pd.isna(val):
                continue
            # Convert numpy types to native Python so requests can serialise them
            if hasattr(val, "item"):
                val = val.item()
            payload[col] = val
    return payload


def predict_batch(rows: list[dict], url: str) -> list[dict]:
    """POST a chunk to /predict/batch. Returns list of result dicts."""
    try:
        r = requests.post(f"{url}/predict/batch", json=rows, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        print(f"  HTTP error {e.response.status_code}: {e.response.text[:200]}")
        return []
    except Exception as e:
        print(f"  Request failed: {e}")
        return []


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch-predict from a CSV file")
    parser.add_argument("--input",  required=True, help="Path to input CSV")
    parser.add_argument("--output", default="",    help="Path to output CSV (default: <input>_predictions.csv)")
    parser.add_argument("--chunk",  type=int, default=100, help="Rows per API call (default: 100)")
    parser.add_argument("--url",    default="http://localhost:8000", help="API base URL")
    parser.add_argument("--limit",  type=int, default=0, help="Only process first N rows (0 = all)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"ERROR: File not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + "_predictions.csv"
    )

    # ── 1. Health check ───────────────────────────────────────────────────────
    print(f"\nChecking API at {args.url} ...")
    if not check_api(args.url):
        sys.exit("ERROR: API is not reachable. Start it with:  uvicorn main:app --reload")

    # ── 2. Load CSV ───────────────────────────────────────────────────────────
    print(f"\nLoading {input_path} ...")
    df = pd.read_csv(input_path, parse_dates=["datetime"] if "datetime" in pd.read_csv(input_path, nrows=0).columns else [])
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    # Drop leaky columns if present
    leaked = [c for c in LEAKY_COLS if c in df.columns]
    if leaked:
        df = df.drop(columns=leaked)
        print(f"  Dropped {len(leaked)} leaky columns: {leaked}")

    # Limit rows if requested
    if args.limit > 0:
        df = df.head(args.limit)
        print(f"  Limited to first {args.limit} rows")

    # Identify which feature columns are actually present in this CSV
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing_from_csv = [c for c in FEATURE_COLS if c not in df.columns]
    print(f"  Feature columns found: {len(available)}/{len(FEATURE_COLS)}")
    if missing_from_csv:
        print(f"  Missing (will be NaN in model): {missing_from_csv}")

    # Context columns to carry through to output
    context_cols = [c for c in KEEP_COLS if c in df.columns]

    # ── 3. Batch predict ──────────────────────────────────────────────────────
    print(f"\nRunning predictions in chunks of {args.chunk} ...")
    all_results = []
    total = len(df)

    for start in range(0, total, args.chunk):
        chunk = df.iloc[start : start + args.chunk]
        payloads = [row_to_payload(row) for _, row in chunk.iterrows()]
        results  = predict_batch(payloads, args.url)

        if not results:
            # Fill with nulls on failure so row count stays aligned
            results = [{"should_move": None, "move_probability": None, "predicted_uplift": None}
                       for _ in range(len(chunk))]

        all_results.extend(results)
        done = min(start + args.chunk, total)
        pct  = done / total * 100
        moves = sum(1 for r in results if r.get("should_move") == 1)
        print(f"  [{done:>{len(str(total))}}/{total}] {pct:5.1f}%  — {moves}/{len(results)} flagged for move in this chunk")

    # ── 4. Build output DataFrame ─────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)[["should_move", "move_probability", "predicted_uplift"]]
    output_df  = pd.concat([df[context_cols + available].reset_index(drop=True), results_df], axis=1)

    output_df.to_csv(output_path, index=False)

    # ── 5. Summary ────────────────────────────────────────────────────────────
    total_flags = results_df["should_move"].sum()
    avg_conf    = results_df["move_probability"].mean() * 100
    avg_uplift  = results_df.loc[results_df["should_move"] == 1, "predicted_uplift"].mean()

    print(f"\n{'='*50}")
    print(f"  Total rows processed : {total:,}")
    print(f"  Flagged for move     : {int(total_flags):,}  ({total_flags/total*100:.1f}%)")
    print(f"  Avg move probability : {avg_conf:.1f}%")
    print(f"  Avg uplift (move=1)  : {avg_uplift:.4f}")
    print(f"  Output saved to      : {output_path}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()

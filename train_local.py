"""
train_local.py
==============
Train or incrementally update CTV Promo models from any local CSV.

The CSV must contain label columns:
  - should_move          : 0 (keep) or 1 (move)
  - uplift_if_optimised  : float — expected effectiveness uplift if moved

All other columns are treated as features. Leaky post-hoc columns
(actual_status, was_missed, etc.) are dropped automatically.

Usage
-----
  # Full retrain from scratch
  python train_local.py --data your_data.csv

  # Warm-start: add 100 new trees to existing model (fast update)
  python train_local.py --data new_data.csv --incremental

  # Custom model directory
  python train_local.py --data your_data.csv --model-dir /path/to/model
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from trainer import LEAKY_COLUMNS, TARGET_CLS, TARGET_REG, train_models


def main():
    p = argparse.ArgumentParser(description="Train CTV Promo models from a local CSV")
    p.add_argument("--data", required=True, help="Path to labeled training CSV")
    p.add_argument(
        "--incremental", action="store_true",
        help="Warm-start: add new trees on top of existing model (no full retrain)",
    )
    p.add_argument(
        "--new-trees", type=int, default=100,
        help="Trees to add in incremental mode (default: 100)",
    )
    p.add_argument(
        "--model-dir", default="model",
        help="Directory to save model artifacts (default: model/)",
    )
    args = p.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"ERROR: {data_path} not found")

    model_dir = Path(args.model_dir)

    # Load CSV — parse datetime column if present
    print(f"\nLoading {data_path} ...")
    peek = pd.read_csv(data_path, nrows=0)
    df = pd.read_csv(
        data_path,
        parse_dates=["datetime"] if "datetime" in peek.columns else [],
    )
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    # Validate required label columns
    missing = [c for c in [TARGET_CLS, TARGET_REG] if c not in df.columns]
    if missing:
        sys.exit(
            f"\nERROR: Training CSV is missing required label columns: {missing}\n"
            f"\n  '{TARGET_CLS}'         = 0 (keep) or 1 (move)\n"
            f"  '{TARGET_REG}' = float (expected uplift if moved)\n"
            f"\nIf you only have unlabelled schedule data, use predict_from_csv.py instead.\n"
        )

    # Report leaky columns found
    leaked = [c for c in LEAKY_COLUMNS if c in df.columns]
    if leaked:
        print(f"  Dropping {len(leaked)} leaky post-hoc columns: {leaked}")

    mode = "incremental (warm-start)" if args.incremental else "full retrain from scratch"
    print(f"\nTraining ({mode}) ...")

    metrics = train_models(
        df, model_dir,
        incremental=args.incremental,
        n_new_trees=args.new_trees,
    )

    print(f"\n{'='*54}")
    print(f"  Classifier PR-AUC  : {metrics['pr_auc']}")
    print(f"  Regressor R²       : {metrics['r2']}")
    print(f"  Rows trained       : {metrics['rows_trained']:,}")
    print(f"  Feature columns    : {metrics['features']}")
    print(f"  Mode               : {'incremental' if metrics['incremental'] else 'full retrain'}")
    print(f"  Artifacts saved to : {model_dir}/")
    print(f"{'='*54}")

    print("\nTo use the updated models:")
    print("  • If the API is running : POST http://localhost:8000/reload")
    print("  • Otherwise             : uvicorn main:app --reload\n")


if __name__ == "__main__":
    main()

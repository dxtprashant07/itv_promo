"""
generate_schema.py
==================
Recover inference_schema.json from saved XGBoost pkl files.

Run this whenever inference_schema.json is missing (e.g. after copying model
artifacts without the schema, or after Kaggle training where the schema was
written to /kaggle/working/ but the pkls were downloaded separately).

Limitations
-----------
- category_values cannot be recovered from pkl files alone.  The schema is
  written with an empty dict for category_values, which means unseen strings
  will pass through as NaN instead of being rejected.  Re-run the full
  ctv_promo_pipeline.py on the original CSV to get the complete vocabulary.
"""

import json
import sys
from pathlib import Path

import joblib

MODEL_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("model")


def main() -> None:
    cls_path = MODEL_DIR / "promo_move_classifier.pkl"
    reg_path = MODEL_DIR / "promo_uplift_regressor.pkl"

    if not cls_path.exists():
        sys.exit(f"ERROR: {cls_path} not found")
    if not reg_path.exists():
        sys.exit(f"ERROR: {reg_path} not found")

    print(f"Loading models from {MODEL_DIR} ...")
    cls_model = joblib.load(cls_path)

    # XGBoost stores feature names and types when trained with a DataFrame
    if not hasattr(cls_model, "feature_names_in_"):
        sys.exit("ERROR: model has no feature_names_in_ — was it trained with a DataFrame?")

    feature_cols = list(cls_model.feature_names_in_)
    print(f"  {len(feature_cols)} features found")

    # feature_types_ is a list of 'c' (categorical) or 'q' (quantitative)
    feature_types = getattr(cls_model, "feature_types_", None) or getattr(cls_model, "feature_types", None)
    if feature_types:
        categorical_cols = [feature_cols[i] for i, t in enumerate(feature_types) if t == "c"]
        print(f"  {len(categorical_cols)} categorical features: {categorical_cols}")
    else:
        categorical_cols = []
        print("  WARNING: feature_types not found — no categoricals identified. "
              "Add them manually to the schema if needed.")

    schema = {
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols,
        # Cannot recover vocabulary from pkl — leave empty.
        # Consequence: any string value passes through as NaN rather than
        # being validated against training categories.
        "category_values": {col: [] for col in categorical_cols},
        "_note": (
            "Partial schema extracted from pkl. Re-run ctv_promo_pipeline.py "
            "on the original CSV to populate category_values."
        ),
    }

    out_path = MODEL_DIR / "inference_schema.json"
    with open(out_path, "w") as f:
        json.dump(schema, f, indent=2)

    print(f"\nSchema written to {out_path}")
    print("NOTE: category_values is empty — categorical inputs won't be validated.")
    print("      Run ctv_promo_pipeline.py on the original data for the full schema.")


if __name__ == "__main__":
    main()

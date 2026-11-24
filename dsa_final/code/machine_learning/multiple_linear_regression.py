"""
multiple_linear_regression.py
=============================
Multiple Linear Regression predicting continuous Behavior_Index from Q2, Q3, and Q4.
"""

import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.linear_model import LinearRegression

_CODE_DIR = str(Path(__file__).resolve().parents[1])
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from machine_learning.data_loader import (
    REGRESSION_MLR_FEATURES,
    evaluate_regression_predictions,
    get_regression_split,
    load_cleaned_dataset,
)

MODEL_NAME = "Multiple Linear Regression"
FEATURE_LABELS = ["Q2 Usage", "Q3 Energy", "Q4 Carbon"]


def train_and_evaluate(df: pd.DataFrame) -> Dict[str, Any]:
    """Fit Multiple Linear Regression on the 70/30 split and return predictions and metrics."""
    y = df["Behavior_Index"].values.astype(float)
    X_mlr = df[REGRESSION_MLR_FEATURES].values.astype(float)
    idx_train, idx_test = get_regression_split(df)

    model = LinearRegression().fit(X_mlr[idx_train], y[idx_train])
    y_pred = model.predict(X_mlr[idx_test])
    eval_out = evaluate_regression_predictions(
        MODEL_NAME, "Q2, Q3, Q4", len(idx_train), y[idx_test], y_pred
    )

    coefs = dict(zip(FEATURE_LABELS, [round(float(c), 4) for c in model.coef_]))
    return {
        **eval_out,
        "y_test": y[idx_test],
        "y_pred": y_pred,
        "coefs": coefs,
        "intercept": round(float(model.intercept_), 4),
    }


def main() -> None:
    """Standalone runner for Multiple Linear Regression."""
    df = load_cleaned_dataset()
    res = train_and_evaluate(df)
    print(f"{MODEL_NAME} (N = {len(df)}, 70/30 Split):")
    print(f"  Coefs  : {res['coefs']} | Intercept = {res['intercept']}")
    print(f"  Metrics: {res['row']}")


if __name__ == "__main__":
    main()

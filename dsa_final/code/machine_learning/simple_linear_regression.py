"""
simple_linear_regression.py
===========================
Simple Linear Regression predicting continuous Behavior_Index from Awareness_Index.
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
    REGRESSION_SLR_FEATURES,
    evaluate_regression_predictions,
    get_regression_split,
    load_cleaned_dataset,
)

MODEL_NAME = "Simple Linear Regression"


def train_and_evaluate(df: pd.DataFrame) -> Dict[str, Any]:
    """Fit Simple Linear Regression on the 70/30 split and return predictions and metrics."""
    y = df["Behavior_Index"].values.astype(float)
    X_slr = df[REGRESSION_SLR_FEATURES].values.astype(float)
    idx_train, idx_test = get_regression_split(df)

    model = LinearRegression().fit(X_slr[idx_train], y[idx_train])
    y_pred = model.predict(X_slr[idx_test])
    eval_out = evaluate_regression_predictions(
        MODEL_NAME, "Awareness_Index", len(idx_train), y[idx_test], y_pred
    )

    return {
        **eval_out,
        "idx_train": idx_train,
        "idx_test": idx_test,
        "x_train": X_slr[idx_train].ravel(),
        "y_train": y[idx_train],
        "x_test": X_slr[idx_test].ravel(),
        "y_test": y[idx_test],
        "y_pred": y_pred,
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
    }


def main() -> None:
    """Standalone runner for Simple Linear Regression."""
    df = load_cleaned_dataset()
    res = train_and_evaluate(df)
    print(f"{MODEL_NAME} (N = {len(df)}, 70/30 Split):")
    print(f"  Equation: Behavior_Index = {res['slope']:.4f} * Awareness_Index + {res['intercept']:.4f}")
    print(f"  Metrics : {res['row']}")


if __name__ == "__main__":
    main()

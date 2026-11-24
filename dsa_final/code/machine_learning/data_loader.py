"""
data_loader.py
==============
Shared paths, train-test splitters, plotting configuration, and evaluation
helpers for the six supervised machine learning models (N = 72, 70/30 split).
"""

import os
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

ML_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(ML_DIR, ".."))
PROJECT_DIR = os.path.abspath(os.path.join(CODE_DIR, ".."))
INPUT_CSV = os.path.join(PROJECT_DIR, "output", "cleaned_main_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

TEST_SIZE = 0.30
RANDOM_STATE = 42

REGRESSION_SLR_FEATURES: List[str] = ["Awareness_Index"]
REGRESSION_MLR_FEATURES: List[str] = ["usage_freq_n", "aware_energy_n", "aware_carbon_n"]
CLASSIFICATION_FEATURES: List[str] = ["usage_freq_n", "aware_energy_n", "aware_carbon_n"]


def configure_plot_style() -> None:
    """Apply consistent publication-grade Matplotlib styling."""
    plt.rcParams.update({
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "font.family": "DejaVu Sans",
    })


def load_cleaned_dataset(csv_path: str = INPUT_CSV) -> pd.DataFrame:
    """Load `cleaned_main_dataset.csv`, raising a clear error if Stage 1 has not run."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Run `code/01_clean_and_analyze.py` (or `code/data_cleaning/main.py`) first.")
    return pd.read_csv(csv_path)


def get_regression_split(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return deterministic 70/30 row indices (50 train / 22 test) for regression."""
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return idx_train, idx_test


def get_classification_split(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return stratified 70/30 feature and target arrays (14 High / 8 Low-Mod test)."""
    X = df[CLASSIFICATION_FEATURES].values.astype(float)
    y = df["high_behavior"].values.astype(int)
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def evaluate_regression_predictions(
    model_name: str,
    predictors: str,
    train_n: int,
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute R², RMSE, and MAE summary row plus unrounded metrics."""
    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    return {
        "row": {
            "model": model_name,
            "predictors": predictors,
            "train_n": int(train_n),
            "test_n": int(len(y_test)),
            "test_r2": round(r2, 4),
            "test_rmse": round(rmse, 4),
            "test_mae": round(mae, 4),
        },
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
    }


def evaluate_classification_predictions(
    model_name: str,
    train_n: int,
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute classification metrics, confusion matrix, and per-class report."""
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    rep = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Low/Moderate (<4.0)", "High (>=4.0)"],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(rep).transpose().assign(model=model_name)

    return {
        "model": model_name,
        "row": {
            "model": model_name,
            "train_n": int(train_n),
            "test_n": int(len(y_test)),
            "accuracy": round(acc, 4),
            "precision_weighted": round(prec, 4),
            "recall_weighted": round(rec, 4),
            "f1_weighted": round(f1, 4),
        },
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "report_df": report_df,
    }

"""
main.py
=======
Coordinator for Stage 2: Runs all 6 supervised machine learning modules,
exports evaluation tables and `ml_summary.txt`, and plots Figures 5 and 6.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

_CODE_DIR = str(Path(__file__).resolve().parents[1])
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from machine_learning import (
    decision_tree,
    knn,
    multiple_linear_regression,
    naive_bayes,
    simple_linear_regression,
    svm,
)
from machine_learning.data_loader import (
    FIG_DIR,
    OUTPUT_DIR,
    get_classification_split,
    load_cleaned_dataset,
)
from machine_learning.visualizations import generate_all_ml_figures


def run_regression_suite(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run Simple and Multiple Linear Regression and export `ml_regression_results.csv`."""
    slr_res = simple_linear_regression.train_and_evaluate(df)
    mlr_res = multiple_linear_regression.train_and_evaluate(df)

    reg_df = pd.DataFrame([slr_res["row"], mlr_res["row"]])
    reg_df.to_csv(os.path.join(OUTPUT_DIR, "ml_regression_results.csv"), index=False, encoding="utf-8-sig")

    reg_info = {
        "idx_train": slr_res["idx_train"],
        "idx_test": slr_res["idx_test"],
        "x_train": slr_res["x_train"],
        "y_train": slr_res["y_train"],
        "x_test": slr_res["x_test"],
        "y_test": slr_res["y_test"],
        "slr_slope": slr_res["slope"],
        "slr_intercept": slr_res["intercept"],
        "slr_pred": slr_res["y_pred"],
        "mlr_pred": mlr_res["y_pred"],
        "mlr_coefs": mlr_res["coefs"],
        "mlr_intercept": mlr_res["intercept"],
        "slr_r2": slr_res["r2"],
        "slr_rmse": slr_res["rmse"],
        "slr_mae": slr_res["mae"],
        "mlr_r2": mlr_res["r2"],
        "mlr_rmse": mlr_res["rmse"],
        "mlr_mae": mlr_res["mae"],
    }
    return reg_df, reg_info


def run_classification_suite(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Run all 4 classifiers and export classification metrics and per-class reports."""
    _, _, _, y_test = get_classification_split(df)
    modules = [decision_tree, svm, naive_bayes, knn]

    rows = []
    preds = {}
    cms = {}
    report_frames = []

    for mod in modules:
        res = mod.train_and_evaluate(df)
        name = res["model"]
        rows.append(res["row"])
        preds[name] = res["y_pred"]
        cms[name] = res["confusion_matrix"]
        report_frames.append(res["report_df"])

    clf_df = pd.DataFrame(rows).sort_values(["accuracy", "f1_weighted"], ascending=False).reset_index(drop=True)
    clf_df.to_csv(os.path.join(OUTPUT_DIR, "ml_classification_results.csv"), index=False, encoding="utf-8-sig")

    rep_df = pd.concat(report_frames).round(4)
    rep_df.index.name = "class_or_avg"
    rep_df.to_csv(os.path.join(OUTPUT_DIR, "ml_classification_reports.csv"), encoding="utf-8-sig")

    return clf_df, y_test, preds, cms


def main() -> None:
    """Execute all 6 supervised machine learning models and export tables, summary, and plots."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 74)
    print("  STEP 2: SUPERVISED MACHINE LEARNING (N = 72, 70/30 TRAIN-TEST SPLIT)")
    print("=" * 74)

    df = load_cleaned_dataset()
    print(f"Loaded cleaned main dataset: N = {len(df)}")
    print("Target Counts (High >= 4.0 vs Low/Mod < 4.0):", df["high_behavior"].value_counts().to_dict())

    reg_df, reg_info = run_regression_suite(df)
    clf_df, y_test, preds, cms = run_classification_suite(df)

    print("\n--- REGRESSION RESULTS (Test Set n = 22) ---")
    print(reg_df.to_string(index=False))
    print(f"SLR Equation (Train n=50): Behavior_Index = {reg_info['slr_slope']:.4f} * Awareness_Index + {reg_info['slr_intercept']:.4f}")
    print(f"MLR Coefficients (Train n=50): {reg_info['mlr_coefs']} | Intercept = {reg_info['mlr_intercept']}")

    print("\n--- CLASSIFICATION RESULTS (Test Set n = 22; 14 High, 8 Low/Moderate) ---")
    print(clf_df.to_string(index=False))
    for name, cm in cms.items():
        print(f"  {name} Confusion Matrix [[TN, FP], [FN, TP]]: {cm.tolist()}")

    generate_all_ml_figures(reg_info, clf_df, y_test, preds)

    summary_lines = [
        "SUPERVISED MACHINE LEARNING SUMMARY (N = 72, 70/30 TRAIN-TEST SPLIT)",
        "=" * 70,
        "Split: 50 Training Records (70%) / 22 Testing Records (30%), random_state = 42",
        "",
        "REGRESSION (Continuous Behavior_Index):",
        reg_df.to_string(index=False),
        f"SLR Formula: Behavior_Index = {reg_info['slr_slope']:.4f} * Awareness_Index + {reg_info['slr_intercept']:.4f}",
        f"MLR Coefs  : {reg_info['mlr_coefs']} | Intercept = {reg_info['mlr_intercept']}",
        "",
        "CLASSIFICATION (High >= 4.0 [14 test] vs. Low/Moderate < 4.0 [8 test]):",
        clf_df.to_string(index=False),
    ]
    with open(os.path.join(OUTPUT_DIR, "ml_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"\nSaved supervised ML tables, summary, and 2 figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

"""
visualizations.py
=================
Publication-grade supervised machine learning plots (Figures 5 and 6).
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

_CODE_DIR = str(Path(__file__).resolve().parents[1])
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from machine_learning.data_loader import FIG_DIR, configure_plot_style


def plot_fig5_regression(reg_info: Dict[str, Any]) -> None:
    """Figure 5: Linear Regression Fit and Test Predictions (70/30 Split)."""
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), gridspec_kw={"wspace": 0.28})
    rng = np.random.default_rng(42)

    xtr = reg_info["x_train"] + rng.uniform(-0.07, 0.07, size=len(reg_info["x_train"]))
    ytr = reg_info["y_train"] + rng.uniform(-0.07, 0.07, size=len(reg_info["y_train"]))
    xte = reg_info["x_test"] + rng.uniform(-0.07, 0.07, size=len(reg_info["x_test"]))
    yte = reg_info["y_test"] + rng.uniform(-0.07, 0.07, size=len(reg_info["y_test"]))

    axes[0].scatter(xtr, ytr, color="#2563eb", alpha=0.58, s=36, edgecolor="white", linewidth=0.6, label="Training Set (n = 50)")
    axes[0].scatter(xte, yte, color="#ea580c", marker="D", alpha=0.88, s=40, edgecolor="white", linewidth=0.6, label="Test Set (n = 22)")

    x_grid = np.linspace(1.0, 5.0, 100)
    y_line = reg_info["slr_slope"] * x_grid + reg_info["slr_intercept"]
    axes[0].plot(
        x_grid, y_line, color="#b91c1c", linewidth=2.3,
        label=f"SLR: y = {reg_info['slr_slope']:.2f}x + {reg_info['slr_intercept']:.2f}",
    )
    axes[0].text(
        0.04, 0.94,
        f"Test R² = {reg_info['slr_r2']:.3f}\nTest RMSE = {reg_info['slr_rmse']:.3f}\nTest MAE = {reg_info['slr_mae']:.3f}",
        transform=axes[0].transAxes, va="top", fontsize=8.3, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )
    axes[0].set_xlim(0.75, 5.25)
    axes[0].set_ylim(0.75, 5.25)
    axes[0].set_xlabel("Awareness Index (Q3 & Q4 Mean)")
    axes[0].set_ylabel("Behavior Index (Q5 & Q6 Mean)")
    axes[0].set_title("A. Simple Linear Regression (70/30 Split)", fontweight="bold", pad=10)
    axes[0].legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=7.8)

    y_actual = reg_info["y_test"]
    jitter = rng.uniform(-0.05, 0.05, size=len(y_actual))
    axes[1].scatter(y_actual + jitter, reg_info["slr_pred"], color="#0f766e", alpha=0.78, s=42, edgecolor="white", linewidth=0.6, label=f"SLR (Test R² = {reg_info['slr_r2']:.3f})")
    axes[1].scatter(y_actual - jitter, reg_info["mlr_pred"], color="#7c3aed", marker="^", alpha=0.78, s=42, edgecolor="white", linewidth=0.6, label=f"MLR (Test R² = {reg_info['mlr_r2']:.3f})")
    axes[1].plot([1.0, 5.0], [1.0, 5.0], color="#b91c1c", linestyle="--", linewidth=1.8, label="Ideal Prediction (y = x)")
    axes[1].set_xlim(0.75, 5.25)
    axes[1].set_ylim(0.75, 5.25)
    axes[1].set_xlabel("Actual Test Behavior Index")
    axes[1].set_ylabel("Predicted Behavior Index")
    axes[1].set_title("B. Actual vs. Predicted Behavior (Test n = 22)", fontweight="bold", pad=10)
    axes[1].legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=7.8)

    fig.savefig(os.path.join(FIG_DIR, "fig5_ml_regression.png"), bbox_inches="tight")
    plt.close(fig)


def plot_fig6_classification(
    clf_df: pd.DataFrame,
    y_test: np.ndarray,
    preds: Dict[str, np.ndarray],
) -> None:
    """Figure 6: Supervised Classifier Test Accuracy and Top Confusion Matrix (70/30 Split)."""
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), gridspec_kw={"width_ratios": [1.3, 1.0], "wspace": 0.46})

    plot_df = clf_df.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(plot_df))
    h = 0.36
    b1 = axes[0].barh(y_pos + h / 2, plot_df["accuracy"] * 100.0, height=h, color="#1d4ed8", alpha=0.92, label="Test Accuracy (%)")
    b2 = axes[0].barh(y_pos - h / 2, plot_df["f1_weighted"] * 100.0, height=h, color="#60a5fa", alpha=0.92, label="Weighted F1 (%)")

    for bar in b1:
        v = bar.get_width()
        axes[0].text(v + 0.8, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8.0, fontweight="bold", color="#1e3a8a")
    for bar in b2:
        v = bar.get_width()
        axes[0].text(v + 0.8, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=7.8, color="#334155")

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(plot_df["model"], fontsize=8.6, fontweight="bold")
    axes[0].set_xlim(45, 92)
    axes[0].set_xlabel("Test Score (%)")
    axes[0].set_title("A. Supervised Classifier Comparison (Test n = 22)", fontweight="bold", pad=10)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.48, -0.17), ncol=2, frameon=False, fontsize=8.2)

    top_name = clf_df.iloc[0]["model"]
    y_pred = preds[top_name]
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    acc = accuracy_score(y_test, y_pred) * 100.0
    f1w = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    im = axes[1].imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    lbls = ["Low / Moderate\n(< 4.00)", "High Responsible\n(>= 4.00)"]
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(lbls, fontsize=8.4, fontweight="bold")
    axes[1].set_yticklabels(lbls, fontsize=8.4, fontweight="bold")
    axes[1].set_xlabel("Predicted Behavior Tier")
    axes[1].set_ylabel("Actual Behavior Tier")
    axes[1].set_title(f"B. {top_name} Confusion Matrix\n(Test Acc = {acc:.1f}%, Weighted F1 = {f1w:.3f})", fontweight="bold", pad=10)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            pct = v * 100.0 / cm.sum()
            axes[1].text(
                j, i, f"{v}\n({pct:.1f}%)", ha="center", va="center",
                color="white" if v > thresh else "#0f172a", fontweight="bold", fontsize=10,
            )

    fig.savefig(os.path.join(FIG_DIR, "fig6_ml_classification.png"), bbox_inches="tight")
    plt.close(fig)


def generate_all_ml_figures(
    reg_info: Dict[str, Any],
    clf_df: pd.DataFrame,
    y_test: np.ndarray,
    preds: Dict[str, np.ndarray],
) -> None:
    """Create both supervised machine learning figures (Figures 5 and 6)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    plot_fig5_regression(reg_info)
    plot_fig6_classification(clf_df, y_test, preds)

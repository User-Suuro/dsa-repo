"""
02_machine_learning.py
======================
Supervised Machine Learning for:
  "AI and the Environment: An Analysis of Public Awareness and Responsible AI Usage"

Dataset:
  - `output/cleaned_main_dataset.csv` (N = 72 cleaned primary survey records).

Models (Standard 70/30 Train-Test Split: 50 Train / 22 Test):
  Regression (predicting continuous Behavior_Index):
    1. Simple Linear Regression (Awareness_Index)
    2. Multiple Linear Regression (Q2, Q3, Q4)

  Classification (predicting High Responsible Behavior >= 4.0 vs. Low/Moderate < 4.0
  using Q2, Q3, and Q4):
    3. Decision Tree Classifier (max_depth=3)
    4. Support Vector Machine (SVM, RBF kernel)
    5. Gaussian Naive Bayes
    6. K-Nearest Neighbors (KNN, k=5)

Outputs -> `output/`:
  - `ml_regression_results.csv`
  - `ml_classification_results.csv`
  - `ml_classification_reports.csv`
  - `ml_summary.txt`
  - `figures/fig5_ml_regression.png`
  - `figures/fig6_ml_classification.png`
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CODE_DIR, ".."))
INPUT_CSV = os.path.join(PROJECT_DIR, "output", "cleaned_main_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

TEST_SIZE = 0.30
RANDOM_STATE = 42

plt.rcParams.update({
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "font.family": "DejaVu Sans",
})


def run_regression_models(df: pd.DataFrame) -> tuple:
    """Train and evaluate Simple and Multiple Linear Regression on a 70/30 split."""
    y = df["Behavior_Index"].values.astype(float)
    X_slr = df[["Awareness_Index"]].values.astype(float)
    X_mlr = df[["usage_freq_n", "aware_energy_n", "aware_carbon_n"]].values.astype(float)

    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 1. Simple Linear Regression
    slr = LinearRegression().fit(X_slr[idx_train], y[idx_train])
    slr_pred = slr.predict(X_slr[idx_test])

    # 2. Multiple Linear Regression
    mlr = LinearRegression().fit(X_mlr[idx_train], y[idx_train])
    mlr_pred = mlr.predict(X_mlr[idx_test])

    y_test = y[idx_test]
    rows = [
        {
            "model": "Simple Linear Regression",
            "predictors": "Awareness_Index",
            "train_n": len(idx_train),
            "test_n": len(idx_test),
            "test_r2": round(float(r2_score(y_test, slr_pred)), 4),
            "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, slr_pred))), 4),
            "test_mae": round(float(mean_absolute_error(y_test, slr_pred)), 4),
        },
        {
            "model": "Multiple Linear Regression",
            "predictors": "Q2, Q3, Q4",
            "train_n": len(idx_train),
            "test_n": len(idx_test),
            "test_r2": round(float(r2_score(y_test, mlr_pred)), 4),
            "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, mlr_pred))), 4),
            "test_mae": round(float(mean_absolute_error(y_test, mlr_pred)), 4),
        },
    ]
    reg_df = pd.DataFrame(rows)
    reg_df.to_csv(os.path.join(OUTPUT_DIR, "ml_regression_results.csv"), index=False, encoding="utf-8-sig")

    reg_info = {
        "idx_train": idx_train,
        "idx_test": idx_test,
        "x_train": X_slr[idx_train].ravel(),
        "y_train": y[idx_train],
        "x_test": X_slr[idx_test].ravel(),
        "y_test": y_test,
        "slr_slope": float(slr.coef_[0]),
        "slr_intercept": float(slr.intercept_),
        "slr_pred": slr_pred,
        "mlr_pred": mlr_pred,
        "mlr_coefs": dict(zip(["Q2 Usage", "Q3 Energy", "Q4 Carbon"], [round(float(c), 4) for c in mlr.coef_])),
        "mlr_intercept": round(float(mlr.intercept_), 4),
        "slr_r2": float(r2_score(y_test, slr_pred)),
        "slr_rmse": float(np.sqrt(mean_squared_error(y_test, slr_pred))),
        "slr_mae": float(mean_absolute_error(y_test, slr_pred)),
        "mlr_r2": float(r2_score(y_test, mlr_pred)),
        "mlr_rmse": float(np.sqrt(mean_squared_error(y_test, mlr_pred))),
        "mlr_mae": float(mean_absolute_error(y_test, mlr_pred)),
    }
    return reg_df, reg_info


def run_classification_models(df: pd.DataFrame) -> tuple:
    """Train and evaluate 4 supervised classifiers on a stratified 70/30 split."""
    X = df[["usage_freq_n", "aware_energy_n", "aware_carbon_n"]].values.astype(float)
    y = df["high_behavior"].values.astype(int)  # 1 = High (>=4.0), 0 = Low/Moderate (<4.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    classifiers = [
        ("Decision Tree (max_depth=3)", DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)),
        ("Support Vector Machine (SVM)", make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, random_state=RANDOM_STATE))),
        ("Gaussian Naive Bayes", GaussianNB()),
        ("K-Nearest Neighbors (k=5)", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))),
    ]

    rows = []
    preds = {}
    cms = {}
    report_frames = []

    for name, clf in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        preds[name] = y_pred
        cms[name] = confusion_matrix(y_test, y_pred, labels=[0, 1])

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        rows.append({
            "model": name,
            "train_n": len(X_train),
            "test_n": len(X_test),
            "accuracy": round(float(acc), 4),
            "precision_weighted": round(float(prec), 4),
            "recall_weighted": round(float(rec), 4),
            "f1_weighted": round(float(f1), 4),
        })

        rep = classification_report(
            y_test, y_pred,
            labels=[0, 1],
            target_names=["Low/Moderate (<4.0)", "High (>=4.0)"],
            output_dict=True,
            zero_division=0,
        )
        report_frames.append(pd.DataFrame(rep).transpose().assign(model=name))

    clf_df = pd.DataFrame(rows).sort_values(["accuracy", "f1_weighted"], ascending=False).reset_index(drop=True)
    clf_df.to_csv(os.path.join(OUTPUT_DIR, "ml_classification_results.csv"), index=False, encoding="utf-8-sig")

    rep_df = pd.concat(report_frames).round(4)
    rep_df.index.name = "class_or_avg"
    rep_df.to_csv(os.path.join(OUTPUT_DIR, "ml_classification_reports.csv"), encoding="utf-8-sig")

    return clf_df, y_test, preds, cms


def plot_fig5_regression(reg_info: dict) -> None:
    """Figure 5: Linear Regression Fit and Test Predictions (70/30 Split)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), gridspec_kw={"wspace": 0.28})
    rng = np.random.default_rng(42)

    # Panel A: Training & Test points + SLR regression line
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

    # Panel B: Actual vs. Predicted on Test Set (n = 22) for SLR & MLR
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


def plot_fig6_classification(clf_df: pd.DataFrame, y_test: np.ndarray, preds: dict) -> None:
    """Figure 6: Supervised Classifier Test Accuracy and Top Confusion Matrix (70/30 Split)."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), gridspec_kw={"width_ratios": [1.3, 1.0], "wspace": 0.46})

    # Panel A: Test Accuracy & Weighted F1 across the 4 classifiers
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

    # Panel B: Confusion Matrix for Top Classifier (Decision Tree)
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


def main() -> None:
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError("Run `code/01_clean_and_analyze.py` first.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 74)
    print("  STEP 2: SUPERVISED MACHINE LEARNING (N = 72, 70/30 TRAIN-TEST SPLIT)")
    print("=" * 74)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded cleaned main dataset: N = {len(df)}")
    print("Target Counts (High >= 4.0 vs Low/Mod < 4.0):", df["high_behavior"].value_counts().to_dict())

    reg_df, reg_info = run_regression_models(df)
    clf_df, y_test, preds, cms = run_classification_models(df)

    print("\n--- REGRESSION RESULTS (Test Set n = 22) ---")
    print(reg_df.to_string(index=False))
    print(f"SLR Equation (Train n=50): Behavior_Index = {reg_info['slr_slope']:.4f} * Awareness_Index + {reg_info['slr_intercept']:.4f}")
    print(f"MLR Coefficients (Train n=50): {reg_info['mlr_coefs']} | Intercept = {reg_info['mlr_intercept']}")

    print("\n--- CLASSIFICATION RESULTS (Test Set n = 22; 14 High, 8 Low/Moderate) ---")
    print(clf_df.to_string(index=False))
    for name, cm in cms.items():
        print(f"  {name} Confusion Matrix [[TN, FP], [FN, TP]]: {cm.tolist()}")

    plot_fig5_regression(reg_info)
    plot_fig6_classification(clf_df, y_test, preds)

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

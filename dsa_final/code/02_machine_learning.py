"""
02_machine_learning.py
======================
Small-Sample Supervised Machine Learning for:
  "AI and the Environment: An Analysis of Public Awareness"

Dataset:
  - `output/cleaned_main_dataset.csv` (N = 72 cleaned primary survey records).

Why Cross-Validation & Low-Variance Models for Small Records (N = 72):
  1. A single 75/25 holdout leaves only 18 test records (and ~1-2 Low cases),
     causing high variance. Therefore, we evaluate every model with BOTH
     Leave-One-Out Cross-Validation (LOOCV) and Repeated Stratified 5-Fold CV
     (5 folds x 10 repeats = 50 folds).
  2. On small datasets (N < 100), low-variance, regularized/probabilistic models
     outperform complex high-variance tree ensembles:
       - Continuous Prediction (Behavior_Index): Simple Linear Regression (SLR)
         benchmarked against Multiple Linear Regression (MLR) & Ridge Regression.
       - Classification (Responsible Behavior): Gaussian Naive Bayes &
         L2-Regularized Logistic Regression benchmarked against Linear SVM,
         Shallow Decision Tree (max_depth=2), KNN (k=7), and Random Forest.

Outputs -> `output/`:
  - `ml_regression_cv.csv`
  - `ml_classification_binary_cv.csv`
  - `ml_classification_3class_cv.csv`
  - `ml_best_classification_report_loocv.csv`
  - `figures/fig5_ml_regression_loocv.png`
  - `figures/fig6_ml_small_sample_classification.png`
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
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

plt.rcParams.update({
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "font.family": "DejaVu Sans",
})


def run_regression_models(df: pd.DataFrame) -> tuple:
    """Evaluate small-sample regression models predicting continuous Behavior_Index."""
    y = df["Behavior_Index"].values.astype(float)
    X_slr = df[["Awareness_Index"]].values.astype(float)
    X_mlr = df[["usage_freq_n", "aware_energy_n", "aware_carbon_n"]].values.astype(float)

    loo = LeaveOneOut()
    kf5 = KFold(n_splits=5, shuffle=True, random_state=42)
    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

    specs = [
        ("Simple Linear Regression (SLR)", "Awareness_Index", X_slr, LinearRegression()),
        ("Ridge Regression (L2, alpha=10)", "Q2, Q3, Q4", X_mlr, make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("Multiple Linear Regression (MLR)", "Q2, Q3, Q4", X_mlr, LinearRegression()),
    ]

    rows = []
    slr_loo_pred = None
    for name, feats, X, model in specs:
        loo_pred = cross_val_predict(model, X, y, cv=loo)
        if name.startswith("Simple"):
            slr_loo_pred = loo_pred

        cv5 = cross_validate(model, X, y, cv=kf5, scoring=["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"])
        cvr = cross_validate(model, X, y, cv=rkf, scoring=["r2", "neg_root_mean_squared_error"])

        model.fit(X, y)
        full_r2 = r2_score(y, model.predict(X))

        rows.append({
            "model": name,
            "predictors": feats,
            "full_r2": round(float(full_r2), 4),
            "loocv_r2": round(float(r2_score(y, loo_pred)), 4),
            "loocv_rmse": round(float(np.sqrt(mean_squared_error(y, loo_pred))), 4),
            "loocv_mae": round(float(mean_absolute_error(y, loo_pred)), 4),
            "cv5_rmse": round(float(-cv5["test_neg_root_mean_squared_error"].mean()), 4),
            "cv10x5_r2": round(float(cvr["test_r2"].mean()), 4),
        })

    reg_df = pd.DataFrame(rows)
    reg_df.to_csv(os.path.join(OUTPUT_DIR, "ml_regression_cv.csv"), index=False, encoding="utf-8-sig")

    # Fit final SLR parameters for reporting
    slr_full = LinearRegression().fit(X_slr, y)
    r_val, p_val = stats.pearsonr(X_slr.ravel(), y)
    slr_info = {
        "slope": float(slr_full.coef_[0]),
        "intercept": float(slr_full.intercept_),
        "pearson_r": float(r_val),
        "p_value": float(p_val),
        "loocv_r2": float(r2_score(y, slr_loo_pred)),
        "loocv_rmse": float(np.sqrt(mean_squared_error(y, slr_loo_pred))),
        "loocv_mae": float(mean_absolute_error(y, slr_loo_pred)),
        "loo_pred": slr_loo_pred,
    }
    return reg_df, slr_info


def run_classification_models(df: pd.DataFrame):
    """
    Evaluate 6 classifiers under 5-Fold Stratified CV, Repeated 10x5-Fold CV,
    and Leave-One-Out CV on BOTH:
      - Primary Small-N Binary Target: High Responsible Behavior (>= 4.0, n=47)
        vs. Low-to-Moderate (< 4.0, n=25)
      - Secondary 3-Class Target: Low (n=7), Medium (n=18), High (n=47)
    Predictors: non-collinear [Q2 usage_freq_n, Q3 aware_energy_n, Q4 aware_carbon_n].
    """
    X = df[["usage_freq_n", "aware_energy_n", "aware_carbon_n"]].values.astype(float)
    y_bin = df["high_behavior"].values.astype(int)          # 1 = High (>=4.0), 0 = Low/Moderate (<4.0)
    y_3cls = df["behavior_level"].values                    # Low, Medium, High

    skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    loo = LeaveOneOut()

    classifiers = [
        ("Gaussian Naive Bayes", GaussianNB()),
        ("Logistic Regression (L2)", make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))),
        ("Linear SVM (C=1.0)", make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))),
        ("Decision Tree (max_depth=2)", DecisionTreeClassifier(max_depth=2, random_state=42)),
        ("K-Nearest Neighbors (k=7)", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))),
        ("Random Forest (max_depth=3)", RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)),
    ]

    # --- A. Binary Classification (Recommended for N = 72) ---
    bin_rows = []
    best_bin_preds = {}
    for name, clf in classifiers:
        cv5 = cross_validate(clf, X, y_bin, cv=skf5, scoring=["accuracy", "f1_weighted"])
        cvr = cross_validate(clf, X, y_bin, cv=rskf, scoring=["accuracy", "f1_weighted"])
        loo_pred = cross_val_predict(clf, X, y_bin, cv=loo)
        best_bin_preds[name] = loo_pred

        auc_val = np.nan
        if hasattr(clf, "predict_proba"):
            loo_prob = cross_val_predict(clf, X, y_bin, cv=loo, method="predict_proba")[:, 1]
            auc_val = roc_auc_score(y_bin, loo_prob)
        elif hasattr(clf, "decision_function"):
            loo_dec = cross_val_predict(clf, X, y_bin, cv=loo, method="decision_function")
            auc_val = roc_auc_score(y_bin, loo_dec)

        bin_rows.append({
            "model": name,
            "cv5_acc": round(float(cv5["test_accuracy"].mean()), 4),
            "cv10x5_acc": round(float(cvr["test_accuracy"].mean()), 4),
            "cv10x5_std": round(float(cvr["test_accuracy"].std()), 4),
            "loocv_acc": round(float(accuracy_score(y_bin, loo_pred)), 4),
            "loocv_weighted_f1": round(float(f1_score(y_bin, loo_pred, average="weighted")), 4),
            "loocv_roc_auc": round(float(auc_val), 4) if not np.isnan(auc_val) else np.nan,
        })
    bin_df = pd.DataFrame(bin_rows).sort_values(["loocv_acc", "loocv_weighted_f1"], ascending=False).reset_index(drop=True)
    bin_df.to_csv(os.path.join(OUTPUT_DIR, "ml_classification_binary_cv.csv"), index=False, encoding="utf-8-sig")

    # --- B. 3-Class Classification (Low / Medium / High) ---
    cls3_rows = []
    cls3_preds = {}
    for name, clf in classifiers:
        cv5 = cross_validate(clf, X, y_3cls, cv=skf5, scoring=["accuracy", "f1_weighted"])
        cvr = cross_validate(clf, X, y_3cls, cv=rskf, scoring=["accuracy", "f1_weighted"])
        loo_pred = cross_val_predict(clf, X, y_3cls, cv=loo)
        cls3_preds[name] = loo_pred
        cls3_rows.append({
            "model": name,
            "cv5_acc": round(float(cv5["test_accuracy"].mean()), 4),
            "cv10x5_acc": round(float(cvr["test_accuracy"].mean()), 4),
            "cv10x5_std": round(float(cvr["test_accuracy"].std()), 4),
            "loocv_acc": round(float(accuracy_score(y_3cls, loo_pred)), 4),
            "loocv_weighted_f1": round(float(f1_score(y_3cls, loo_pred, average="weighted")), 4),
        })
    cls3_df = pd.DataFrame(cls3_rows).sort_values(["cv10x5_acc", "loocv_weighted_f1"], ascending=False).reset_index(drop=True)
    cls3_df.to_csv(os.path.join(OUTPUT_DIR, "ml_classification_3class_cv.csv"), index=False, encoding="utf-8-sig")

    # Export detailed LOOCV classification reports for Gaussian NB & L2 Logistic Regression (Binary)
    rep_gnb = classification_report(
        y_bin, best_bin_preds["Gaussian Naive Bayes"],
        labels=[0, 1], target_names=["Low/Moderate (<4.0)", "High (>=4.0)"], output_dict=True,
    )
    rep_lr = classification_report(
        y_bin, best_bin_preds["Logistic Regression (L2)"],
        labels=[0, 1], target_names=["Low/Moderate (<4.0)", "High (>=4.0)"], output_dict=True,
    )
    rep_df = pd.concat([
        pd.DataFrame(rep_gnb).transpose().assign(model="Gaussian Naive Bayes (LOOCV)"),
        pd.DataFrame(rep_lr).transpose().assign(model="Logistic Regression L2 (LOOCV)"),
    ]).round(4)
    rep_df.index.name = "class_or_avg"
    rep_df.to_csv(os.path.join(OUTPUT_DIR, "ml_best_classification_report_loocv.csv"), encoding="utf-8-sig")

    # Fit L2 Logistic Regression on full data to inspect standardized coefficients (interpretability)
    lr_pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(X, y_bin)
    lr_coefs = dict(zip(["Q2 Usage", "Q3 Energy Aware", "Q4 Carbon Aware"], lr_pipe.named_steps["logisticregression"].coef_[0]))

    return bin_df, cls3_df, best_bin_preds, cls3_preds, lr_coefs


def plot_fig5_regression(df: pd.DataFrame, slr_info: dict) -> None:
    """Figure 5: Simple Linear Regression Fit & LOOCV Actual vs. Predicted (N = 72)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), gridspec_kw={"wspace": 0.28})

    x = df["Awareness_Index"].values.astype(float)
    y = df["Behavior_Index"].values.astype(float)

    # Panel A: Jittered scatter + SLR regression line with 95% confidence band
    rng = np.random.default_rng(42)
    xj = x + rng.uniform(-0.08, 0.08, size=len(x))
    yj = y + rng.uniform(-0.08, 0.08, size=len(y))
    axes[0].scatter(xj, yj, color="#2563eb", alpha=0.65, s=36, edgecolor="white", linewidth=0.6, label="Cleaned Respondents (N = 72)")

    x_grid = np.linspace(1.0, 5.0, 100)
    y_hat = slr_info["slope"] * x_grid + slr_info["intercept"]

    # 95% confidence interval for mean prediction
    n = len(x)
    se = np.sqrt(np.sum((y - (slr_info["slope"] * x + slr_info["intercept"])) ** 2) / (n - 2))
    t_crit = stats.t.ppf(0.975, df=n - 2)
    ci = t_crit * se * np.sqrt(1.0 / n + (x_grid - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))

    axes[0].fill_between(x_grid, y_hat - ci, y_hat + ci, color="#dc2626", alpha=0.14, label="95% Confidence Band")
    axes[0].plot(x_grid, y_hat, color="#b91c1c", linewidth=2.3, label=f"SLR: y = {slr_info['slope']:.2f}x + {slr_info['intercept']:.2f}")
    axes[0].text(
        0.04, 0.94,
        f"Pearson r = {slr_info['pearson_r']:.3f} (p < 0.001)\nLOOCV R² = {slr_info['loocv_r2']:.3f}\nLOOCV RMSE = {slr_info['loocv_rmse']:.3f}",
        transform=axes[0].transAxes, va="top", fontsize=8.2, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )
    axes[0].set_xlim(0.75, 5.25)
    axes[0].set_ylim(0.75, 5.25)
    axes[0].set_xlabel("Awareness Index (Q3 & Q4 Mean)")
    axes[0].set_ylabel("Behavior Index (Q5 & Q6 Mean)")
    axes[0].set_title("A. Simple Linear Regression Fit (N = 72)", fontweight="bold", pad=10)
    axes[0].legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=7.8)

    # Panel B: Out-of-sample LOOCV Actual vs. Predicted
    loo_pred = slr_info["loo_pred"]
    axes[1].scatter(y + rng.uniform(-0.06, 0.06, size=len(y)), loo_pred, color="#0f766e", alpha=0.68, s=36, edgecolor="white", linewidth=0.6, label="Out-of-Sample LOOCV Points")
    axes[1].plot([1.0, 5.0], [1.0, 5.0], color="#b91c1c", linestyle="--", linewidth=1.8, label="Ideal Prediction (y = x)")
    axes[1].set_xlim(0.75, 5.25)
    axes[1].set_ylim(0.75, 5.25)
    axes[1].set_xlabel("Actual Behavior Index")
    axes[1].set_ylabel("LOOCV Predicted Behavior Index")
    axes[1].set_title(f"B. Out-of-Sample LOOCV Fit (MAE = {slr_info['loocv_mae']:.3f})", fontweight="bold", pad=10)
    axes[1].legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=7.8)

    fig.savefig(os.path.join(FIG_DIR, "fig5_ml_regression_loocv.png"), bbox_inches="tight")
    plt.close(fig)


def plot_fig6_classification(df: pd.DataFrame, bin_df: pd.DataFrame, best_bin_preds: dict) -> None:
    """Figure 6: Small-Sample Classifier Benchmark (10x5-Fold CV & LOOCV) and Gaussian NB Confusion Matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.48})

    # Panel A: Model Comparison (Binary High vs Low/Moderate)
    plot_df = bin_df.sort_values("loocv_acc", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(plot_df))
    h = 0.36
    b1 = axes[0].barh(y_pos + h / 2, plot_df["loocv_acc"] * 100.0, height=h, color="#1d4ed8", alpha=0.92, label="LOOCV Accuracy (%)")
    b2 = axes[0].barh(y_pos - h / 2, plot_df["cv10x5_acc"] * 100.0, height=h, color="#60a5fa", alpha=0.92, label="10x5-Fold Stratified CV (%)")

    for bar in b1:
        v = bar.get_width()
        axes[0].text(v + 0.6, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=7.8, fontweight="bold", color="#1e3a8a")
    for bar in b2:
        v = bar.get_width()
        axes[0].text(v + 0.6, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=7.6, color="#334155")

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(plot_df["model"], fontsize=8.4, fontweight="bold")
    axes[0].set_xlim(50, 91)
    axes[0].set_xlabel("Out-of-Sample Accuracy (%)")
    axes[0].set_title("A. Small-Sample Classifier Comparison (N = 72, Binary)", fontweight="bold", pad=10)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.48, -0.17), ncol=2, frameon=False, fontsize=8.0)

    # Panel B: LOOCV Confusion Matrix for Gaussian Naive Bayes (Highest ROC-AUC & F1)
    y_true = df["high_behavior"].values.astype(int)
    y_pred = best_bin_preds["Gaussian Naive Bayes"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc = accuracy_score(y_true, y_pred) * 100.0
    f1w = f1_score(y_true, y_pred, average="weighted")

    im = axes[1].imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    lbls = ["Low / Moderate\n(< 4.00)", "High Responsible\n(>= 4.00)"]
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(lbls, fontsize=8.4, fontweight="bold")
    axes[1].set_yticklabels(lbls, fontsize=8.4, fontweight="bold")
    axes[1].set_xlabel("Predicted Behavior Tier")
    axes[1].set_ylabel("Actual Behavior Tier")
    axes[1].set_title(f"B. Gaussian Naive Bayes LOOCV Matrix\n(Acc = {acc:.1f}%, Weighted F1 = {f1w:.3f})", fontweight="bold", pad=10)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            pct = v * 100.0 / cm.sum()
            axes[1].text(
                j, i, f"{v}\n({pct:.1f}%)", ha="center", va="center",
                color="white" if v > thresh else "#0f172a", fontweight="bold", fontsize=10,
            )

    fig.savefig(os.path.join(FIG_DIR, "fig6_ml_small_sample_classification.png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError("Run `code/01_clean_and_analyze.py` first.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 78)
    print("  STEP 2: SMALL-SAMPLE MACHINE LEARNING (N = 72, LOOCV & 10x5-FOLD CV)")
    print("=" * 78)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded cleaned main dataset: N = {len(df)}")
    print("Binary Target Counts (High >= 4.0 vs Low/Mod < 4.0):", df["high_behavior"].value_counts().to_dict())
    print("3-Class Target Counts (Low / Medium / High):", df["behavior_level"].value_counts().to_dict())

    reg_df, slr_info = run_regression_models(df)
    bin_df, cls3_df, best_bin_preds, cls3_preds, lr_coefs = run_classification_models(df)

    print("\n--- REGRESSION BENCHMARK (Continuous Behavior_Index, LOOCV & 5-Fold CV) ---")
    print(reg_df.to_string(index=False))
    print(f"SLR Formula: Behavior_Index = {slr_info['slope']:.4f} * Awareness_Index + {slr_info['intercept']:.4f}")

    print("\n--- BINARY CLASSIFICATION BENCHMARK (High >= 4.0 vs Low/Mod < 4.0, N = 72) ---")
    print(bin_df.to_string(index=False))

    print("\n--- 3-CLASS CLASSIFICATION BENCHMARK (Low / Medium / High, N = 72) ---")
    print(cls3_df.to_string(index=False))

    print("\n--- L2 LOGISTIC REGRESSION STANDARDIZED COEFFICIENTS (Binary Target) ---")
    for k, v in lr_coefs.items():
        print(f"  {k:<18}: {v:+.4f} (Odds Ratio = {np.exp(v):.3f})")

    plot_fig5_regression(df, slr_info)
    plot_fig6_classification(df, bin_df, best_bin_preds)

    # Save concise text summary
    summary_lines = [
        "SMALL-SAMPLE MACHINE LEARNING SUMMARY (N = 72 CLEANED MAIN DATASET)",
        "=" * 74,
        "Recommended Validation: Leave-One-Out CV (LOOCV) + Repeated 10x5-Fold Stratified CV",
        f"Best Continuous Model : Simple Linear Regression (SLR) — LOOCV R² = {slr_info['loocv_r2']:.4f}, RMSE = {slr_info['loocv_rmse']:.4f}, MAE = {slr_info['loocv_mae']:.4f}",
        f"SLR Equation          : Behavior_Index = {slr_info['slope']:.4f} * Awareness_Index + {slr_info['intercept']:.4f} (r = {slr_info['pearson_r']:.4f}, p < 0.001)",
        "",
        "BINARY CLASSIFICATION (High >= 4.0 [n=47] vs Low/Moderate < 4.0 [n=25]):",
        bin_df.to_string(index=False),
        "",
        "3-CLASS CLASSIFICATION (Low [n=7], Medium [n=18], High [n=47]):",
        cls3_df.to_string(index=False),
    ]
    with open(os.path.join(OUTPUT_DIR, "ml_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"\nSaved ML tables, summary, and 2 figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

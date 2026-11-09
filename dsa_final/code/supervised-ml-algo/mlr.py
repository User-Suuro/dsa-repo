"""
mlr.py
======
Multiple Linear Regression (MLR)
Study: AI and the Environment: An Analysis of Public Awareness

Predictors (X):
  - Q2 AI Usage Frequency (usage_freq_n)
  - Q3 Energy Awareness (aware_energy_n)
  - Q4 Carbon Awareness (aware_carbon_n)
  - Q7 Water Concern (c_water)
  - Q7 Energy Concern (c_energy)
  - Q7 Carbon Concern (c_carbon)
Target (y): Continuous Behavior_Index [1.0 to 5.0]

Outputs:
  - `output/supervised-ml-algo/mlr_metrics.csv`
  - `output/supervised-ml-algo/mlr_coefficients.csv`
  - `output/supervised-ml-algo/figures/mlr_fit.png`
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CODE_DIR, "..", ".."))
INPUT_CSV = os.path.join(PROJECT_DIR, "output", "data-cleaning", "reduction", "integrated_master_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output", "supervised-ml-algo")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

FEATURE_COLS = [
    "usage_freq_n",
    "aware_energy_n",
    "aware_carbon_n",
    "c_water",
    "c_energy",
    "c_carbon",
]
FEATURE_LABELS = {
    "usage_freq_n": "Q2 Usage Freq",
    "aware_energy_n": "Q3 Energy Aware",
    "aware_carbon_n": "Q4 Carbon Aware",
    "c_water": "Q7 Water Concern",
    "c_energy": "Q7 Energy Concern",
    "c_carbon": "Q7 Carbon Concern",
}


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 70)
    print("  MULTIPLE LINEAR REGRESSION (MLR)")
    print("=" * 70)

    data = pd.read_csv(INPUT_CSV)
    for c in FEATURE_COLS:
        if c not in data.columns:
            data[c] = 0.0
        data[c] = data[c].fillna(0.0)

    X = data[FEATURE_COLS]
    y = data["Behavior_Index"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"RMSE              : {rmse:.4f}")
    print(f"MAE               : {mae:.4f}")
    print(f"R-squared (R2)    : {r2:.4f}")
    print("\nModel Coefficients:")
    coef_rows = []
    for feat, coef in zip(FEATURE_COLS, model.coef_):
        lbl = FEATURE_LABELS.get(feat, feat)
        print(f"  {lbl:<20}: {coef:+.4f}")
        coef_rows.append({"feature": feat, "display_name": lbl, "coefficient": round(coef, 4)})
    coef_rows.append({"feature": "intercept", "display_name": "Model Intercept", "coefficient": round(model.intercept_, 4)})
    print(f"  {'Intercept':<20}: {model.intercept_:+.4f}")

    pd.DataFrame(coef_rows).to_csv(os.path.join(OUTPUT_DIR, "mlr_coefficients.csv"), index=False)

    metrics_df = pd.DataFrame([{
        "model": "Multiple Linear Regression (MLR)",
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
    }])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "mlr_metrics.csv"), index=False)

    # Plot Actual vs. Predicted
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.scatter(y_test, y_pred, color="#2166ac", alpha=0.40, s=22, label="Predicted vs. Actual")
    min_v, max_v = 1.0, 5.0
    ax.plot([min_v, max_v], [min_v, max_v], color="#b2182b", linestyle="--", linewidth=2.0, label="Ideal Fit (y = x)")

    m_slope, b_inter = np.polyfit(y_test, y_pred, 1)
    x_line = np.linspace(min_v, max_v, 100)
    ax.plot(x_line, m_slope * x_line + b_inter, color="#1b7837", linewidth=2.0, label=f"Fit Trend (slope = {m_slope:.2f})")

    ax.text(
        0.05, 0.88,
        f"R² = {r2:.4f}\nRMSE = {rmse:.4f}\nN = {len(y_test):,}",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )

    ax.set_xlabel("Actual Behavior Index")
    ax.set_ylabel("Predicted Behavior Index")
    ax.set_title("Multiple Linear Regression: Actual vs. Predicted Behavior", fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)

    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "mlr_fit.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {fig_path}")


if __name__ == "__main__":
    main()


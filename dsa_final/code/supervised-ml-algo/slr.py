"""
slr.py
======
Simple Linear Regression (SLR)
Study: AI and the Environment: An Analysis of Public Awareness

Predictor (X): Overall Awareness_Index (mean of Q3 Energy Awareness and Q4 Carbon Awareness)
Target (y)   : Continuous Behavior_Index (mean of Q5 Consider Impact and Q6 Avoid Unnecessary Use)

Outputs:
  - `output/supervised-ml-algo/slr_metrics.csv`
  - `output/supervised-ml-algo/figures/slr_fit.png`
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


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 70)
    print("  SIMPLE LINEAR REGRESSION (SLR)")
    print("=" * 70)

    data = pd.read_csv(INPUT_CSV)
    X = data[["Awareness_Index"]].values
    y = data["Behavior_Index"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)

    print(f"Slope (Coefficient): {slope:.4f}")
    print(f"Intercept          : {intercept:.4f}")
    print(f"Mean Squared Error : {mse:.4f}")
    print(f"RMSE               : {rmse:.4f}")
    print(f"R-squared (R2)     : {r2:.4f}")

    metrics_df = pd.DataFrame([{
        "model": "Simple Linear Regression (SLR)",
        "predictor": "Awareness_Index",
        "target": "Behavior_Index",
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
    }])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "slr_metrics.csv"), index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.scatter(X_test, y_test, color="#6baed6", alpha=0.45, s=22, label="Actual Observations (Test Set)")

    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    ax.plot(x_line, y_line, color="#b2182b", linewidth=2.5, label=f"Fit Line: y = {slope:.2f}x + {intercept:.2f}")

    ax.text(
        0.05, 0.88,
        f"R² = {r2:.4f}\nRMSE = {rmse:.4f}\nN = {len(y_test):,}",
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )

    ax.set_xlabel("Awareness Index (Q3 Energy & Q4 Carbon Awareness)")
    ax.set_ylabel("Behavior Index (Q5 & Q6 Eco-Conscious Action)")
    ax.set_title("Simple Linear Regression: Awareness vs. Responsible Behavior", fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)

    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "slr_fit.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {fig_path}")


if __name__ == "__main__":
    main()


"""
naive_bayes.py
==============
Gaussian Naive Bayes Classifier
Study: AI and the Environment: An Analysis of Public Awareness

Predictors (X):
  - Q2 AI Usage Freq (usage_freq_n)
  - Q3 Energy Awareness (aware_energy_n)
  - Q4 Carbon Awareness (aware_carbon_n)
  - Overall Awareness_Index
Target (y): Categorical `behavior_level` (Low, Medium, High)

Outputs:
  - `output/supervised-ml-algo/naive_bayes_metrics.csv`
  - `output/supervised-ml-algo/figures/naive_bayes_cm.png`
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CODE_DIR, "..", ".."))
INPUT_CSV = os.path.join(PROJECT_DIR, "output", "data-cleaning", "reduction", "integrated_master_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output", "supervised-ml-algo")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

FEATURE_COLS = ["usage_freq_n", "aware_energy_n", "aware_carbon_n", "Awareness_Index"]


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 70)
    print("  GAUSSIAN NAIVE BAYES CLASSIFIER")
    print("=" * 70)

    data = pd.read_csv(INPUT_CSV)
    X = data[FEATURE_COLS].values
    y = data["behavior_level"].values

    class_names = ["Low", "Medium", "High"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    metrics_df = pd.DataFrame([{
        "model": "Gaussian Naive Bayes",
        "accuracy": round(accuracy, 4),
    }])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "naive_bayes_metrics.csv"), index=False)

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontweight="bold")
    ax.set_yticklabels(class_names, fontweight="bold")
    ax.set_xlabel("Predicted Behavior Level")
    ax.set_ylabel("Actual Behavior Level")
    ax.set_title(f"Naive Bayes Confusion Matrix (Acc = {accuracy * 100:.1f}%)", fontweight="bold", pad=12)

    total_pts = cm.sum()
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm[i, j]
            pct = (val / max(total_pts, 1)) * 100.0
            color = "white" if val > thresh else "#1e293b"
            ax.text(j, i, f"{val:,}\n({pct:.1f}%)", ha="center", va="center", color=color, fontweight="bold", fontsize=8.5)

    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "naive_bayes_cm.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {fig_path}")


if __name__ == "__main__":
    main()


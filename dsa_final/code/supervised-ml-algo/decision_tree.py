"""
decision_tree.py
================
Decision Tree Classifier
Study: AI and the Environment: An Analysis of Public Awareness

Predictors (X):
  - Q2 AI Usage Freq (usage_freq_n)
  - Q3 Energy Awareness (aware_energy_n)
  - Q4 Carbon Awareness (aware_carbon_n)
  - Overall Awareness_Index
Target (y): Categorical `behavior_level` (Low, Medium, High)

Outputs:
  - `output/supervised-ml-algo/decision_tree_metrics.csv`
  - `output/supervised-ml-algo/figures/decision_tree.png`
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CODE_DIR, "..", ".."))
INPUT_CSV = os.path.join(PROJECT_DIR, "output", "data-cleaning", "reduction", "integrated_master_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output", "supervised-ml-algo")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

FEATURE_COLS = ["usage_freq_n", "aware_energy_n", "aware_carbon_n", "Awareness_Index"]
FEATURE_NAMES = ["Q2 Usage", "Q3 Energy Aware", "Q4 Carbon Aware", "Awareness Index"]


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 70)
    print("  DECISION TREE CLASSIFIER")
    print("=" * 70)

    data = pd.read_csv(INPUT_CSV)
    X = data[FEATURE_COLS].values
    y = data["behavior_level"].values

    class_names = ["Low", "Medium", "High"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("\nFeature Importances:")
    for feat, imp in zip(FEATURE_NAMES, model.feature_importances_):
        print(f"  {feat:<18}: {imp * 100:.2f}%")

    metrics_df = pd.DataFrame([{
        "model": "Decision Tree Classifier",
        "accuracy": round(accuracy, 4),
        "max_depth": 3,
    }])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "decision_tree_metrics.csv"), index=False)

    # Visualize tree
    fig, ax = plt.subplots(figsize=(13.0, 7.5))
    tree.plot_tree(
        model,
        feature_names=FEATURE_NAMES,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(f"Decision Tree for Responsible AI Behavior (Accuracy = {accuracy * 100:.1f}%)", fontweight="bold", pad=12)

    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "decision_tree.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved figure to: {fig_path}")


if __name__ == "__main__":
    main()

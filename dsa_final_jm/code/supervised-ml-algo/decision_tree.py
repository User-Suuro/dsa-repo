"""
decision_tree.py
================
Decision Tree Classifier
Study: Assessing College Students Awareness and Preparedness Toward Flood Risks

Predictors (X):
  - awareness_score (Overall Flood Risk Awareness)
  - flood_history_n (Flood Experience Intensity)
  - aw_warning_index (Warning Signals & Weather Monitoring)
  - aw_local_hazard_index (Residence Risk & Emergency Contacts)
  - aw_safety_health_index (Evacuation Routes & Health Risks)
Target (y): Categorical preparedness_level (Low, Moderate, High)

Outputs -> `output/supervised-ml-algo/`:
  - `decision_tree_metrics.csv`
  - `figures/decision_tree.png`
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

FEATURE_COLS = [
    "awareness_score",
    "flood_history_n",
    "aw_warning_index",
    "aw_local_hazard_index",
    "aw_safety_health_index",
]
FEATURE_NAMES = [
    "Awareness Score",
    "Flood Experience",
    "Warning Index",
    "Local Hazard Index",
    "Safety Health Index",
]

plt.rcParams.update({
    "figure.dpi": 140,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "font.family": "sans-serif",
})


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 70)
    print("  DECISION TREE CLASSIFIER — FLOOD STUDY")
    print("=" * 70)

    data = pd.read_csv(INPUT_CSV)
    for c in FEATURE_COLS:
        data[c] = data[c].fillna(0.0)

    X = data[FEATURE_COLS].values
    y = data["preparedness_level"].values

    class_names = ["Low", "Moderate", "High"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    print("\nFeature Importances:")
    imp_rows = []
    for feat, imp in zip(FEATURE_NAMES, model.feature_importances_):
        print(f"  {feat:<24}: {imp * 100:.2f}%")
        imp_rows.append({"feature": feat, "importance": round(imp, 4), "importance_pct": round(imp * 100, 2)})

    pd.DataFrame(imp_rows).to_csv(os.path.join(OUTPUT_DIR, "feature_importances.csv"), index=False)

    metrics_df = pd.DataFrame([{
        "model": "Decision Tree Classifier",
        "accuracy": round(accuracy, 4),
        "max_depth": 3,
        "n_classes": len(class_names),
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
    ax.set_title(f"Decision Tree for Flood Preparedness Level (Accuracy = {accuracy * 100:.1f}%)", fontweight="bold", pad=12)

    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "decision_tree.png")
    fig.savefig(fig_path, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved figure to: {fig_path}")


if __name__ == "__main__":
    main()

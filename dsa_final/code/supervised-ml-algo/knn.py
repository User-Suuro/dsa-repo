"""
knn.py
======
K-Nearest Neighbors (KNN) Classifier
Study: AI and the Environment: An Analysis of Public Awareness

Predictors (X):
  - Q2 AI Usage Freq (usage_freq_n)
  - Q3 Energy Awareness (aware_energy_n)
  - Q4 Carbon Awareness (aware_carbon_n)
  - Overall Awareness_Index
Target (y): Categorical `behavior_level` (Low, Medium, High)

Outputs:
  - `output/supervised-ml-algo/knn_metrics.csv`
  - `output/supervised-ml-algo/figures/knn_accuracy_k.png`
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

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
    print("  K-NEAREST NEIGHBORS (KNN) CLASSIFIER")
    print("=" * 70)

    data = pd.read_csv(INPUT_CSV)
    X = data[FEATURE_COLS].values
    y = data["behavior_level"].values

    class_names = ["Low", "Medium", "High"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    k_chosen = 5
    model = KNeighborsClassifier(n_neighbors=k_chosen)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc_5 = accuracy_score(y_test, y_pred)

    print(f"Accuracy (k = {k_chosen}): {acc_5 * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Evaluate accuracy across k = 1..15
    k_range = range(1, 16)
    accuracies = []
    for k_val in k_range:
        knn_k = KNeighborsClassifier(n_neighbors=k_val)
        knn_k.fit(X_train_scaled, y_train)
        pred_k = knn_k.predict(X_test_scaled)
        accuracies.append(accuracy_score(y_test, pred_k))

    best_k = list(k_range)[np.argmax(accuracies)]
    best_acc = max(accuracies)
    print(f"\nBest k: {best_k} with Accuracy: {best_acc * 100:.2f}%")

    metrics_df = pd.DataFrame([{
        "model": "K-Nearest Neighbors (KNN)",
        "chosen_k": k_chosen,
        "chosen_k_accuracy": round(acc_5, 4),
        "optimal_k": best_k,
        "optimal_k_accuracy": round(best_acc, 4),
    }])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "knn_metrics.csv"), index=False)

    # Plot Accuracy vs. k
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(k_range, [a * 100 for a in accuracies], marker="o", color="#2166ac", linewidth=2.0)
    ax.axvline(best_k, color="#b2182b", linestyle="--", linewidth=1.5, label=f"Best k = {best_k} ({best_acc * 100:.1f}%)")
    ax.set_xlabel("Number of Neighbors (k)")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_xticks(list(k_range))
    ax.set_title("KNN Classification Accuracy Across Different k Values", fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)

    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "knn_accuracy_k.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {fig_path}")


if __name__ == "__main__":
    main()


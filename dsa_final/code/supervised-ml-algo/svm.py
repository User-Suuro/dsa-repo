"""
svm.py
======
Support Vector Machine (SVM) Classifier
Study: AI and the Environment: An Analysis of Public Awareness

Predictors (X):
  - Q2 AI Usage Freq (usage_freq_n)
  - Q3 Energy Awareness (aware_energy_n)
  - Q4 Carbon Awareness (aware_carbon_n)
  - Overall Awareness_Index
Target (y): Categorical `behavior_level` (Low, Medium, High)

Outputs:
  - `output/supervised-ml-algo/svm_metrics.csv`
  - `output/supervised-ml-algo/figures/svm_decision_boundaries.png`
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
    print("  SUPPORT VECTOR MACHINE (SVM) CLASSIFIER")
    print("=" * 70)

    data = pd.read_csv(INPUT_CSV)
    X = data[FEATURE_COLS].values
    y = data["behavior_level"].values

    class_names = ["Low", "Medium", "High"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel="rbf", C=1.0, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy (RBF Kernel, C=1.0): {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    metrics_df = pd.DataFrame([{
        "model": "Support Vector Machine (SVM)",
        "kernel": "rbf",
        "C": 1.0,
        "accuracy": round(accuracy, 4),
    }])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "svm_metrics.csv"), index=False)

    # 2D Decision Boundary visualization using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(scaler.fit_transform(X))

    # Map labels to integers for contour plotting
    label_map = {"Low": 0, "Medium": 1, "High": 2}
    y_num = np.array([label_map[lbl] for lbl in y])

    svm_2d = SVC(kernel="rbf", C=1.0, random_state=42)
    svm_2d.fit(X_pca, y_num)

    x_min, x_max = X_pca[:, 0].min() - 0.8, X_pca[:, 0].max() + 0.8
    y_min, y_max = X_pca[:, 1].min() - 0.8, X_pca[:, 1].max() + 0.8
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="Blues")

    scatter_colors = ["#b2182b", "#ef8a62", "#2166ac"]
    for idx_c, lbl in enumerate(class_names):
        mask = (y_num == idx_c)
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"{lbl} Restraint", color=scatter_colors[idx_c], alpha=0.5, s=20, edgecolors="none")

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(f"SVM Decision Boundaries in 2D PCA Space (Acc = {accuracy * 100:.1f}%)", fontweight="bold", pad=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", loc="upper right", fontsize=8.2)

    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "svm_decision_boundaries.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {fig_path}")


if __name__ == "__main__":
    main()


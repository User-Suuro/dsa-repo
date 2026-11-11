"""
kmeans.py
=========
Unsupervised Machine Learning — K-Means Clustering
Study: Assessing College Students Awareness and Preparedness Toward Flood Risks

Performs:
  1. Standardizes core construct features (warnings, residence hazard, safety health, supplies, planning, flood history).
  2. Evaluates k = 2 to 6 using Silhouette Score and Inertia (Elbow curve).
  3. Fits optimal K-Means (k = 4) to discover empirical student disaster-response personas.
  4. Profiles clusters with construct means and cross-tabulates with rule-based gap_typology.
  5. Projects clusters into 2D PCA latent space with marked centroids.

Outputs -> `output/unsupervised-ml-algo/`:
  - `kmeans_evaluation_k2_k6.csv` / `clustering_evaluation_k2_k6.csv`
  - `kmeans_profiles.csv` / `cluster_profiles.csv`
  - `kmeans_vs_gap_typology.csv` / `cluster_vs_gap_typology.csv`
  - `figures/fig1_kmeans_elbow_silhouette.png`
  - `figures/fig2_kmeans_pca_clusters.png`
  - `figures/fig3_kmeans_cluster_profiles.png`
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CODE_DIR, "..", ".."))
INPUT_CSV = os.path.join(PROJECT_DIR, "output", "data-cleaning", "reduction", "integrated_master_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output", "unsupervised-ml-algo")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

CONSTRUCT_COLS = [
    "aw_warning_index",
    "aw_local_hazard_index",
    "aw_safety_health_index",
    "prep_supplies_index",
    "prep_planning_index",
    "flood_history_n",
]

plt.rcParams.update({
    "figure.dpi": 140,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "font.family": "sans-serif",
})


def evaluate_k_range(X_scaled: np.ndarray, k_range: range = range(2, 7)) -> pd.DataFrame:
    """Evaluate K-Means across k = 2..6 to find optimal cluster count."""
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results.append({
            "k": k,
            "silhouette_score": round(float(sil), 4),
            "inertia_wssse": round(float(km.inertia_), 2),
        })
    return pd.DataFrame(results)


def plot_fig1_elbow_silhouette(eval_df: pd.DataFrame) -> None:
    """Figure 1: Dual-Axis Elbow Curve (Inertia) & Silhouette Score across k = 2..6."""
    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))

    color_w = "#C0392B"
    ax1.set_xlabel("Number of Clusters (k)", fontweight="bold")
    ax1.set_ylabel("Inertia / WSSSE", color=color_w, fontweight="bold")
    l1 = ax1.plot(eval_df["k"], eval_df["inertia_wssse"], color=color_w, marker="o", linewidth=2.0, label="Elbow Curve (Inertia)")
    ax1.tick_params(axis="y", labelcolor=color_w)

    ax2 = ax1.twinx()
    color_s = "#2980B9"
    ax2.set_ylabel("Silhouette Score", color=color_s, fontweight="bold")
    l2 = ax2.plot(eval_df["k"], eval_df["silhouette_score"], color=color_s, marker="s", linestyle="--", linewidth=2.0, label="Silhouette Score")
    ax2.tick_params(axis="y", labelcolor=color_s)
    ax2.set_ylim(0.0, max(eval_df["silhouette_score"].max() + 0.15, 0.6))

    for k_val, sil in zip(eval_df["k"], eval_df["silhouette_score"]):
        ax2.text(k_val, sil + 0.02, f"{sil:.3f}", color=color_s, fontweight="bold", ha="center", fontsize=8.2)

    ax1.set_xticks(eval_df["k"])
    ax1.set_title("Figure 1: K-Means Cluster Optimization (Elbow & Silhouette)", fontweight="bold", pad=12)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_kmeans_elbow_silhouette.png"), bbox_inches="tight")
    plt.close(fig)


def plot_fig2_pca_clusters(X_pca: np.ndarray, labels: np.ndarray, profile_df: pd.DataFrame) -> None:
    """Figure 2: 2D PCA Scatter Plot with K-Means Clusters and Centroids."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    colors = ["#2980B9", "#27AE60", "#E67E22", "#8E44AD"]

    for c_id in sorted(profile_df["cluster_id"].unique()):
        mask = labels == c_id
        row = profile_df[profile_df["cluster_id"] == c_id].iloc[0]
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=colors[c_id % len(colors)],
            label=f"Cluster {c_id}: {row['persona_name']} ({row['share_pct']:.1f}%)",
            alpha=0.5,
            s=22,
            edgecolors="none",
        )

    # Plot Centroids
    for c_id in sorted(profile_df["cluster_id"].unique()):
        mask = labels == c_id
        cent_x = X_pca[mask, 0].mean()
        cent_y = X_pca[mask, 1].mean()
        ax.scatter(cent_x, cent_y, s=180, c="black", marker="X", edgecolors="white", linewidth=1.5, zorder=5)

    ax.set_xlabel("PCA Component 1 (General Preparedness & Resource Axis)")
    ax.set_ylabel("PCA Component 2 (Hazard Awareness & Warning Axis)")
    ax.set_title("Figure 2: K-Means Personas Projected in 2D Latent Space (k = 4)", fontweight="bold", pad=12)
    ax.legend(frameon=True, loc="upper right", fontsize=8.2)
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_kmeans_pca_clusters.png"), bbox_inches="tight")
    plt.close(fig)


def plot_fig3_kmeans_cluster_profiles(profile_df: pd.DataFrame) -> None:
    """Figure 3: Mean Construct Scores across the 4 K-Means Personas."""
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    x = np.arange(len(profile_df))
    w = 0.24

    c1 = ax.bar(x - w, profile_df["mean_awareness"], w, label="Awareness Score", color="#3498DB")
    c2 = ax.bar(x, profile_df["mean_preparedness"], w, label="Preparedness Score", color="#E67E22")
    c3 = ax.bar(x + w, profile_df["mean_supplies"], w, label="Emergency Supplies Index", color="#2ECC71")

    ax.bar_label(c1, fmt="%.2f", padding=2, fontsize=7.8)
    ax.bar_label(c2, fmt="%.2f", padding=2, fontsize=7.8)
    ax.bar_label(c3, fmt="%.2f", padding=2, fontsize=7.8)

    ax.set_xticks(x)
    x_labels = [f"Cluster {int(r['cluster_id'])}\n{r['persona_name']}\n(n={int(r['count']):,}, {r['share_pct']:.1f}%)" for _, r in profile_df.iterrows()]
    ax.set_xticklabels(x_labels, fontweight="bold", fontsize=8.0)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Construct Score (1.0 to 5.0 Likert)")
    ax.set_title("Figure 3: Empirical Construct Profiles Across K-Means Personas", fontweight="bold", pad=12)
    ax.legend(frameon=True, loc="upper right", fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_kmeans_cluster_profiles.png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 70)
    print("  K-MEANS CLUSTERING (k = 4 PERSONAS) — FLOOD STUDY")
    print("=" * 70)

    data = pd.read_csv(INPUT_CSV)
    for c in CONSTRUCT_COLS:
        data[c] = data[c].fillna(data[c].median())

    X = data[CONSTRUCT_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Evaluate k = 2..6
    eval_df = evaluate_k_range(X_scaled)
    eval_df.to_csv(os.path.join(OUTPUT_DIR, "kmeans_evaluation_k2_k6.csv"), index=False)
    eval_df.to_csv(os.path.join(OUTPUT_DIR, "clustering_evaluation_k2_k6.csv"), index=False)
    plot_fig1_elbow_silhouette(eval_df)
    print("Cluster Optimization across k = 2..6:")
    print(eval_df.to_string(index=False))

    # 2. Fit K-Means with k = 4
    k_optimal = 4
    km = KMeans(n_clusters=k_optimal, random_state=42, n_init=15)
    labels = km.fit_predict(X_scaled)
    data["kmeans_cluster"] = labels

    sil_4 = silhouette_score(X_scaled, labels)
    print(f"\nFitted K-Means (k = {k_optimal}) Silhouette Score: {sil_4:.4f}")

    # 3. Persona labeling based on construct means
    total_n = len(data)
    prof_rows = []
    for c_id in range(k_optimal):
        subset = data[data["kmeans_cluster"] == c_id]
        mean_aw = subset["awareness_score"].mean()
        mean_prep = subset["preparedness_score"].mean()
        mean_supplies = subset["prep_supplies_index"].mean()
        mean_planning = subset["prep_planning_index"].mean()
        mean_exp = subset["flood_history_n"].mean()

        # Persona identification
        if mean_aw >= 4.0 and mean_prep >= 3.8:
            persona = "Resilient Ready Guardians"
        elif mean_aw >= 3.4 and mean_prep < 2.7:
            persona = "Aware but Unprepared"
        elif mean_aw >= 3.0 and mean_prep >= 2.6:
            persona = "Low-Exposure Learners"
        else:
            persona = "Vulnerable Novices"

        prof_rows.append({
            "cluster_id": c_id,
            "persona_name": persona,
            "count": len(subset),
            "share_pct": round(len(subset) / total_n * 100, 2),
            "mean_awareness": round(mean_aw, 3),
            "mean_preparedness": round(mean_prep, 3),
            "mean_supplies": round(mean_supplies, 3),
            "mean_planning": round(mean_planning, 3),
            "mean_flood_history": round(mean_exp, 3),
        })

    profile_df = pd.DataFrame(prof_rows).sort_values("cluster_id").reset_index(drop=True)
    profile_df.to_csv(os.path.join(OUTPUT_DIR, "kmeans_profiles.csv"), index=False)
    profile_df.to_csv(os.path.join(OUTPUT_DIR, "cluster_profiles.csv"), index=False)
    print("\nK-Means Cluster Profiles (k = 4):")
    print(profile_df.to_string(index=False))

    # 4. Cross-tabulate with gap_typology
    cross_tab = pd.crosstab(data["kmeans_cluster"], data["gap_typology"], margins=True)
    cross_tab.to_csv(os.path.join(OUTPUT_DIR, "kmeans_vs_gap_typology.csv"))
    cross_tab.to_csv(os.path.join(OUTPUT_DIR, "cluster_vs_gap_typology.csv"))

    # 5. PCA Projection & Figures
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plot_fig2_pca_clusters(X_pca, labels, profile_df)
    plot_fig3_kmeans_cluster_profiles(profile_df)
    print(f"Generated figures in {FIG_DIR}")


if __name__ == "__main__":
    main()

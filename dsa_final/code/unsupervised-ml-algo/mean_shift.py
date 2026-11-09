"""
mean_shift.py
=============
Unsupervised Machine Learning — Mean-Shift Clustering
Study: AI and the Environment: An Analysis of Public Awareness

Performs:
  1. Standardizes the 5 survey constructs (Q2..Q6).
  2. Estimates optimal bandwidth using `estimate_bandwidth(quantile=0.25)`.
  3. Fits Mean-Shift (density-based mode-seeking clustering) to automatically
     discover the natural number of clusters without presetting k.
  4. Profiles discovered density modes with construct means and shares.
  5. Projects density clusters into 2D PCA space with marked cluster centers.

Outputs -> `output/unsupervised-ml-algo/`:
  - `mean_shift_metrics.csv`
  - `mean_shift_profiles.csv`
  - `figures/fig4_mean_shift_clusters.png`
  - `figures/fig5_mean_shift_profiles.png`
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CODE_DIR, "..", ".."))
INPUT_CSV = os.path.join(PROJECT_DIR, "output", "data-cleaning", "reduction", "integrated_master_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output", "unsupervised-ml-algo")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

CONSTRUCT_COLS = [
    "usage_freq_n",
    "aware_energy_n",
    "aware_carbon_n",
    "consider_impact_n",
    "avoid_use_n",
]


def plot_fig4_mean_shift_clusters(X_pca: np.ndarray, labels: np.ndarray, centers_pca: np.ndarray, n_clusters: int) -> None:
    """Figure 4: Mean-Shift Clusters Projected into 2D PCA Space with Centroids."""
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    cmap = plt.get_cmap("viridis", n_clusters)

    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap, alpha=0.45, s=20, edgecolors="none")

    # Mark cluster centers
    ax.scatter(
        centers_pca[:, 0],
        centers_pca[:, 1],
        s=160,
        c="red",
        marker="X",
        edgecolors="black",
        linewidth=1.5,
        label="Discovered Density Modes (Centers)",
        zorder=6,
    )

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean-Shift Cluster Label")
    ax.set_xlabel("Principal Component 1 (General Awareness & Behavior Axis)")
    ax.set_ylabel("Principal Component 2 (Usage Frequency Axis)")
    ax.set_title(f"Figure 4: Mean-Shift Density Clustering in PCA Space ({n_clusters} Discovered Modes)", fontweight="bold", pad=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", loc="upper right", fontsize=8.2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_mean_shift_clusters.png"), bbox_inches="tight")
    plt.close(fig)


def plot_fig5_mean_shift_profiles(profile_df: pd.DataFrame) -> None:
    """Figure 5: Mean Construct Scores Across Discovered Mean-Shift Density Clusters."""
    fig, ax = plt.subplots(figsize=(8.8, 4.4))

    x = np.arange(len(profile_df))
    w = 0.26

    c1 = ax.bar(x - w, profile_df["mean_q2_usage"], w, label="Q2 AI Usage Freq", color="#94a3b8", edgecolor="white")
    c2 = ax.bar(x, profile_df["mean_awareness"], w, label="Awareness Index (Q3-Q4)", color="#2166ac", edgecolor="white")
    c3 = ax.bar(x + w, profile_df["mean_behavior"], w, label="Behavior Index (Q5-Q6)", color="#ef8a62", edgecolor="white")

    ax.bar_label(c1, fmt="%.2f", padding=2, fontsize=7.8)
    ax.bar_label(c2, fmt="%.2f", padding=2, fontsize=7.8)
    ax.bar_label(c3, fmt="%.2f", padding=2, fontsize=7.8)

    ax.set_xticks(x)
    x_labels = [f"Mode {int(r['cluster_id'])}\n({r['share_pct']:.1f}%)\nn={int(r['count']):,}" for _, r in profile_df.iterrows()]
    ax.set_xticklabels(x_labels, fontweight="bold", fontsize=8.2)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Mean Construct Score (1-5 Scale)")
    ax.set_title(f"Figure 5: Construct Means Across Mean-Shift Density Clusters (N = {int(profile_df['count'].sum()):,})", fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=8.2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_mean_shift_profiles.png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 70)
    print("  MEAN-SHIFT CLUSTERING (DENSITY-BASED MODE DISCOVERY)")
    print("=" * 70)

    data = pd.read_csv(INPUT_CSV)
    X = data[CONSTRUCT_COLS].values
    total_n = len(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Bandwidth estimation
    print("Estimating bandwidth (quantile = 0.25, n_samples = 1000)...")
    bandwidth = estimate_bandwidth(X_scaled, quantile=0.25, n_samples=1000, random_state=42)
    print(f"Estimated bandwidth: {bandwidth:.4f}")

    # 2. Fit Mean-Shift
    mean_shift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = mean_shift.fit_predict(X_scaled)
    n_clusters = len(np.unique(labels))
    cluster_centers = mean_shift.cluster_centers_

    print(f"Number of natural clusters found by Mean-Shift: {n_clusters}")

    sil_score = silhouette_score(X_scaled, labels) if n_clusters > 1 else 0.0
    print(f"Silhouette Score: {sil_score:.4f}")

    metrics_df = pd.DataFrame([{
        "model": "Mean-Shift Clustering",
        "estimated_bandwidth": round(bandwidth, 4),
        "clusters_discovered": n_clusters,
        "silhouette_score": round(sil_score, 4),
    }])
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "mean_shift_metrics.csv"), index=False)

    # 3. Profile discovered modes
    data["mean_shift_cluster"] = labels
    profiles = []
    for c_id in sorted(np.unique(labels)):
        sub = data[data["mean_shift_cluster"] == c_id]
        aw = sub["Awareness_Index"].mean()
        bh = sub["Behavior_Index"].mean()
        us = sub["usage_freq_n"].mean()
        gap = sub["Awareness_Behavior_Gap"].mean()

        profiles.append({
            "cluster_id": c_id,
            "count": len(sub),
            "share_pct": round(len(sub) * 100.0 / total_n, 1),
            "mean_q2_usage": round(us, 2),
            "mean_q3_energy": round(sub["aware_energy_n"].mean(), 2),
            "mean_q4_carbon": round(sub["aware_carbon_n"].mean(), 2),
            "mean_q5_consider": round(sub["consider_impact_n"].mean(), 2),
            "mean_q6_avoid": round(sub["avoid_use_n"].mean(), 2),
            "mean_awareness": round(aw, 2),
            "mean_behavior": round(bh, 2),
            "mean_gap": round(gap, 2),
        })

    profile_df = pd.DataFrame(profiles)
    profile_df.to_csv(os.path.join(OUTPUT_DIR, "mean_shift_profiles.csv"), index=False)
    print("\nMean-Shift Discovered Cluster Profiles:")
    print(profile_df[["cluster_id", "count", "share_pct", "mean_q2_usage", "mean_awareness", "mean_behavior", "mean_gap"]].to_string(index=False))

    # 4. Visualizations
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    centers_pca = pca.transform(cluster_centers)

    plot_fig4_mean_shift_clusters(X_pca, labels, centers_pca, n_clusters)
    plot_fig5_mean_shift_profiles(profile_df)

    print(f"\n[Mean-Shift] Successfully exported 2 CSVs and 2 figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

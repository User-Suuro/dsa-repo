"""
kmeans.py
=========
Unsupervised Machine Learning — K-Means Clustering
Study: AI and the Environment: An Analysis of Public Awareness

Performs:
  1. Standardizes the 5 survey constructs (Q2..Q6).
  2. Evaluates k = 2 to 6 using Silhouette Score and Inertia (Elbow curve).
  3. Fits optimal K-Means (k = 4) to discover empirical user personas.
  4. Profiles clusters with construct means and cross-tabulates with rule-based gap_type.
  5. Projects clusters into 2D PCA latent space with marked centroids.

Outputs -> `output/unsupervised-ml-algo/`:
  - `kmeans_evaluation_k2_k6.csv`
  - `kmeans_profiles.csv`
  - `kmeans_vs_gap_typology.csv`
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
    "usage_freq_n",
    "aware_energy_n",
    "aware_carbon_n",
    "consider_impact_n",
    "avoid_use_n",
]


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
    fig, ax1 = plt.subplots(figsize=(7.4, 4.2))

    color_w = "#b2182b"
    ax1.set_xlabel("Number of Clusters (k)", fontweight="bold")
    ax1.set_ylabel("Inertia / WSSSE", color=color_w, fontweight="bold")
    l1 = ax1.plot(eval_df["k"], eval_df["inertia_wssse"], color=color_w, marker="o", linewidth=2.0, label="Elbow Curve (Inertia)")
    ax1.tick_params(axis="y", labelcolor=color_w)

    ax2 = ax1.twinx()
    color_s = "#2166ac"
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
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    colors = ["#2166ac", "#1b7837", "#ef8a62", "#9970ab"]

    for c_id in sorted(np.unique(labels)):
        mask = (labels == c_id)
        p_row = profile_df[profile_df["cluster_id"] == c_id].iloc[0]
        lbl = f"Cluster {c_id}: {p_row['persona_name']} (n={int(p_row['count']):,})"
        colr = colors[c_id % len(colors)]
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.45, color=colr, label=lbl, s=18, edgecolors="none")

        c_x = X_pca[mask, 0].mean()
        c_y = X_pca[mask, 1].mean()
        ax.scatter(c_x, c_y, color="black", marker="X", s=90, edgecolors="white", linewidth=1.5, zorder=5)

    ax.set_xlabel("Principal Component 1 (General Awareness & Behavior Axis)")
    ax.set_ylabel("Principal Component 2 (Usage Frequency Axis)")
    ax.set_ylim(-4.2, 3.2)
    ax.set_title(f"Figure 2: K-Means (k = 4) Clusters in PCA Latent Space (N = {len(labels):,})", fontweight="bold", pad=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cbd5e1", loc="upper right", fontsize=7.8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_kmeans_pca_clusters.png"), bbox_inches="tight")
    plt.close(fig)


def plot_fig3_profiles(profile_df: pd.DataFrame) -> None:
    """Figure 3: Grouped Construct Comparison Across Discovered Personas."""
    fig, ax = plt.subplots(figsize=(9.2, 4.4))

    x = np.arange(len(profile_df))
    w = 0.26

    c1 = ax.bar(x - w, profile_df["mean_q2_usage"], w, label="Q2 AI Usage Freq", color="#94a3b8", edgecolor="white")
    c2 = ax.bar(x, profile_df["mean_awareness"], w, label="Awareness Index (Q3-Q4)", color="#2166ac", edgecolor="white")
    c3 = ax.bar(x + w, profile_df["mean_behavior"], w, label="Behavior Index (Q5-Q6)", color="#ef8a62", edgecolor="white")

    ax.bar_label(c1, fmt="%.2f", padding=2, fontsize=7.8)
    ax.bar_label(c2, fmt="%.2f", padding=2, fontsize=7.8)
    ax.bar_label(c3, fmt="%.2f", padding=2, fontsize=7.8)

    ax.set_xticks(x)
    x_labels = [f"Cluster {int(r['cluster_id'])}\n({r['share_pct']:.1f}%)\n{r['persona_name'].split('(')[0].strip()}" for _, r in profile_df.iterrows()]
    ax.set_xticklabels(x_labels, fontweight="bold", fontsize=8.0)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Mean Construct Score (1-5 Scale)")
    ax.set_title("Figure 3: Mean Construct Scores Across Discovered Personas (K-Means k = 4)", fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=8.2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_kmeans_cluster_profiles.png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 70)
    print("  K-MEANS CLUSTERING (k = 4)")
    print("=" * 70)

    data = pd.read_csv(INPUT_CSV)
    X = data[CONSTRUCT_COLS].values
    total_n = len(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Evaluate k = 2..6
    eval_df = evaluate_k_range(X_scaled)
    eval_df.to_csv(os.path.join(OUTPUT_DIR, "kmeans_evaluation_k2_k6.csv"), index=False)
    print("\nK-Means Evaluation (k = 2 to 6):")
    print(eval_df.to_string(index=False))
    plot_fig1_elbow_silhouette(eval_df)

    # 2. Fit final k = 4
    k_opt = 4
    kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=15)
    labels = kmeans.fit_predict(X_scaled)
    data["kmeans_cluster"] = labels

    # 3. Compute cluster profiles
    profiles = []
    for c_id in sorted(np.unique(labels)):
        sub = data[data["kmeans_cluster"] == c_id]
        aw = sub["Awareness_Index"].mean()
        bh = sub["Behavior_Index"].mean()
        us = sub["usage_freq_n"].mean()
        gap = sub["Awareness_Behavior_Gap"].mean()

        if aw >= 3.8 and bh >= 3.8:
            p_name = "Conscious Pro-Eco Users (High Awareness & Action)"
        elif aw < 2.2 and bh < 2.3:
            p_name = "Passive Unaware Consumers (Very Low Awareness & Restraint)"
        elif us < 3.0 and bh >= 3.5:
            p_name = "Mindful Minimalists (Light Usage & High Restraint)"
        elif gap > 0.2:
            p_name = "Emerging-Awareness Users (Moderate Usage, Action Lags)"
        else:
            p_name = "Moderate Neutral Users (Mid Awareness & Restraint)"

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
            "persona_name": p_name,
        })

    profile_df = pd.DataFrame(profiles)
    profile_df.to_csv(os.path.join(OUTPUT_DIR, "kmeans_profiles.csv"), index=False)
    print("\nDiscovered K-Means Cluster Profiles:")
    print(profile_df[["cluster_id", "count", "share_pct", "mean_q2_usage", "mean_awareness", "mean_behavior", "persona_name"]].to_string(index=False))

    # Cross-tabulation vs gap_type
    crosstab_df = pd.crosstab(data["kmeans_cluster"], data["gap_type"])
    crosstab_df.to_csv(os.path.join(OUTPUT_DIR, "kmeans_vs_gap_typology.csv"))

    # PCA 2D projection
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plot_fig2_pca_clusters(X_pca, labels, profile_df)
    plot_fig3_profiles(profile_df)

    print(f"\n[K-Means] Successfully exported 3 CSVs and 3 figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

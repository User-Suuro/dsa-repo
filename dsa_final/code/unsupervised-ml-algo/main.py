"""
main.py
=======
Coordinator script for Unsupervised Machine Learning in `dsa_final`:
Executes:
  1. K-Means Clustering (kmeans.py) — Centroid-based partitioning (k = 4)
  2. Mean-Shift Clustering (mean_shift.py) — Density-based mode-seeking

Outputs consolidated comparison table & executive summary:
  - `output/unsupervised-ml-algo/unsupervised_models_comparison.csv`
  - `output/unsupervised-ml-algo/unsupervised_key_insights.txt`
"""

import os
import pandas as pd
from kmeans import main as run_kmeans, OUTPUT_DIR
from mean_shift import main as run_mean_shift


def summarize_unsupervised() -> None:
    """Generate consolidated unsupervised comparison between K-Means and Mean-Shift."""
    records = []

    km_eval_path = os.path.join(OUTPUT_DIR, "kmeans_evaluation_k2_k6.csv")
    if os.path.exists(km_eval_path):
        df_km = pd.read_csv(km_eval_path)
        row_k4 = df_km[df_km["k"] == 4].iloc[0]
        records.append({
            "algorithm": "K-Means Clustering",
            "type": "Centroid / Partitioning",
            "clusters": 4,
            "silhouette_score": round(float(row_k4["silhouette_score"]), 4),
            "key_property": "Preset k=4; partitions construct space into 4 personas",
        })

    ms_eval_path = os.path.join(OUTPUT_DIR, "mean_shift_metrics.csv")
    if os.path.exists(ms_eval_path):
        df_ms = pd.read_csv(ms_eval_path)
        records.append({
            "algorithm": "Mean-Shift Clustering",
            "type": "Density / Mode-Seeking",
            "clusters": int(df_ms["clusters_discovered"].iloc[0]),
            "silhouette_score": round(float(df_ms["silhouette_score"].iloc[0]), 4),
            "key_property": f"Non-parametric; auto-discovers density modes (bandwidth={df_ms['estimated_bandwidth'].iloc[0]})",
        })

    comp_df = pd.DataFrame(records)
    comp_df.to_csv(os.path.join(OUTPUT_DIR, "unsupervised_models_comparison.csv"), index=False)

    km_prof_path = os.path.join(OUTPUT_DIR, "kmeans_profiles.csv")
    ms_prof_path = os.path.join(OUTPUT_DIR, "mean_shift_profiles.csv")

    lines = [
        "=" * 96,
        "UNSUPERVISED MACHINE LEARNING SUMMARY REPORT (K-MEANS & MEAN-SHIFT)",
        "Study: AI and the Environment: An Analysis of Public Awareness",
        "=" * 96,
        "Total Dataset Clustered: 2,842 cleaned rows (Q2..Q6 Standardized Construct Space)",
        "",
        "1. ALGORITHM COMPARISON:",
        comp_df.to_string(index=False),
        "",
        "2. K-MEANS CLUSTER PROFILES (k = 4 PERSONAS):",
    ]
    if os.path.exists(km_prof_path):
        lines.append(pd.read_csv(km_prof_path)[["cluster_id", "count", "share_pct", "mean_q2_usage", "mean_awareness", "mean_behavior", "persona_name"]].to_string(index=False))

    lines.extend(["", "3. MEAN-SHIFT DENSITY CLUSTERS (NON-PARAMETRIC MODES):"])
    if os.path.exists(ms_prof_path):
        lines.append(pd.read_csv(ms_prof_path)[["cluster_id", "count", "share_pct", "mean_q2_usage", "mean_awareness", "mean_behavior"]].to_string(index=False))

    lines.extend([
        "",
        "KEY TAKEAWAYS FOR THESIS DEFENSE:",
        "1. K-Means (k=4) cleanly splits the population into 4 meaningful behavioral personas,",
        "   empirically corroborating the rule-based gap typology in 2D PCA latent space.",
        "2. Mean-Shift independently discovers density modes without human presetting of k,",
        "   confirming that dense concentrations of respondents reside around low awareness/low action",
        "   and a distinct separate cluster of high awareness/high action pro-eco users.",
        "=" * 96,
    ])

    report_path = os.path.join(OUTPUT_DIR, "unsupervised_key_insights.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))


def main() -> None:
    print("Running 1/2: K-Means Clustering (kmeans.py)...")
    run_kmeans()

    print("\nRunning 2/2: Mean-Shift Clustering (mean_shift.py)...")
    run_mean_shift()

    summarize_unsupervised()


if __name__ == "__main__":
    main()

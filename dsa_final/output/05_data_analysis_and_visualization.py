"""
05_data_analysis_and_visualization.py
=====================================
RUN ORDER: 5 of 5

Stage 5 — Topic-Aligned Analysis, Cleaning Audit & Visualizations (PySpark + Matplotlib)
----------------------------------------------------------------------------------------
Revolves strictly around our study in `docs/DSA - Proposal.pdf` and
`docs/Data Scalability and Analytics - Data transformation (Pyspark) (1).pdf`:
  "AI and the Environment: An Analysis of Public Awareness"

Analyzes the single unified dataset (`output/reduction/integrated_master_dataset.csv`)
and generates all statistical tables and figures in a clear, step-by-step progression:
  - Figure 1 (`fig1_data_cleaning_summary.png`): Data Integration & Cleaning Summary
  - Figure 2 (`fig2_unified_indices_by_study.png`): Unified Dataset — Mean Indices by Study
  - Figure 3 (`fig3_likert_diverging.png`): Awareness vs. Behavior — Likert Responses (Q3–Q6)
  - Figure 4 (`fig4_index_by_usage.png`): Awareness vs. Behavior by AI-Usage Segment
  - Figure 5 (`fig5_open_ended_themes.png`): Open-Ended Qualitative Themes (Q7 & Q8)
  - Figure 6 (`fig6_correlation_and_gap.png`): Pearson Correlation Matrix & Gap Alignment
"""

import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as F

from pipeline_utils import (
    ALL_TOPIC_THEME_COLS,
    ANALYSIS_DIR,
    ANALYSIS_LOST_ROWS_CSV,
    FIG_DIR,
    MAIN_ITEM_SCORE_COLS,
    MASTER_UNIFIED_CSV,
    SAMPLED_UNIFIED_CSV,
    STAGE1_LOST_REASONS_CSV,
    STAGE1_LOST_SUMMARY_CSV,
    banner,
    create_spark_session,
    log,
    read_csv_spark,
    spark_to_pandas,
)

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
})

DESC_VARS: List[str] = [
    "aware_energy_n",
    "aware_carbon_n",
    "consider_impact_n",
    "avoid_use_n",
    "usage_freq_n",
    "Awareness_Index",
    "Behavior_Index",
    "Awareness_Behavior_Gap",
]

LIKERT_ITEMS: Dict[str, str] = {
    "aware_energy": "Aware AI uses lots of energy (Q3)",
    "aware_carbon": "Aware data centers add emissions (Q4)",
    "consider_impact": "Consider env. impact when using AI (Q5)",
    "avoid_unnecessary_use": "Avoid unnecessary AI use (Q6)",
}

LIKERT_ORDER: List[str] = [
    "Strongly Disagree",
    "Disagree",
    "Neutral",
    "Agree",
    "Strongly Agree",
]

STUDY_DISPLAY_NAMES: Dict[str, str] = {
    "Main_Study_PH_Env_Awareness": "Main Study (PH)",
    "Public_S2_BD_Cognitive_Dependence": "Public S2 (BD)",
    "Public_S3_UTAUT_Responsible_GenAI": "Public S3 (UTAUT)",
    "Public_S4A_UK_Liverpool_Students": "Public S4A (UK Students)",
    "Public_S4B_UK_Liverpool_Staff": "Public S4B (UK Staff)",
}

THEME_LABELS: Dict[str, str] = {
    "c_water": "Water use / scarcity",
    "c_energy": "Energy / electricity use",
    "c_carbon": "Carbon / climate / pollution",
    "c_ewaste": "E-waste / resources",
    "c_noise_heat": "Noise / heat",
    "c_ecosystem": "Ecosystem / land impact",
    "s_education": "Education & training",
    "s_campaign": "Awareness campaigns / media",
    "s_policy": "Policy / regulation",
    "s_transparency": "Transparency / reporting",
    "s_tech": "Efficient / renewable tech",
    "s_responsible_use": "Responsible / mindful use",
}


# ==============================================================================
# 1. DATA PREPARATION & CLEANING AUDIT
# ==============================================================================
def prepare_unified_df(df: DataFrame) -> DataFrame:
    """Cast all numeric and integer columns in the unified dataset and cache in memory."""
    double_cols = DESC_VARS + [f"{c}_z" for c in MAIN_ITEM_SCORE_COLS] + ["pca_1", "pca_2", "pca_3"]
    for col_name in double_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast("double"))

    int_cols = [
        "has_concern_text",
        "has_awareness_text",
        "concern_len_words",
        "awareness_len_words",
        "concern_theme_count",
        "solution_theme_count",
    ] + ALL_TOPIC_THEME_COLS
    for col_name in int_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast("int"))

    return df.cache()


def analyze_and_display_lost_rows(spark: SparkSession) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the per-study cleaning audit tables, append a Grand Total row, and export to CSV."""
    summary_df = spark_to_pandas(read_csv_spark(spark, STAGE1_LOST_SUMMARY_CSV))
    reasons_df = spark_to_pandas(read_csv_spark(spark, STAGE1_LOST_REASONS_CSV))

    for col_name in ["raw_rows", "cleaned_rows", "rows_lost"]:
        summary_df[col_name] = summary_df[col_name].astype(int)
    for col_name in ["cleaned_pct", "lost_pct"]:
        summary_df[col_name] = summary_df[col_name].astype(float)
    reasons_df["rows_lost_for_reason"] = reasons_df["rows_lost_for_reason"].astype(int)

    total_raw = int(summary_df["raw_rows"].sum())
    total_clean = int(summary_df["cleaned_rows"].sum())
    total_lost = int(summary_df["rows_lost"].sum())

    reason_rollup = (
        reasons_df.groupby("invalid_reason")["rows_lost_for_reason"]
        .sum()
        .sort_values(ascending=False)
    )
    rollup_str = "; ".join([f"{reason} (n={int(cnt)})" for reason, cnt in reason_rollup.items()])

    total_row = pd.DataFrame([{
        "source_study": "TOTAL_UNIFIED_DATASET",
        "study_role": "All Combined (Main + Public)",
        "raw_rows": total_raw,
        "cleaned_rows": total_clean,
        "rows_lost": total_lost,
        "cleaned_pct": round(total_clean * 100.0 / max(total_raw, 1), 2),
        "lost_pct": round(total_lost * 100.0 / max(total_raw, 1), 2),
        "why_rows_lost": rollup_str,
    }])

    full_audit = pd.concat([summary_df, total_row], ignore_index=True)
    full_audit.to_csv(ANALYSIS_LOST_ROWS_CSV, index=False, encoding="utf-8-sig")

    banner("ROWS LOST / REMOVED PER STUDY DURING INTEGRATION & CLEANING (AND WHY)")
    print(full_audit[["source_study", "raw_rows", "cleaned_rows", "rows_lost", "lost_pct", "why_rows_lost"]].to_string(index=False))
    return summary_df, reasons_df


# ==============================================================================
# 2. STATISTICAL ANALYSES (ANALYSES 1 TO 8 FROM `docs/`)
# ==============================================================================
def compute_source_summary(df: DataFrame) -> pd.DataFrame:
    """Compute sample size and mean scores across Q2..Q6 and composite indices by study."""
    df_src = (
        df.groupBy("source_study", "study_role")
        .agg(
            F.count("*").alias("n_cleaned"),
            F.round(F.mean("usage_freq_n"), 3).alias("mean_q2_usage"),
            F.round(F.mean("aware_energy_n"), 3).alias("mean_q3_energy_aware"),
            F.round(F.mean("aware_carbon_n"), 3).alias("mean_q4_carbon_aware"),
            F.round(F.mean("consider_impact_n"), 3).alias("mean_q5_consider_impact"),
            F.round(F.mean("avoid_use_n"), 3).alias("mean_q6_avoid_unnecessary"),
            F.round(F.mean("Awareness_Index"), 3).alias("mean_Awareness_Index"),
            F.round(F.mean("Behavior_Index"), 3).alias("mean_Behavior_Index"),
            F.round(F.mean("Awareness_Behavior_Gap"), 3).alias("mean_Gap"),
        )
        .orderBy("source_study")
    )
    out = spark_to_pandas(df_src)
    out.to_csv(os.path.join(ANALYSIS_DIR, "integration_summary_by_source.csv"), index=False, encoding="utf-8-sig")
    return out


def compute_analysis1_descriptives(df: DataFrame) -> pd.DataFrame:
    """Analysis 1: Descriptive statistics (mean, sd, median, min, max) across Q2..Q6 and indices."""
    rows: List[Dict[str, object]] = []
    for col_name in DESC_VARS:
        stats = df.select(
            F.count(F.col(col_name)).alias("n"),
            F.round(F.mean(col_name), 2).alias("mean"),
            F.round(F.stddev(col_name), 2).alias("sd"),
            F.round(F.percentile_approx(col_name, 0.5), 2).alias("median"),
            F.round(F.min(col_name), 2).alias("min"),
            F.round(F.max(col_name), 2).alias("max"),
        ).collect()[0]
        rows.append({"variable": col_name, **stats.asDict()})

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ANALYSIS_DIR, "analysis1_descriptive_stats.csv"), index=False, encoding="utf-8-sig")
    return out


def compute_analysis2_likert_distribution(df: DataFrame) -> pd.DataFrame:
    """Analysis 2: Likert response counts and agreement percentages across Q3..Q6."""
    n_tot = df.count()
    rows: List[Dict[str, object]] = []
    for col_name, label in LIKERT_ITEMS.items():
        counts = {r[col_name]: int(r["count"]) for r in df.groupBy(col_name).count().collect()}
        row: Dict[str, object] = {"item": label}
        for k in LIKERT_ORDER:
            row[k] = counts.get(k, 0)
        row["pct_agree"] = round(100.0 * (int(row["Agree"]) + int(row["Strongly Agree"])) / max(n_tot, 1), 1)
        row["pct_disagree"] = round(100.0 * (int(row["Disagree"]) + int(row["Strongly Disagree"])) / max(n_tot, 1), 1)
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ANALYSIS_DIR, "analysis2_likert_distribution.csv"), index=False, encoding="utf-8-sig")
    return out


def compute_analysis3_gap_types(df: DataFrame) -> pd.DataFrame:
    """Analysis 3: Awareness-to-Behavior gap typology counts and percentages."""
    n_tot = df.count()
    gap_df = (
        df.groupBy("gap_type")
        .count()
        .withColumn("pct", F.round(100.0 * F.col("count") / max(n_tot, 1), 1))
        .orderBy(F.desc("count"))
    )
    out = spark_to_pandas(gap_df)
    out.to_csv(os.path.join(ANALYSIS_DIR, "analysis3_gap_types.csv"), index=False, encoding="utf-8-sig")
    return out


def compute_analysis4_by_usage_segment(df: DataFrame) -> pd.DataFrame:
    """Analysis 4: Mean Awareness, Behavior, and Gap by AI usage frequency segment."""
    by_seg = spark_to_pandas(
        df.groupBy("usage_segment").agg(
            F.count("*").alias("n"),
            F.round(F.mean("Awareness_Index"), 2).alias("mean_awareness"),
            F.round(F.mean("Behavior_Index"), 2).alias("mean_behavior"),
            F.round(F.mean("Awareness_Behavior_Gap"), 2).alias("mean_gap"),
            F.round(F.stddev("Behavior_Index"), 2).alias("sd_behavior"),
        )
    )
    seg_order = ["Light (rarely)", "Moderate (weekly)", "Heavy (daily+)"]
    by_seg["usage_segment"] = pd.Categorical(by_seg["usage_segment"], seg_order, ordered=True)
    by_seg = by_seg.sort_values("usage_segment").reset_index(drop=True)
    by_seg.to_csv(os.path.join(ANALYSIS_DIR, "analysis4_by_usage_segment.csv"), index=False, encoding="utf-8-sig")
    return by_seg


def compute_analysis5_correlations(df: DataFrame) -> Tuple[pd.DataFrame, float]:
    """Analysis 5: Pearson correlations across Q2..Q6, Awareness_Index, and Behavior_Index."""
    pairs = [
        ("aware_energy_n", "aware_carbon_n"),
        ("Awareness_Index", "Behavior_Index"),
        ("consider_impact_n", "avoid_use_n"),
        ("usage_freq_n", "Awareness_Index"),
        ("usage_freq_n", "Behavior_Index"),
        ("aware_energy_n", "consider_impact_n"),
        ("aware_carbon_n", "avoid_use_n"),
    ]
    rows = [
        {"var_1": a, "var_2": b, "pearson_r": round(float(df.stat.corr(a, b) or 0.0), 4)}
        for a, b in pairs
    ]
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ANALYSIS_DIR, "analysis5_correlations.csv"), index=False, encoding="utf-8-sig")

    overall_r = round(float(df.stat.corr("Awareness_Index", "Behavior_Index") or 0.0), 4)
    return out, overall_r


def compute_analysis6_open_ended_themes(df: DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """Analysis 6: Frequency and percentage of Q7 environmental concerns (`c_*`) and Q8 solutions (`s_*`)."""
    present = [c for c in THEME_LABELS if c in df.columns]
    n_c = df.filter(F.col("has_concern_text") == 1).count()
    n_s = df.filter(F.col("has_awareness_text") == 1).count()

    sums_c = df.filter(F.col("has_concern_text") == 1).agg(
        *[F.sum(c).alias(c) for c in present if c.startswith("c_")]
    ).collect()[0].asDict()
    sums_s = df.filter(F.col("has_awareness_text") == 1).agg(
        *[F.sum(c).alias(c) for c in present if c.startswith("s_")]
    ).collect()[0].asDict()

    rows = []
    for col_name in present:
        is_q7 = col_name.startswith("c_")
        mentions = int((sums_c if is_q7 else sums_s).get(col_name) or 0)
        denom = max(n_c if is_q7 else n_s, 1)
        rows.append({
            "question": "Q7 concern" if is_q7 else "Q8 solution",
            "theme_col": col_name,
            "theme": THEME_LABELS[col_name],
            "mentions": mentions,
            "pct_of_answers": round(100.0 * mentions / denom, 1),
        })

    out = pd.DataFrame(rows).sort_values(["question", "mentions"], ascending=[True, False])
    out.to_csv(os.path.join(ANALYSIS_DIR, "analysis6_open_ended_themes.csv"), index=False, encoding="utf-8-sig")
    return out, n_c, n_s


def compute_analysis7_and_8(df: DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Analysis 7 (Age-group profile) & Analysis 8 (Behavior by water-concern mention `c_water`)."""
    n_final = df.count()
    by_age = spark_to_pandas(
        df.groupBy("age_group")
        .agg(
            F.count("*").alias("n"),
            F.round(F.mean("Awareness_Index"), 2).alias("mean_awareness"),
            F.round(F.mean("Behavior_Index"), 2).alias("mean_behavior"),
            F.round(F.mean("usage_freq_n"), 2).alias("mean_usage_freq"),
        )
        .orderBy("age_group")
    )
    by_age["pct"] = (100.0 * by_age["n"] / max(n_final, 1)).round(1)
    by_age.to_csv(os.path.join(ANALYSIS_DIR, "analysis7_by_age_group.csv"), index=False, encoding="utf-8-sig")

    by_water = spark_to_pandas(
        df.filter(F.col("has_concern_text") == 1)
        .groupBy("c_water")
        .agg(
            F.count("*").alias("n"),
            F.round(F.mean("Awareness_Index"), 2).alias("mean_awareness"),
            F.round(F.mean("Behavior_Index"), 2).alias("mean_behavior"),
        )
        .orderBy("c_water")
    )
    by_water.to_csv(os.path.join(ANALYSIS_DIR, "analysis8_water_concern_vs_behavior.csv"), index=False, encoding="utf-8-sig")
    return by_age, by_water


# ==============================================================================
# 3. VISUALIZATIONS (ORDERED LOGICAL PROGRESSION: FIGURE 1 TO FIGURE 6)
# ==============================================================================
def clean_existing_figures() -> None:
    """Remove any old PNG files in `output/analysis/figures/` before saving the new numbered figures."""
    if os.path.exists(FIG_DIR):
        for fname in os.listdir(FIG_DIR):
            if fname.lower().endswith(".png"):
                os.remove(os.path.join(FIG_DIR, fname))


def _simplify_removal_reason(reason_str: str) -> str:
    """Group detailed row-removal messages into 5 concise, readable categories for Figure 1B."""
    r_low = str(reason_str).lower()
    if "duplicate" in r_low:
        return "Duplicate Records"
    if "not related to main topic" in r_low:
        return "Unrelated to Main Topic"
    if "straight-line" in r_low:
        return "Straight-Line Responses"
    if "attention check" in r_low:
        return "Failed Attention Check"
    return "Incomplete / Blank Q7-Q8"


def plot_fig1_data_cleaning_summary(summary_df: pd.DataFrame, reasons_df: pd.DataFrame) -> None:
    """
    Figure 1: Data Integration & Cleaning Summary
      - Panel A: Cleaned vs. Removed rows by study (% and exact counts)
      - Panel B: Simplified 5-category breakdown of why rows were removed
    """
    fig, axes = plt.subplots(
        1, 2,
        figsize=(12.6, 4.8),
        gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.48},
    )

    tot_raw = int(summary_df["raw_rows"].sum())
    tot_clean = int(summary_df["cleaned_rows"].sum())
    tot_lost = int(summary_df["rows_lost"].sum())

    labels, clean_pcts, loss_pcts = [], [], []
    for src_key, disp_label in reversed(list(STUDY_DISPLAY_NAMES.items())):
        match = summary_df[summary_df["source_study"] == src_key]
        if not match.empty:
            row = match.iloc[0]
            c_cnt = int(row["cleaned_rows"])
            r_cnt = int(row["raw_rows"])
            labels.append(f"{disp_label}\n({c_cnt:,} / {r_cnt:,} kept)")
            clean_pcts.append(float(row["cleaned_pct"]))
            loss_pcts.append(float(row["lost_pct"]))

    y = np.arange(len(labels))
    bar_h = 0.58
    axes[0].barh(
        y, clean_pcts, height=bar_h, color="#16a34a", alpha=0.9,
        edgecolor="white", linewidth=0.8, label=f"Cleaned & Kept ({tot_clean:,})",
    )
    axes[0].barh(
        y, loss_pcts, left=clean_pcts, height=bar_h, color="#dc2626", alpha=0.88,
        edgecolor="white", linewidth=0.8, label=f"Removed ({tot_lost:,})",
    )

    for i in range(len(labels)):
        if clean_pcts[i] >= 13.0:
            axes[0].text(
                clean_pcts[i] / 2.0, i, f"{clean_pcts[i]:.1f}%",
                va="center", ha="center", color="white", fontweight="bold", fontsize=8.5,
            )
        if loss_pcts[i] >= 13.0:
            axes[0].text(
                clean_pcts[i] + loss_pcts[i] / 2.0, i, f"{loss_pcts[i]:.1f}%",
                va="center", ha="center", color="white", fontweight="bold", fontsize=8.5,
            )

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8.8, fontweight="bold")
    axes[0].set_xlim(0, 100)
    axes[0].set_xticks([0, 20, 40, 60, 80, 100])
    axes[0].set_xlabel("Percentage of Raw Study Rows (%)")
    axes[0].set_title("A. Cleaned vs. Removed Rows by Study", fontweight="bold", pad=28)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=8.5)

    reasons_plot = reasons_df.copy()
    reasons_plot["simple_category"] = reasons_plot["invalid_reason"].map(_simplify_removal_reason)
    reasons_plot["is_main"] = reasons_plot["source_study"].str.contains("Main_Study")

    cat_summary = (
        reasons_plot.groupby("simple_category", as_index=False)
        .agg(
            total_removed=("rows_lost_for_reason", "sum"),
            main_removed=("rows_lost_for_reason", lambda s: int(s[reasons_plot.loc[s.index, "is_main"]].sum())),
        )
        .sort_values("total_removed", ascending=True)
        .reset_index(drop=True)
    )
    cat_summary["public_removed"] = cat_summary["total_removed"] - cat_summary["main_removed"]

    tot_pub_rem = int(cat_summary["public_removed"].sum())
    tot_main_rem = int(cat_summary["main_removed"].sum())
    tot_removed = int(cat_summary["total_removed"].sum())

    y_c = np.arange(len(cat_summary))
    axes[1].barh(
        y_c, cat_summary["public_removed"], height=bar_h,
        color="#ef4444", alpha=0.88, edgecolor="white", linewidth=0.8, label=f"Public Studies ({tot_pub_rem:,})",
    )
    axes[1].barh(
        y_c, cat_summary["main_removed"], left=cat_summary["public_removed"], height=bar_h,
        color="#f59e0b", alpha=0.95, edgecolor="white", linewidth=0.8, label=f"Main Study PH ({tot_main_rem:,})",
    )

    max_cat = int(cat_summary["total_removed"].max())
    for idx_i, row_c in cat_summary.iterrows():
        cnt_v = int(row_c["total_removed"])
        pct_v = (cnt_v * 100.0 / tot_removed) if tot_removed else 0.0
        main_v = int(row_c["main_removed"])
        note = f"  [{main_v} Main]" if main_v > 0 else ""
        axes[1].text(
            cnt_v + max_cat * 0.02, idx_i, f"{cnt_v:,} ({pct_v:.1f}%){note}",
            va="center", ha="left", color="#1e293b", fontweight="bold", fontsize=8.3,
        )

    axes[1].set_yticks(y_c)
    axes[1].set_yticklabels(cat_summary["simple_category"], fontsize=8.8, fontweight="bold")
    axes[1].set_xlim(0, max_cat * 1.38)
    axes[1].set_xlabel(f"Number of Removed Rows (Total = {tot_removed:,})")
    axes[1].set_title("B. Why Rows Were Removed (by Reason)", fontweight="bold", pad=28)
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=8.5)

    fig.suptitle(
        f"Figure 1: Data Integration & Cleaning Summary ({tot_clean:,} Cleaned / {tot_lost:,} Removed of {tot_raw:,} Raw Rows)",
        fontweight="bold",
        fontsize=11.5,
        y=1.08,
    )
    fig.savefig(os.path.join(FIG_DIR, "fig1_data_cleaning_summary.png"), bbox_inches="tight")
    plt.close(fig)


def plot_fig2_unified_indices_by_study(src_summary: pd.DataFrame) -> None:
    """Figure 2: Unified Dataset — Mean Construct Indices by Study."""
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    x = np.arange(len(src_summary))
    w = 0.25

    b1 = ax.bar(x - w, src_summary["mean_q2_usage"], w, label="Q2: AI Usage Freq", color="#64748b", edgecolor="white")
    b2 = ax.bar(x, src_summary["mean_Awareness_Index"], w, label="Part II: Awareness Index (Q3-Q4)", color="#2166ac", edgecolor="white")
    b3 = ax.bar(x + w, src_summary["mean_Behavior_Index"], w, label="Part III: Behavior Index (Q5-Q6)", color="#ef8a62", edgecolor="white")

    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=7.8)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=7.8)
    ax.bar_label(b3, fmt="%.2f", padding=2, fontsize=7.8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{STUDY_DISPLAY_NAMES.get(r['source_study'], r['source_study'])}\n(n={int(r['n_cleaned']):,})" for _, r in src_summary.iterrows()],
        fontsize=8.5,
        fontweight="bold",
    )
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Mean Score (1-5 Scale)")
    tot_n = int(src_summary["n_cleaned"].sum())
    ax.set_title(f"Figure 2: Unified Dataset — Mean Construct Indices by Study (N = {tot_n:,})", fontweight="bold", pad=24)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=8.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_unified_indices_by_study.png"), bbox_inches="tight")
    plt.close(fig)


def plot_fig3_likert_diverging(likert_df: pd.DataFrame) -> None:
    """Figure 3: 100% Stacked Likert Responses (Q3..Q6) across the Unified Dataset."""
    fig, ax = plt.subplots(figsize=(9.2, 4.1))
    colors = ["#b2182b", "#ef8a62", "#d9d9d9", "#67a9cf", "#2166ac"]

    tot = likert_df[LIKERT_ORDER].sum(axis=1)
    n_resp = int(tot.iloc[0])
    share = likert_df[LIKERT_ORDER].div(tot, axis=0) * 100.0
    labels = likert_df["item"].tolist()[::-1]
    share = share.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(labels))
    start = np.zeros(len(labels))

    for cat, colr in zip(LIKERT_ORDER, colors):
        widths = share[cat].values
        ax.barh(y_pos, widths, left=start, height=0.60, color=colr, label=cat, edgecolor="white", linewidth=0.8)
        txt_col = "white" if cat in ("Strongly Disagree", "Strongly Agree") else "#1e293b"
        for idx_i, (st_v, w_v) in enumerate(zip(start, widths)):
            if w_v >= 3.5:
                f_sz = 7.0 if w_v < 5.0 else 8.0
                ax.text(
                    st_v + w_v / 2.0, idx_i, f"{w_v:.1f}%",
                    va="center", ha="center", color=txt_col, fontweight="bold", fontsize=f_sz,
                )
        start = start + widths

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.2, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xticklabels([f"{x}%" for x in range(0, 101, 20)])
    ax.set_xlabel("Percentage of Respondents (%)")
    ax.set_title(f"Figure 3: Awareness vs. Behavior — Likert Responses (N = {n_resp:,})", fontweight="bold", pad=12)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_likert_diverging.png"), bbox_inches="tight")
    plt.close(fig)


def plot_fig4_index_by_usage(by_seg: pd.DataFrame) -> None:
    """Figure 4: Awareness vs. Behavior Index by AI-Usage Segment across the Unified Dataset."""
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    tot_n = int(by_seg["n"].sum())

    x = np.arange(len(by_seg))
    w = 0.36
    b1 = ax.bar(x - w / 2, by_seg["mean_awareness"], w, label="Awareness Index (Q3-Q4)", color="#2166ac", edgecolor="white")
    b2 = ax.bar(x + w / 2, by_seg["mean_behavior"], w, label="Behavior Index (Q5-Q6)", color="#ef8a62", edgecolor="white")

    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=8.5, fontweight="bold")
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={int(n):,})" for s, n in zip(by_seg["usage_segment"], by_seg["n"])], fontweight="bold")
    ax.set_ylim(0, 5.4)
    ax.set_ylabel("Mean Score (1-5)")
    ax.set_title(f"Figure 4: Awareness vs. Behavior by AI-Usage Segment (N = {tot_n:,})", fontweight="bold", pad=12)
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_index_by_usage.png"), bbox_inches="tight")
    plt.close(fig)


def plot_fig5_open_ended_themes(theme_pdf: pd.DataFrame, n_c: int, n_s: int, total_n: int) -> None:
    """Figure 5: Open-Ended Qualitative Themes — Environmental Concerns (Q7) & Solutions (Q8)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    for ax, q, colr, ttl in zip(
        axes,
        ["Q7 concern", "Q8 solution"],
        ["#b2182b", "#1a9850"],
        [
            f"A. Environmental Concerns (Q7: n={n_c:,} text answers / N={total_n:,})",
            f"B. Suggested Solutions (Q8: n={n_s:,} text answers / N={total_n:,})",
        ],
    ):
        sub = theme_pdf[theme_pdf["question"] == q].sort_values("pct_of_answers")
        bars = ax.barh(sub["theme"], sub["pct_of_answers"], color=colr, alpha=0.9, edgecolor="white")
        for bar, m_cnt in zip(bars, sub["mentions"]):
            w_val = bar.get_width()
            ax.text(
                w_val + 1.2,
                bar.get_y() + bar.get_height() / 2,
                f"{w_val:.1f}% (n={int(m_cnt):,})",
                va="center",
                fontweight="bold",
                fontsize=8.2,
            )
        ax.set_xlim(0, max(60, sub["pct_of_answers"].max() + 18))
        ax.set_title(ttl, fontweight="bold", fontsize=9.5)
        ax.set_xlabel("% of Valid Open-Ended Answers")

    fig.suptitle(
        f"Figure 5: Open-Ended Qualitative Themes Across Unified Dataset (N = {total_n:,})",
        fontweight="bold",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_open_ended_themes.png"), bbox_inches="tight")
    plt.close(fig)


def plot_fig6_correlation_and_gap(df: DataFrame, gap_pdf: pd.DataFrame) -> None:
    """Figure 6: Pearson Correlation Matrix (Q2–Q6) & Awareness-Behavior Alignment Pie Chart."""
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    hm_vars = ["aware_energy_n", "aware_carbon_n", "consider_impact_n", "avoid_use_n", "usage_freq_n"]
    hm_lbl = ["Q3 energy\naware", "Q4 carbon\naware", "Q5 consider\nimpact", "Q6 avoid\nunnecessary", "Q2 usage\nfreq"]
    cm = np.array([[1.0 if a == b else float(df.stat.corr(a, b) or 0.0) for b in hm_vars] for a in hm_vars])

    im = axes[0].imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_xticks(range(5))
    axes[0].set_xticklabels(hm_lbl, fontsize=8)
    axes[0].set_yticks(range(5))
    axes[0].set_yticklabels(hm_lbl, fontsize=8)
    for i in range(5):
        for j in range(5):
            axes[0].text(
                j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(cm[i, j]) > 0.5 else "black", fontweight="bold",
            )

    gp = gap_pdf.set_index("gap_type")["count"]
    tot_n = int(gp.sum())
    axes[0].set_title(f"A. Pearson Correlation Matrix (Q2-Q6, N = {tot_n:,})", fontweight="bold")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    pie_cols = {
        "Unaware & not acting": "#94a3b8",
        "Aware & taking action": "#2166ac",
        "Aware but not acting": "#ef8a62",
        "Acting beyond awareness": "#16a34a",
    }
    wedges, texts, autotexts = axes[1].pie(
        gp,
        labels=[f"{k}\n(n={int(v):,})" for k, v in gp.items()],
        autopct="%1.1f%%",
        pctdistance=0.72,
        startangle=140,
        colors=[pie_cols.get(k, "#94a3b8") for k in gp.index],
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        textprops={"fontsize": 8.5, "fontweight": "bold"},
    )
    for idx_w, k in enumerate(gp.index):
        if k in ("Aware & taking action", "Acting beyond awareness"):
            autotexts[idx_w].set_color("white")
    axes[1].set_title(f"B. Awareness-Behavior Alignment (N = {tot_n:,})", fontweight="bold")

    fig.suptitle(
        f"Figure 6: Construct Correlations & Awareness-Behavior Alignment (N = {tot_n:,})",
        fontweight="bold",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig6_correlation_and_gap.png"), bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# 4. EXECUTIVE SUMMARY REPORT (`key_insights.txt`)
# ==============================================================================
def write_key_insights_report(
    total_resp: int,
    sampled_cnt: int,
    n_c: int,
    n_s: int,
    overall_r: float,
    summary_df: pd.DataFrame,
    reasons_df: pd.DataFrame,
    src_summary: pd.DataFrame,
    a1_df: pd.DataFrame,
    likert_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    by_seg: pd.DataFrame,
    corr_df: pd.DataFrame,
    theme_df: pd.DataFrame,
) -> None:
    """Export the comprehensive topic-aligned unified-dataset analysis and lost-rows report."""
    tot_raw = int(summary_df["raw_rows"].sum())
    tot_lost = int(summary_df["rows_lost"].sum())
    main_row = summary_df[summary_df["source_study"] == "Main_Study_PH_Env_Awareness"].iloc[0]
    m_raw = int(main_row["raw_rows"])
    m_clean = int(main_row["cleaned_rows"])
    m_lost = int(main_row["rows_lost"])
    m_lost_pct = float(main_row["lost_pct"])

    lines = [
        "=" * 96,
        "TOPIC-ALIGNED UNIFIED DATASET ANALYTICS REPORT (BASIS: `main_dataset.csv` & `docs/`)",
        "Study: AI and the Environment: An Analysis of Public Awareness (Azul, de la Cruz, Galvez - BSIT 3B)",
        "=" * 96,
        f"Unified Dataset Raw Rows Integrated      : {tot_raw:,} rows",
        f"Unified Dataset Cleaned Topic-Valid Rows : {total_resp:,} rows ({total_resp * 100.0 / max(tot_raw, 1):.2f}% retained)",
        f"Unified Dataset Total Rows Lost/Removed  : {tot_lost:,} rows ({tot_lost * 100.0 / max(tot_raw, 1):.2f}% removed)",
        f"Main Study (`main_dataset.csv`) Retention: {m_clean:,} cleaned / {m_raw:,} raw rows ({m_lost:,} rows removed = {m_lost_pct:.2f}%)",
        f"50% Stratified Sample Rows               : {sampled_cnt:,} rows",
        f"Topic-Valid Open-Ended Answers           : Q7 Concerns = {n_c:,} rows | Q8 Solutions = {n_s:,} rows",
        f"Correlation (Awareness_Index vs Behavior_Index): r = {overall_r:+.4f} (Unified Dataset, N = {total_resp:,})",
        "",
        "1. HOW MANY ROWS WERE LOST / REMOVED AS UNRELATED IN EACH STUDY AND WHY (SUMMARY):",
        summary_df.to_string(index=False),
        "",
        "2. EXACT BREAKDOWN OF WHY ROWS WERE LOST OR REMOVED PER STUDY:",
        reasons_df.to_string(index=False),
        "",
        "3. UNIFIED DATASET ITEM & INDEX MEANS BY STUDY (Q2 TO Q6, AWARENESS, BEHAVIOR, GAP):",
        src_summary.to_string(index=False),
        "",
        "4. ANALYSIS 1 — DESCRIPTIVE STATISTICS (UNIFIED DATASET):",
        a1_df.to_string(index=False),
        "",
        "5. ANALYSIS 2 — LIKERT RESPONSE DISTRIBUTION (UNIFIED DATASET):",
        likert_df.to_string(index=False),
        "",
        "6. ANALYSIS 3 — AWARENESS-TO-BEHAVIOR GAP TYPOLOGY:",
        gap_df.to_string(index=False),
        "",
        "7. ANALYSIS 4 — INDEX BREAKDOWN BY AI USAGE FREQUENCY SEGMENT:",
        by_seg.to_string(index=False),
        "",
        "8. ANALYSIS 5 — PEARSON CORRELATIONS (UNIFIED DATASET):",
        corr_df.to_string(index=False),
        "",
        "9. ANALYSIS 6 — OPEN-ENDED ENVIRONMENTAL CONCERNS (Q7) & SOLUTIONS (Q8):",
        theme_df.to_string(index=False),
        "",
    ]
    out_path = os.path.join(ANALYSIS_DIR, "key_insights.txt")
    with open(out_path, "w", encoding="utf-8") as f_txt:
        f_txt.write("\n".join(lines))


# ==============================================================================
# 5. MAIN ENTRY POINT
# ==============================================================================
def main() -> None:
    if not os.path.exists(MASTER_UNIFIED_CSV):
        raise FileNotFoundError("Stage 4 output not found. Please run `04_data_reduction.py` first.")

    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    spark = create_spark_session("DSA-Final-05-Topic-Aligned-Analysis")
    banner("STAGE 5: TOPIC-ALIGNED ANALYSIS, CLEANING AUDIT & VISUALIZATIONS")

    df = prepare_unified_df(read_csv_spark(spark, MASTER_UNIFIED_CSV))
    sampled_cnt = read_csv_spark(spark, SAMPLED_UNIFIED_CSV).count() if os.path.exists(SAMPLED_UNIFIED_CSV) else 0
    total_resp = df.count()

    # Compute all statistical tables dynamically
    summary_df, reasons_df = analyze_and_display_lost_rows(spark)
    src_summary = compute_source_summary(df)
    a1_df = compute_analysis1_descriptives(df)
    likert_df = compute_analysis2_likert_distribution(df)
    gap_df = compute_analysis3_gap_types(df)
    by_seg = compute_analysis4_by_usage_segment(df)
    corr_df, overall_r = compute_analysis5_correlations(df)
    theme_df, n_c, n_s = compute_analysis6_open_ended_themes(df)
    compute_analysis7_and_8(df)
    log("Saved 10 topic-aligned analytical & cleaning-audit CSV tables to output/analysis/")

    # Generate figures in logical narrative order (Figure 1 through Figure 6)
    clean_existing_figures()
    plot_fig1_data_cleaning_summary(summary_df, reasons_df)
    plot_fig2_unified_indices_by_study(src_summary)
    plot_fig3_likert_diverging(likert_df)
    plot_fig4_index_by_usage(by_seg)
    plot_fig5_open_ended_themes(theme_df, n_c, n_s, total_resp)
    plot_fig6_correlation_and_gap(df, gap_df)
    log("Saved 6 ordered figures (fig1..fig6) to output/analysis/figures/")

    write_key_insights_report(
        total_resp, sampled_cnt, n_c, n_s, overall_r,
        summary_df, reasons_df, src_summary, a1_df, likert_df, gap_df, by_seg, corr_df, theme_df,
    )
    log("Saved topic-aligned executive summary to output/analysis/key_insights.txt")

    df.unpersist()
    spark.stop()
    log("Stage 5 (Topic-Aligned Analysis & Visualization) finished successfully.")


if __name__ == "__main__":
    main()

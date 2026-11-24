"""
statistics.py
=============
Descriptive statistics, Cronbach's alpha, Likert distributions, segment summaries,
gap typology, and Pearson/Spearman correlation analyses for N = 72 valid records.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from scipy import stats

_CODE_DIR = str(Path(__file__).resolve().parents[1])
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from data_cleaning.config import OUTPUT_DIR


def compute_cronbach_alpha(pdf_clean: pd.DataFrame) -> float:
    """Compute Cronbach's alpha internal reliability across Q3-Q6."""
    items = pdf_clean[["aware_energy_n", "aware_carbon_n", "consider_impact_n", "avoid_use_n"]].values.astype(float)
    return float((4.0 / 3.0) * (1.0 - items.var(axis=0, ddof=1).sum() / items.sum(axis=1).var(ddof=1)))


def compute_cleaning_audit(pdf_all: pd.DataFrame) -> pd.DataFrame:
    """Summarize record validity counts and percentages across all 84 raw submissions."""
    clean_audit = (
        pdf_all.groupby("record_status", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )
    clean_audit["pct_of_raw84"] = (clean_audit["count"] * 100.0 / len(pdf_all)).round(2)
    clean_audit.to_csv(os.path.join(OUTPUT_DIR, "cleaning_audit_all84.csv"), index=False, encoding="utf-8-sig")
    return clean_audit


def compute_descriptive_stats(pdf_clean: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, SD, median, mode, min, and max for Q2-Q6 and composite indices."""
    desc_cols = [
        ("usage_freq_n", "Q2: AI Usage Frequency"),
        ("aware_energy_n", "Q3: Energy Awareness"),
        ("aware_carbon_n", "Q4: Carbon Awareness"),
        ("consider_impact_n", "Q5: Consider Env. Impact"),
        ("avoid_use_n", "Q6: Avoid Unnecessary Use"),
        ("Awareness_Index", "Awareness Index (Q3-Q4 Mean)"),
        ("Behavior_Index", "Behavior Index (Q5-Q6 Mean)"),
        ("Awareness_Behavior_Gap", "Awareness-Behavior Gap"),
    ]
    desc_rows = []
    for col, lbl in desc_cols:
        series = pdf_clean[col].astype(float)
        desc_rows.append({
            "variable": col,
            "label": lbl,
            "n": len(series),
            "mean": round(float(series.mean()), 2),
            "sd": round(float(series.std(ddof=1)), 2),
            "median": round(float(series.median()), 2),
            "mode": round(float(series.mode().iloc[0]), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
        })
    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(os.path.join(OUTPUT_DIR, "descriptive_stats.csv"), index=False, encoding="utf-8-sig")
    return desc_df


def compute_likert_distribution(pdf_clean: pd.DataFrame) -> pd.DataFrame:
    """Compute Likert level counts and Agree/Neutral/Disagree percentages for Q3-Q6."""
    likert_order = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
    likert_items = [
        ("aware_energy", "Q3: Aware AI uses lots of energy"),
        ("aware_carbon", "Q4: Aware data centers add emissions"),
        ("consider_impact", "Q5: Consider impact when using AI"),
        ("avoid_unnecessary_use", "Q6: Avoid unnecessary AI use"),
    ]
    likert_rows = []
    n_clean = len(pdf_clean)
    for col, lbl in likert_items:
        vc = pdf_clean[col].str.title().value_counts().to_dict()
        row = {"item": lbl}
        for lvl in likert_order:
            row[lvl] = int(vc.get(lvl, 0))
        row["pct_agree"] = round((row["Agree"] + row["Strongly Agree"]) * 100.0 / n_clean, 1)
        row["pct_neutral"] = round(row["Neutral"] * 100.0 / n_clean, 1)
        row["pct_disagree"] = round((row["Disagree"] + row["Strongly Disagree"]) * 100.0 / n_clean, 1)
        likert_rows.append(row)
    likert_df = pd.DataFrame(likert_rows)
    likert_df.to_csv(os.path.join(OUTPUT_DIR, "likert_distribution.csv"), index=False, encoding="utf-8-sig")
    return likert_df


def compute_usage_segments(pdf_clean: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Awareness_Index, Behavior_Index, and gap across AI usage segments."""
    n_clean = len(pdf_clean)
    seg_order = ["Light (Rarely/1x Wk)", "Moderate (Weekly)", "Heavy (Daily+)"]
    seg_df = (
        pdf_clean.groupby("usage_segment", as_index=False)
        .agg(
            n=("respondent_id", "count"),
            mean_awareness=("Awareness_Index", "mean"),
            sd_awareness=("Awareness_Index", "std"),
            mean_behavior=("Behavior_Index", "mean"),
            sd_behavior=("Behavior_Index", "std"),
            mean_gap=("Awareness_Behavior_Gap", "mean"),
        )
    )
    seg_df["pct"] = (seg_df["n"] * 100.0 / n_clean).round(1)
    for col in ["mean_awareness", "sd_awareness", "mean_behavior", "sd_behavior", "mean_gap"]:
        seg_df[col] = seg_df[col].round(2)
    seg_df["usage_segment"] = pd.Categorical(seg_df["usage_segment"], seg_order, ordered=True)
    seg_df = seg_df.sort_values("usage_segment").reset_index(drop=True)
    seg_df.to_csv(os.path.join(OUTPUT_DIR, "usage_segment_summary.csv"), index=False, encoding="utf-8-sig")
    return seg_df


def compute_age_groups(pdf_clean: pd.DataFrame) -> pd.DataFrame:
    """Summarize usage, awareness, and behavior scores by age bracket."""
    n_clean = len(pdf_clean)
    age_df = (
        pdf_clean.groupby("age_group", as_index=False)
        .agg(
            n=("respondent_id", "count"),
            mean_usage=("usage_freq_n", "mean"),
            mean_awareness=("Awareness_Index", "mean"),
            mean_behavior=("Behavior_Index", "mean"),
        )
        .sort_values("age_group")
    )
    age_df["pct"] = (age_df["n"] * 100.0 / n_clean).round(1)
    for col in ["mean_usage", "mean_awareness", "mean_behavior"]:
        age_df[col] = age_df[col].round(2)
    age_df.to_csv(os.path.join(OUTPUT_DIR, "age_group_summary.csv"), index=False, encoding="utf-8-sig")
    return age_df


def compute_gap_typology(pdf_clean: pd.DataFrame) -> pd.DataFrame:
    """Compute respondent counts and shares across the 4 Awareness-Behavior profiles."""
    n_clean = len(pdf_clean)
    gap_df = (
        pdf_clean.groupby("gap_type", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )
    gap_df["pct"] = (gap_df["count"] * 100.0 / n_clean).round(1)
    gap_df.to_csv(os.path.join(OUTPUT_DIR, "gap_typology_summary.csv"), index=False, encoding="utf-8-sig")
    return gap_df


def compute_correlations(pdf_clean: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson r and Spearman rho correlations with p-values."""
    pairs = [
        ("Awareness_Index", "Behavior_Index"),
        ("aware_energy_n", "aware_carbon_n"),
        ("consider_impact_n", "avoid_use_n"),
        ("aware_carbon_n", "avoid_use_n"),
        ("aware_energy_n", "consider_impact_n"),
        ("usage_freq_n", "Behavior_Index"),
        ("usage_freq_n", "Awareness_Index"),
    ]
    corr_rows = []
    for var_a, var_b in pairs:
        x = pdf_clean[var_a].astype(float)
        y = pdf_clean[var_b].astype(float)
        r, p = stats.pearsonr(x, y)
        rho, rho_p = stats.spearmanr(x, y)
        corr_rows.append({
            "var_1": var_a,
            "var_2": var_b,
            "pearson_r": round(float(r), 4),
            "pearson_p": float(f"{p:.4e}"),
            "spearman_rho": round(float(rho), 4),
            "spearman_p": float(f"{rho_p:.4e}"),
            "significant_05": "Yes" if p < 0.05 else "No",
        })
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(os.path.join(OUTPUT_DIR, "correlations.csv"), index=False, encoding="utf-8-sig")
    return corr_df


def compute_statistics(pdf_all: pd.DataFrame, pdf_clean: pd.DataFrame) -> Dict[str, Any]:
    """Compute and export all cleaning audit, descriptive, and correlation tables."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clean_audit = compute_cleaning_audit(pdf_all)
    desc_df = compute_descriptive_stats(pdf_clean)
    alpha = compute_cronbach_alpha(pdf_clean)
    likert_df = compute_likert_distribution(pdf_clean)
    seg_df = compute_usage_segments(pdf_clean)
    age_df = compute_age_groups(pdf_clean)
    gap_df = compute_gap_typology(pdf_clean)
    corr_df = compute_correlations(pdf_clean)

    print(f"Cronbach's Alpha (Q3..Q6, N = {len(pdf_clean)}): {alpha:.4f}")
    return {
        "clean_audit": clean_audit,
        "desc": desc_df,
        "likert": likert_df,
        "seg": seg_df,
        "age": age_df,
        "gap": gap_df,
        "corr": corr_df,
        "alpha": alpha,
    }

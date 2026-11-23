"""
01_clean_and_analyze.py
=======================
Simple, single-dataset PySpark pipeline for:
  "AI and the Environment: An Analysis of Public Awareness"

Dataset:
  - `dataset/main_dataset.csv` (84 raw responses -> 72 cleaned valid records;
    12 invalid/unrelated rows explicitly audited and marked).

Steps:
  1. Load raw CSV in Apache PySpark and remove PII (`Timestamp`, `Username`).
  2. Audit all 84 responses, marking invalid/unrelated Q7 & Q8 answers.
  3. Clean and encode Q1..Q6, compute Awareness_Index, Behavior_Index, and Gap.
  4. Categorize all Q7 (Concerns) and Q8 (Solutions) — including Unrelated/Invalid.
  5. Export summary tables (CSV) and 4 publication-grade figures (PNG).
"""

import csv
import os
import re
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import DataFrame, SparkSession, functions as F
from scipy import stats

# Windows-safe PySpark configuration (handles spaces in workspace path)
os.environ["SPARK_HOME"] = os.path.dirname(pyspark.__file__)
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Paths
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CODE_DIR, ".."))
RAW_CSV = os.path.join(PROJECT_DIR, "dataset", "main_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

plt.rcParams.update({
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
    "font.family": "DejaVu Sans",
})

# Regex detecting whether Q7/Q8 addresses AI's environmental impact
ENV_TOPIC_REGEX = (
    r"water|drought|drouth|cool|scarcity|energy|electric|power|watt|grid|"
    r"carbon|emission|greenhouse|global warming|climate|co2|pollut|"
    r"e-waste|e waste|waste|hardware|mining|resource|material|contaminat|"
    r"noise|heat|temperature|loud|flora|fauna|wildlife|ecosystem|biodiversity|"
    r"habitat|forest|\bland\b|farm|pax silica|clark|deforest|aeta|"
    r"environment|nature|earth|planet|data cent|server|sustainab|ecological|green"
)

# 6 Core Q7 Environmental Concern Themes + General Environmental fallback
Q7_THEMES: Dict[str, str] = {
    "c_water": r"water|drought|drouth|cool|scarcity|fresh water",
    "c_energy": r"energy|electric|power|watt|grid",
    "c_carbon": r"carbon|emission|greenhouse|global warming|climate|co2|pollut|weather",
    "c_ecosystem": r"flora|fauna|wildlife|ecosystem|biodiversity|habitat|forest|\bland\b|farm|pax silica|clark|deforest|aeta|destroying nature|environmental destruction",
    "c_ewaste": r"e-waste|e waste|electronic waste|waste|hardware|mining|natural resource|resource depletion|contaminat",
    "c_noise_heat": r"noise|heat|temperature|loud",
}

Q7_LABELS: Dict[str, str] = {
    "c_water": "Water Use & Scarcity",
    "c_energy": "Energy & Electricity Use",
    "c_carbon": "Carbon & Climate Change",
    "c_ecosystem": "Ecosystem, Land & Pax Silica",
    "c_ewaste": "E-Waste & Resource Depletion",
    "c_noise_heat": "Heat & Noise Pollution",
    "c_general_env": "General Environmental Mention",
}

# 6 Core Q8 Suggested Solution Themes
Q8_THEMES: Dict[str, str] = {
    "s_education": r"educat|school|teach|seminar|literacy|workshop|training|learn|inform|curricul|universit|orientat|course|class|presentation|enlighten|understand|subject|knowledge",
    "s_campaign": r"campaign|social media|advocacy|advocate|\bposts?\b|media|poster|tiktok|infographic|video|platform|flyer|news|article|ads|page|promotional|camping awareness|highlight|present the pros|reality of data|talking about|more aware",
    "s_responsible_use": r"responsib|moderat|avoid|mindful|wisely|proper use|unnecessary|minimalism|own mind|own work|own skills|own ideas|last resort|sparingly|not rely|relying|depend|defend on it|creativity|non-digital|human resources|books|libraries|discouraging|intentionally|limits|choice",
    "s_policy": r"law|regulat|polic|government|gobyerno|legislat|block|limit|rules|guideline|dict|ntc|heads of the institution|mandatory|capitalization",
    "s_transparency": r"transparen|disclos|report|accountab|label|compan|developer",
    "s_tech": r"efficien|renewable|clean energy|green|sustainab|solar|cooling|equipments placements|manage wastes",
}

Q8_LABELS: Dict[str, str] = {
    "s_education": "Education & Literacy Programs",
    "s_campaign": "Awareness Campaigns & Media",
    "s_responsible_use": "Responsible & Mindful AI Use",
    "s_policy": "Policy, Governance & Limits",
    "s_transparency": "Corporate Transparency & Labels",
    "s_tech": "Green & Sustainable AI Tech",
}


def create_spark() -> SparkSession:
    """Initialize a lightweight local PySpark session."""
    spark = (
        SparkSession.builder
        .appName("DSA-Final-MainDataset-Cleaning-Analysis")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def write_spark_csv(df: DataFrame, path: str) -> None:
    """Write Spark DataFrame directly to a single UTF-8 CSV without winutils.exe."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = df.columns
    with open(path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for row in df.toLocalIterator():
            writer.writerow(["" if v is None else v for v in row])


def classify_off_topic_subtype(q7_text: str) -> str:
    """Assign a clear human-readable subtype to unrelated (non-environmental) Q7 answers."""
    t = q7_text.lower()
    if "privacy" in t or "store the information" in t:
        return "Unrelated: Data Privacy"
    if "hack" in t:
        return "Unrelated: Cybersecurity / Hacking"
    if "creditibility" in t or "credibility" in t:
        return "Unrelated: Information Credibility"
    if "depending" in t or "depend" in t:
        return "Unrelated: AI Over-Dependence"
    if "reality" in t or "not real" in t:
        return "Unrelated: Synthetic / Fake Reality"
    if "thinking for itself" in t or "like a human" in t:
        return "Unrelated: AI Autonomy / Sentience"
    return "Unrelated: Vague Non-Env Remark"


def load_and_clean_with_pyspark(spark: SparkSession) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load `main_dataset.csv` in PySpark, scrub PII, audit all 84 rows,
    separate 72 valid rows from 12 excluded rows, and compute indices + themes.
    """
    raw = (
        spark.read.option("header", "true")
        .option("multiLine", "true")
        .option("quote", '"')
        .option("escape", '"')
        .option("encoding", "UTF-8")
        .csv(RAW_CSV)
    )

    short_cols = [
        "timestamp", "username", "consent", "age_group_raw", "ai_usage_freq",
        "aware_energy", "aware_carbon", "consider_impact", "avoid_unnecessary_use",
        "concern_text", "awareness_text",
    ]
    df = raw.toDF(*short_cols)

    # Assign sequential anonymized IDs and remove PII (Timestamp, Username)
    df = df.withColumn("row_num", F.monotonically_increasing_id() + 1)
    df = df.withColumn("respondent_id", F.concat(F.lit("RESP_"), F.lpad(F.col("row_num").cast("string"), 3, "0")))

    # Trim whitespace & normalize newlines in text columns
    for c in ["consent", "age_group_raw", "ai_usage_freq", "aware_energy",
              "aware_carbon", "consider_impact", "avoid_unnecessary_use",
              "concern_text", "awareness_text"]:
        df = df.withColumn(c, F.trim(F.regexp_replace(F.coalesce(F.col(c), F.lit("")), r"\s+", " ")))

    # Fix age label typo ('24-34' -> '25-34')
    df = df.withColumn(
        "age_group",
        F.when(F.col("age_group_raw") == "24-34", F.lit("25-34")).otherwise(F.col("age_group_raw")),
    )

    # Encode Q2 (usage frequency) and Q3..Q6 (5-point Likert)
    freq_map = F.create_map(
        F.lit("rarely"), F.lit(1.0),
        F.lit("once a week"), F.lit(2.0),
        F.lit("several times a week"), F.lit(3.0),
        F.lit("once a day"), F.lit(4.0),
        F.lit("several times a day"), F.lit(5.0),
    )
    likert_map = F.create_map(
        F.lit("strongly disagree"), F.lit(1.0),
        F.lit("disagree"), F.lit(2.0),
        F.lit("neutral"), F.lit(3.0),
        F.lit("agree"), F.lit(4.0),
        F.lit("strongly agree"), F.lit(5.0),
    )

    df = (
        df.withColumn("usage_freq_n", freq_map[F.lower(F.col("ai_usage_freq"))])
        .withColumn("aware_energy_n", likert_map[F.lower(F.col("aware_energy"))])
        .withColumn("aware_carbon_n", likert_map[F.lower(F.col("aware_carbon"))])
        .withColumn("consider_impact_n", likert_map[F.lower(F.col("consider_impact"))])
        .withColumn("avoid_use_n", likert_map[F.lower(F.col("avoid_unnecessary_use"))])
    )

    # Audit validity of open-ended Q7 & Q8
    q7_empty = F.col("concern_text") == ""
    q8_empty = F.col("awareness_text") == ""
    both_blank = q7_empty & q8_empty
    placeholder_only = (~both_blank) & F.col("concern_text").isin("", ".", "-") & F.col("awareness_text").isin("", ".", "-")
    combined_lower = F.lower(F.concat_ws(" ", F.col("concern_text"), F.col("awareness_text")))
    is_off_topic = (~both_blank) & (~placeholder_only) & (~combined_lower.rlike(ENV_TOPIC_REGEX))

    copied_q8_prompt = F.lower(F.col("awareness_text")).contains(
        "what do you think should be done to increase public awareness"
    )

    df = (
        df.withColumn(
            "record_status",
            F.when(both_blank, F.lit("Excluded: Blank Q7 & Q8"))
            .when(placeholder_only, F.lit("Excluded: Placeholder ('.')"))
            .when(is_off_topic, F.lit("Excluded: Unrelated / Off-Topic Q7"))
            .otherwise(F.lit("Valid")),
        )
        .withColumn("is_valid", F.when(F.col("record_status") == "Valid", F.lit(1)).otherwise(F.lit(0)))
        .withColumn(
            "q8_status",
            F.when(F.col("record_status") != "Valid", F.col("record_status"))
            .when(copied_q8_prompt, F.lit("Invalid Q8: Copied Prompt"))
            .otherwise(F.lit("Valid")),
        )
    )

    # Composite indices & behavioral labels
    df = (
        df.withColumn("Awareness_Index", F.round((F.col("aware_energy_n") + F.col("aware_carbon_n")) / 2.0, 2))
        .withColumn("Behavior_Index", F.round((F.col("consider_impact_n") + F.col("avoid_use_n")) / 2.0, 2))
        .withColumn("Awareness_Behavior_Gap", F.round(F.col("Awareness_Index") - F.col("Behavior_Index"), 2))
        .withColumn(
            "usage_segment",
            F.when(F.col("usage_freq_n") >= 4.0, F.lit("Heavy (Daily+)"))
            .when(F.col("usage_freq_n") >= 3.0, F.lit("Moderate (Weekly)"))
            .otherwise(F.lit("Light (Rarely/1x Wk)")),
        )
        .withColumn(
            "behavior_level",
            F.when(F.col("Behavior_Index") >= 4.0, F.lit("High"))
            .when(F.col("Behavior_Index") >= 3.0, F.lit("Medium"))
            .otherwise(F.lit("Low")),
        )
        .withColumn(
            "high_behavior",
            F.when(F.col("Behavior_Index") >= 4.0, F.lit(1)).otherwise(F.lit(0)),
        )
        .withColumn(
            "gap_type",
            F.when(F.col("Awareness_Behavior_Gap") >= 1.0, F.lit("Aware but not acting"))
            .when(F.col("Awareness_Behavior_Gap") <= -1.0, F.lit("Acting beyond awareness"))
            .when(F.col("Awareness_Index") >= 3.0, F.lit("Aware & taking action"))
            .otherwise(F.lit("Unaware & not acting")),
        )
    )

    # Regex theme coding for Q7 and Q8
    for col_name, pattern in Q7_THEMES.items():
        df = df.withColumn(
            col_name,
            F.when((F.col("is_valid") == 1) & F.lower(F.col("concern_text")).rlike(pattern), F.lit(1)).otherwise(F.lit(0)),
        )

    q7_any_specific = sum(F.col(c) for c in Q7_THEMES)
    df = df.withColumn(
        "c_general_env",
        F.when((F.col("is_valid") == 1) & (q7_any_specific == 0), F.lit(1)).otherwise(F.lit(0)),
    )

    for col_name, pattern in Q8_THEMES.items():
        df = df.withColumn(
            col_name,
            F.when((F.col("q8_status") == "Valid") & F.lower(F.col("awareness_text")).rlike(pattern), F.lit(1)).otherwise(F.lit(0)),
        )

    # Drop PII columns
    keep_cols = [
        "respondent_id", "age_group", "ai_usage_freq", "usage_freq_n",
        "aware_energy", "aware_energy_n", "aware_carbon", "aware_carbon_n",
        "consider_impact", "consider_impact_n", "avoid_unnecessary_use", "avoid_use_n",
        "Awareness_Index", "Behavior_Index", "Awareness_Behavior_Gap",
        "usage_segment", "behavior_level", "high_behavior", "gap_type",
        "concern_text", "awareness_text", "record_status", "q8_status", "is_valid",
        *list(Q7_LABELS.keys()), *list(Q8_LABELS.keys()),
    ]
    df_all = df.select(*keep_cols)
    df_clean = df_all.filter(F.col("is_valid") == 1)

    # Export cleaned CSV via Spark streaming writer
    write_spark_csv(df_clean, os.path.join(OUTPUT_DIR, "cleaned_main_dataset.csv"))

    # Collect to Pandas for statistical tests & visualization
    pdf_all = pd.DataFrame([r.asDict() for r in df_all.collect()])
    pdf_clean = pd.DataFrame([r.asDict() for r in df_clean.collect()])
    return pdf_all, pdf_clean


def build_q7_q8_audit_and_summary(pdf_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create row-level Q7/Q8 categorization table (all 84 rows) and category summary."""
    rows_audit = []
    for _, r in pdf_all.iterrows():
        if r["is_valid"] == 1:
            q7_cats = [Q7_LABELS[c] for c in Q7_LABELS if r[c] == 1]
            q7_label = "; ".join(q7_cats)
            if r["q8_status"] == "Valid":
                q8_cats = [Q8_LABELS[c] for c in Q8_LABELS if r[c] == 1]
                q8_label = "; ".join(q8_cats) if q8_cats else "General Awareness"
            else:
                q8_label = r["q8_status"]
        else:
            if r["record_status"] == "Excluded: Unrelated / Off-Topic Q7":
                q7_label = classify_off_topic_subtype(str(r["concern_text"]))
                q8_label = "Excluded (Respondent Q7 Unrelated)"
            else:
                q7_label = r["record_status"]
                q8_label = r["record_status"]

        rows_audit.append({
            "respondent_id": r["respondent_id"],
            "age_group": r["age_group"],
            "usage_segment": r["usage_segment"],
            "record_status": r["record_status"],
            "q7_categories": q7_label,
            "q8_categories": q8_label,
            "concern_text": r["concern_text"],
            "awareness_text": r["awareness_text"],
        })

    audit_df = pd.DataFrame(rows_audit)
    audit_df.to_csv(os.path.join(OUTPUT_DIR, "q7_q8_categorization_all84.csv"), index=False, encoding="utf-8-sig")

    # Summary counts (both % of valid and % of all 84 raw)
    n_raw = len(pdf_all)
    n_valid_q7 = int((pdf_all["is_valid"] == 1).sum())
    n_valid_q8 = int((pdf_all["q8_status"] == "Valid").sum())
    n_off_topic = int((pdf_all["record_status"] == "Excluded: Unrelated / Off-Topic Q7").sum())
    n_blank_dot = int(pdf_all["record_status"].isin(["Excluded: Blank Q7 & Q8", "Excluded: Placeholder ('.')"]).sum())
    n_copied_q8 = int((pdf_all["q8_status"] == "Invalid Q8: Copied Prompt").sum())

    summary_rows = []
    for col, lbl in Q7_LABELS.items():
        cnt = int(pdf_all[col].sum())
        summary_rows.append({
            "question": "Q7",
            "group": "Valid Environmental Concern",
            "category": lbl,
            "count": cnt,
            "pct_of_valid": round(cnt * 100.0 / n_valid_q7, 1),
            "pct_of_raw84": round(cnt * 100.0 / n_raw, 1),
        })
    summary_rows.append({
        "question": "Q7", "group": "Unrelated / Excluded",
        "category": "Unrelated / Off-Topic (Non-Env)",
        "count": n_off_topic, "pct_of_valid": np.nan, "pct_of_raw84": round(n_off_topic * 100.0 / n_raw, 1),
    })
    summary_rows.append({
        "question": "Q7", "group": "Unrelated / Excluded",
        "category": "Blank / Placeholder ('.')",
        "count": n_blank_dot, "pct_of_valid": np.nan, "pct_of_raw84": round(n_blank_dot * 100.0 / n_raw, 1),
    })

    for col, lbl in Q8_LABELS.items():
        cnt = int(pdf_all[col].sum())
        summary_rows.append({
            "question": "Q8",
            "group": "Valid Suggested Solution",
            "category": lbl,
            "count": cnt,
            "pct_of_valid": round(cnt * 100.0 / n_valid_q8, 1),
            "pct_of_raw84": round(cnt * 100.0 / n_raw, 1),
        })
    summary_rows.append({
        "question": "Q8", "group": "Unrelated / Invalid",
        "category": "Excluded w/ Off-Topic Q7",
        "count": n_off_topic, "pct_of_valid": np.nan, "pct_of_raw84": round(n_off_topic * 100.0 / n_raw, 1),
    })
    summary_rows.append({
        "question": "Q8", "group": "Unrelated / Invalid",
        "category": "Blank / Placeholder ('.')",
        "count": n_blank_dot, "pct_of_valid": np.nan, "pct_of_raw84": round(n_blank_dot * 100.0 / n_raw, 1),
    })
    summary_rows.append({
        "question": "Q8", "group": "Unrelated / Invalid",
        "category": "Invalid: Copied Prompt Text",
        "count": n_copied_q8, "pct_of_valid": np.nan, "pct_of_raw84": round(n_copied_q8 * 100.0 / n_raw, 1),
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "q7_q8_category_summary.csv"), index=False, encoding="utf-8-sig")
    return audit_df, summary_df


def compute_statistics(pdf_all: pd.DataFrame, pdf_clean: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Compute cleaning audit, descriptive stats, Likert table, usage segments, gap typology, and correlations."""
    # 1. Cleaning audit
    clean_audit = (
        pdf_all.groupby("record_status", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )
    clean_audit["pct_of_raw84"] = (clean_audit["count"] * 100.0 / len(pdf_all)).round(2)
    clean_audit.to_csv(os.path.join(OUTPUT_DIR, "cleaning_audit_all84.csv"), index=False, encoding="utf-8-sig")

    # 2. Descriptive statistics (N = 72)
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
        s = pdf_clean[col].astype(float)
        mode_v = float(s.mode().iloc[0])
        desc_rows.append({
            "variable": col,
            "label": lbl,
            "n": len(s),
            "mean": round(float(s.mean()), 2),
            "sd": round(float(s.std(ddof=1)), 2),
            "median": round(float(s.median()), 2),
            "mode": round(mode_v, 2),
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
        })
    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(os.path.join(OUTPUT_DIR, "descriptive_stats.csv"), index=False, encoding="utf-8-sig")

    # Cronbach's alpha on Q3..Q6
    items = pdf_clean[["aware_energy_n", "aware_carbon_n", "consider_impact_n", "avoid_use_n"]].values.astype(float)
    alpha = (4.0 / 3.0) * (1.0 - items.var(axis=0, ddof=1).sum() / items.sum(axis=1).var(ddof=1))

    # 3. Likert distribution (Q3..Q6)
    likert_order = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
    likert_items = [
        ("aware_energy", "Q3: Aware AI uses lots of energy"),
        ("aware_carbon", "Q4: Aware data centers add emissions"),
        ("consider_impact", "Q5: Consider impact when using AI"),
        ("avoid_unnecessary_use", "Q6: Avoid unnecessary AI use"),
    ]
    likert_rows = []
    n_c = len(pdf_clean)
    for col, lbl in likert_items:
        vc = pdf_clean[col].str.title().value_counts().to_dict()
        row = {"item": lbl}
        for lvl in likert_order:
            row[lvl] = int(vc.get(lvl, 0))
        row["pct_agree"] = round((row["Agree"] + row["Strongly Agree"]) * 100.0 / n_c, 1)
        row["pct_neutral"] = round(row["Neutral"] * 100.0 / n_c, 1)
        row["pct_disagree"] = round((row["Disagree"] + row["Strongly Disagree"]) * 100.0 / n_c, 1)
        likert_rows.append(row)
    likert_df = pd.DataFrame(likert_rows)
    likert_df.to_csv(os.path.join(OUTPUT_DIR, "likert_distribution.csv"), index=False, encoding="utf-8-sig")

    # 4. Usage segments
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
    seg_df["pct"] = (seg_df["n"] * 100.0 / n_c).round(1)
    for c in ["mean_awareness", "sd_awareness", "mean_behavior", "sd_behavior", "mean_gap"]:
        seg_df[c] = seg_df[c].round(2)
    seg_df["usage_segment"] = pd.Categorical(seg_df["usage_segment"], seg_order, ordered=True)
    seg_df = seg_df.sort_values("usage_segment").reset_index(drop=True)
    seg_df.to_csv(os.path.join(OUTPUT_DIR, "usage_segment_summary.csv"), index=False, encoding="utf-8-sig")

    # 5. Age groups
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
    age_df["pct"] = (age_df["n"] * 100.0 / n_c).round(1)
    for c in ["mean_usage", "mean_awareness", "mean_behavior"]:
        age_df[c] = age_df[c].round(2)
    age_df.to_csv(os.path.join(OUTPUT_DIR, "age_group_summary.csv"), index=False, encoding="utf-8-sig")

    # 6. Gap typology
    gap_df = (
        pdf_clean.groupby("gap_type", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )
    gap_df["pct"] = (gap_df["count"] * 100.0 / n_c).round(1)
    gap_df.to_csv(os.path.join(OUTPUT_DIR, "gap_typology_summary.csv"), index=False, encoding="utf-8-sig")

    # 7. Pearson & Spearman correlations with p-values
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
    for a, b in pairs:
        x = pdf_clean[a].astype(float)
        y = pdf_clean[b].astype(float)
        r, p = stats.pearsonr(x, y)
        rho, rho_p = stats.spearmanr(x, y)
        corr_rows.append({
            "var_1": a,
            "var_2": b,
            "pearson_r": round(float(r), 4),
            "pearson_p": float(f"{p:.4e}"),
            "spearman_rho": round(float(rho), 4),
            "spearman_p": float(f"{rho_p:.4e}"),
            "significant_05": "Yes" if p < 0.05 else "No",
        })
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(os.path.join(OUTPUT_DIR, "correlations.csv"), index=False, encoding="utf-8-sig")

    print(f"Cronbach's Alpha (Q3..Q6, N = {n_c}): {alpha:.4f}")
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


# ==============================================================================
# PUBLICATION-GRADE VISUALIZATIONS (4 FIGURES)
# ==============================================================================
def plot_fig1_cleaning_audit(pdf_all: pd.DataFrame, audit_df: pd.DataFrame) -> None:
    """Figure 1: Data Cleaning Audit of Main Survey Dataset (84 Raw -> 72 Cleaned, 12 Excluded)."""
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.3), gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.78})

    n_valid = int((pdf_all["is_valid"] == 1).sum())
    n_off = int((pdf_all["record_status"] == "Excluded: Unrelated / Off-Topic Q7").sum())
    n_blank = int((pdf_all["record_status"] == "Excluded: Blank Q7 & Q8").sum())
    n_dot = int((pdf_all["record_status"] == "Excluded: Placeholder ('.')").sum())

    cats_a = ["Cleaned & Kept\n(Valid Env. Data)", "Unrelated / Off-Topic\n(Non-Env. Q7)", "Blank Form\n(No Q7 & Q8)", "Placeholder Only\n('.' Entered)"]
    vals_a = [n_valid, n_off, n_blank, n_dot]
    cols_a = ["#16a34a", "#dc2626", "#94a3b8", "#64748b"]

    y_a = np.arange(len(cats_a))[::-1]
    bars_a = axes[0].barh(y_a, vals_a, height=0.58, color=cols_a, edgecolor="white", linewidth=0.9)
    for bar, v in zip(bars_a, vals_a):
        pct = v * 100.0 / len(pdf_all)
        axes[0].text(v + 1.5, bar.get_y() + bar.get_height() / 2, f"{v} ({pct:.1f}%)", va="center", fontweight="bold", fontsize=8.5)
    axes[0].set_yticks(y_a)
    axes[0].set_yticklabels(cats_a, fontweight="bold", fontsize=8.5)
    axes[0].set_xlim(0, 102)
    axes[0].set_xlabel("Number of Survey Responses (Total Raw = 84)")
    axes[0].set_title("A. Raw Survey Cleaning Outcome (N = 84)", fontweight="bold", pad=10)

    # Panel B: Exact breakdown of the 12 Excluded / Unrelated responses
    excl_df = audit_df[audit_df["record_status"] != "Valid"].copy()
    sub_counts = excl_df["q7_categories"].value_counts().sort_values(ascending=True)
    y_b = np.arange(len(sub_counts))
    cols_b = ["#dc2626" if "Unrelated:" in idx else "#64748b" for idx in sub_counts.index]
    bars_b = axes[1].barh(y_b, sub_counts.values, height=0.62, color=cols_b, alpha=0.9, edgecolor="white", hatch="//")
    for bar, v in zip(bars_b, sub_counts.values):
        axes[1].text(v + 0.12, bar.get_y() + bar.get_height() / 2, f"n = {int(v)}", va="center", fontweight="bold", fontsize=8.2)
    axes[1].set_yticks(y_b)
    axes[1].set_yticklabels(sub_counts.index, fontsize=8.2, fontweight="bold")
    axes[1].set_xlim(0, max(sub_counts.values) + 1.3)
    axes[1].set_xticks(range(0, int(max(sub_counts.values)) + 2))
    axes[1].set_xlabel("Excluded Responses (Total Removed = 12)")
    axes[1].set_title("B. Exact Breakdown of 12 Excluded & Unrelated Rows", fontweight="bold", pad=10)

    fig.suptitle("Data Cleaning Audit: Main Community Survey (84 Raw -> 72 Clean Records)", fontweight="bold", fontsize=11.5, y=1.03)
    fig.savefig(os.path.join(FIG_DIR, "fig1_data_cleaning_audit.png"), bbox_inches="tight")
    plt.close(fig)


def plot_fig2_likert_and_usage(likert_df: pd.DataFrame, seg_df: pd.DataFrame) -> None:
    """Figure 2: 100% Stacked Likert Responses (Q3-Q6) and Mean Indices by Usage Segment (N = 72)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), gridspec_kw={"width_ratios": [1.3, 1.05], "wspace": 0.32})

    # Panel A: Likert 100% stacked bar
    order = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
    colors = ["#b2182b", "#ef8a62", "#d9d9d9", "#67a9cf", "#2166ac"]
    tot = likert_df[order].sum(axis=1)
    share = likert_df[order].div(tot, axis=0) * 100.0
    labels = likert_df["item"].tolist()[::-1]
    share = share.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(labels))
    left = np.zeros(len(labels))

    for cat, colr in zip(order, colors):
        w = share[cat].values
        axes[0].barh(y_pos, w, left=left, height=0.58, color=colr, label=cat, edgecolor="white", linewidth=0.8)
        txt_col = "white" if cat in ("Strongly Disagree", "Strongly Agree") else "#1e293b"
        for i, (st, wv) in enumerate(zip(left, w)):
            if wv >= 6.0:
                axes[0].text(st + wv / 2, i, f"{wv:.0f}%", va="center", ha="center", color=txt_col, fontweight="bold", fontsize=7.8)
        left += w

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels, fontsize=8.4, fontweight="bold")
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Percentage of Cleaned Respondents (%)")
    axes[0].set_title("A. Likert Distribution: Awareness & Behavior (N = 72)", fontweight="bold", pad=10)
    axes[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(0.48, -0.18), frameon=False, fontsize=7.8)

    # Panel B: Awareness vs Behavior by AI Usage Segment
    x = np.arange(len(seg_df))
    bw = 0.35
    b1 = axes[1].bar(x - bw / 2, seg_df["mean_awareness"], bw, label="Awareness Index (Q3-Q4)", color="#2166ac", edgecolor="white")
    b2 = axes[1].bar(x + bw / 2, seg_df["mean_behavior"], bw, label="Behavior Index (Q5-Q6)", color="#ef8a62", edgecolor="white")
    axes[1].bar_label(b1, fmt="%.2f", padding=2, fontsize=8.3, fontweight="bold")
    axes[1].bar_label(b2, fmt="%.2f", padding=2, fontsize=8.3, fontweight="bold")
    seg_x_labels = [
        f"{s.split(' (')[0]}\n({s.split(' (')[1]}\n(n={int(n)})"
        for s, n in zip(seg_df["usage_segment"], seg_df["n"])
    ]
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(seg_x_labels, fontsize=8.2, fontweight="bold")
    axes[1].set_ylim(0, 5.5)
    axes[1].set_ylabel("Mean Score (1-5 Scale)")
    axes[1].set_title("B. Mean Indices by AI Usage Segment (N = 72)", fontweight="bold", pad=10)
    axes[1].legend(frameon=False, loc="upper right", fontsize=8.0)

    fig.savefig(os.path.join(FIG_DIR, "fig2_likert_and_usage.png"), bbox_inches="tight")
    plt.close(fig)


def plot_fig3_q7_q8_categorization(summary_df: pd.DataFrame) -> None:
    """
    Figure 3: Complete Q7 & Q8 Qualitative Categorization (N = 84 Raw),
    explicitly separating Valid Themes from Marked Unrelated / Invalid Responses.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), gridspec_kw={"wspace": 0.62})

    for ax, q, valid_col, title in [
        (axes[0], "Q7", "#0f766e", "A. Q7 Environmental Concerns (N = 84 Raw)\n[72 Valid Multi-Theme | 12 Excluded Marked]"),
        (axes[1], "Q8", "#1d4ed8", "B. Q8 Suggested Solutions (N = 84 Raw)\n[71 Valid Multi-Theme | 13 Excluded/Invalid Marked]"),
    ]:
        sub = summary_df[summary_df["question"] == q].copy()
        valid_part = sub[sub["group"].str.startswith("Valid")].sort_values("count", ascending=True)
        excl_part = sub[~sub["group"].str.startswith("Valid")].sort_values("count", ascending=True)

        n_excl = len(excl_part)
        n_val = len(valid_part)
        y_excl = np.arange(n_excl)
        y_val = np.arange(n_excl + 0.75, n_excl + 0.75 + n_val)

        bars_v = ax.barh(y_val, valid_part["count"], height=0.62, color=valid_col, alpha=0.92, edgecolor="white", label="Valid Theme (Included)")
        for bar, (_, r) in zip(bars_v, valid_part.iterrows()):
            ax.text(
                r["count"] + 0.9, bar.get_y() + bar.get_height() / 2,
                f"n={int(r['count'])} ({r['pct_of_valid']:.1f}% valid)",
                va="center", fontsize=8.0, fontweight="bold", color="#0f172a",
            )

        excl_colors = ["#dc2626" if ("Off-Topic" in c or "Copied" in c) else "#64748b" for c in excl_part["category"]]
        bars_e = ax.barh(y_excl, excl_part["count"], height=0.62, color=excl_colors, alpha=0.88, edgecolor="white", hatch="///", label="Marked Unrelated / Invalid (Excluded)")
        for bar, (_, r) in zip(bars_e, excl_part.iterrows()):
            ax.text(
                r["count"] + 0.9, bar.get_y() + bar.get_height() / 2,
                f"n={int(r['count'])} ({r['pct_of_raw84']:.1f}% raw) [EXCLUDED]",
                va="center", fontsize=7.9, fontweight="bold", color="#991b1b",
            )

        sep_y = n_excl - 0.12
        ax.axhline(sep_y, color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.85)
        x_lim_max = sub["count"].max() * 1.62
        ax.text(
            x_lim_max * 0.98, sep_y + 0.10,
            "Excluded / Unrelated below line",
            ha="right", va="bottom", fontsize=7.4, style="italic", color="#991b1b",
        )

        all_y = np.concatenate([y_excl, y_val])
        all_labels = list(excl_part["category"]) + list(valid_part["category"])
        ax.set_yticks(all_y)
        ax.set_yticklabels(all_labels, fontsize=8.3, fontweight="bold")
        for idx_tick, lbl_t in enumerate(ax.get_yticklabels()):
            if idx_tick < n_excl:
                lbl_t.set_color("#991b1b")

        ax.set_xlim(0, x_lim_max)
        ax.set_xlabel("Number of Mentions / Responses")
        ax.set_title(title, fontweight="bold", fontsize=9.3, pad=10)

    # Clean shared legend below both subplots so zero bars or annotations are covered
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#0f766e", edgecolor="white", label="Q7 Valid Environmental Concern (Included)"),
        Patch(facecolor="#1d4ed8", edgecolor="white", label="Q8 Valid Suggested Solution (Included)"),
        Patch(facecolor="#dc2626", edgecolor="white", hatch="///", label="Unrelated / Off-Topic / Copied Prompt (Excluded)"),
        Patch(facecolor="#64748b", edgecolor="white", hatch="///", label="Blank / Placeholder '.' (Excluded)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.2)
    fig.suptitle(
        "Qualitative Categorization of Open-Ended Responses (Q7 & Q8) with Unrelated/Invalid Entries Marked",
        fontweight="bold", fontsize=11.3, y=1.02,
    )
    fig.savefig(os.path.join(FIG_DIR, "fig3_q7_q8_categorization.png"), bbox_inches="tight")
    plt.close(fig)


def plot_fig4_correlation_and_gap(pdf_clean: pd.DataFrame, gap_df: pd.DataFrame) -> None:
    """Figure 4: Pearson Correlation Matrix (Q2-Q6) & Awareness-Behavior Alignment Typology (N = 72)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), gridspec_kw={"wspace": 0.30})

    hm_vars = ["aware_energy_n", "aware_carbon_n", "consider_impact_n", "avoid_use_n", "usage_freq_n"]
    hm_lbls = ["Q3\nEnergy", "Q4\nCarbon", "Q5\nConsider", "Q6\nAvoid", "Q2\nUsage"]
    cm = pdf_clean[hm_vars].astype(float).corr().values

    im = axes[0].imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_xticks(range(5))
    axes[0].set_xticklabels(hm_lbls, fontsize=8.2, fontweight="bold")
    axes[0].set_yticks(range(5))
    axes[0].set_yticklabels(hm_lbls, fontsize=8.2, fontweight="bold")
    for i in range(5):
        for j in range(5):
            axes[0].text(
                j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8.2,
                color="white" if abs(cm[i, j]) > 0.5 else "black", fontweight="bold",
            )
    axes[0].set_title("A. Pearson Correlation Matrix (N = 72)", fontweight="bold", pad=10)
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    pie_colors = {
        "Aware & taking action": "#2166ac",
        "Unaware & not acting": "#94a3b8",
        "Aware but not acting": "#ef8a62",
        "Acting beyond awareness": "#16a34a",
    }
    gp = gap_df.set_index("gap_type")["count"]
    wedges, texts, autotexts = axes[1].pie(
        gp.values,
        labels=[f"{k}\n(n={int(v)})" for k, v in gp.items()],
        autopct="%1.1f%%",
        pctdistance=0.72,
        startangle=135,
        colors=[pie_colors.get(k, "#94a3b8") for k in gp.index],
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        textprops={"fontsize": 8.2, "fontweight": "bold"},
    )
    for w_idx, k in enumerate(gp.index):
        if k in ("Aware & taking action", "Acting beyond awareness"):
            autotexts[w_idx].set_color("white")
    axes[1].set_title("B. Awareness-Behavior Alignment (N = 72)", fontweight="bold", pad=10)

    fig.savefig(os.path.join(FIG_DIR, "fig4_correlation_and_gap.png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 78)
    print("  STEP 1: MAIN DATASET PYSPARK CLEANING, Q7/Q8 CATEGORIZATION & ANALYSIS")
    print("=" * 78)

    spark = create_spark()
    pdf_all, pdf_clean = load_and_clean_with_pyspark(spark)
    spark.stop()

    print(f"Raw records loaded   : {len(pdf_all)}")
    print(f"Cleaned valid records: {len(pdf_clean)} ({len(pdf_clean)*100.0/len(pdf_all):.2f}%)")
    print(f"Excluded records     : {len(pdf_all) - len(pdf_clean)}")

    audit_df, q7q8_summary = build_q7_q8_audit_and_summary(pdf_all)
    stats_out = compute_statistics(pdf_all, pdf_clean)

    print("\n--- DESCRIPTIVE STATISTICS (N = 72) ---")
    print(stats_out["desc"][["label", "mean", "sd", "median", "mode"]].to_string(index=False))

    print("\n--- Q7 & Q8 CATEGORY SUMMARY (ALL 84 RAW, UNRELATED MARKED) ---")
    print(q7q8_summary.to_string(index=False))

    print("\n--- KEY CORRELATIONS (N = 72) ---")
    print(stats_out["corr"].to_string(index=False))

    plot_fig1_cleaning_audit(pdf_all, audit_df)
    plot_fig2_likert_and_usage(stats_out["likert"], stats_out["seg"])
    plot_fig3_q7_q8_categorization(q7q8_summary)
    plot_fig4_correlation_and_gap(pdf_clean, stats_out["gap"])

    print(f"\nSaved cleaned data, CSV summaries, and 4 figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

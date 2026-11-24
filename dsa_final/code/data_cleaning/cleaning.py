"""
cleaning.py
===========
PySpark data ingestion, PII scrubbing, validity auditing, Likert encoding,
composite index engineering, and open-ended theme tagging.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from pyspark.sql import Column, DataFrame, SparkSession, functions as F

_CODE_DIR = str(Path(__file__).resolve().parents[1])
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from data_cleaning.config import (
    CLEANED_CSV,
    ENV_TOPIC_REGEX,
    FREQ_SCORE_MAP,
    LIKERT_SCORE_MAP,
    Q7_LABELS,
    Q7_THEMES,
    Q8_LABELS,
    Q8_THEMES,
    RAW_CSV,
)
from data_cleaning.spark_utils import create_spark_session, write_spark_csv

RAW_SHORT_COLS = [
    "timestamp", "username", "consent", "age_group_raw", "ai_usage_freq",
    "aware_energy", "aware_carbon", "consider_impact", "avoid_unnecessary_use",
    "concern_text", "awareness_text",
]

TEXT_CLEAN_COLS = [
    "consent", "age_group_raw", "ai_usage_freq", "aware_energy",
    "aware_carbon", "consider_impact", "avoid_unnecessary_use",
    "concern_text", "awareness_text",
]

EXPORT_COLS = [
    "respondent_id", "age_group", "ai_usage_freq", "usage_freq_n",
    "aware_energy", "aware_energy_n", "aware_carbon", "aware_carbon_n",
    "consider_impact", "consider_impact_n", "avoid_unnecessary_use", "avoid_use_n",
    "Awareness_Index", "Behavior_Index", "Awareness_Behavior_Gap",
    "usage_segment", "behavior_level", "high_behavior", "gap_type",
    "concern_text", "awareness_text", "record_status", "q8_status", "is_valid",
    *list(Q7_LABELS.keys()), *list(Q8_LABELS.keys()),
]


def _build_spark_map(mapping: Dict[str, float]) -> Column:
    """Convert a Python string-to-float dictionary into a PySpark map column."""
    args = []
    for k, v in mapping.items():
        args.extend([F.lit(k), F.lit(v)])
    return F.create_map(*args)


def read_raw_survey(spark: SparkSession, csv_path: str = RAW_CSV) -> DataFrame:
    """Read the raw Google Forms CSV with multi-line quote support."""
    raw = (
        spark.read.option("header", "true")
        .option("multiLine", "true")
        .option("quote", '"')
        .option("escape", '"')
        .option("encoding", "UTF-8")
        .csv(csv_path)
    )
    return raw.toDF(*RAW_SHORT_COLS)


def scrub_pii_and_normalize(df: DataFrame) -> DataFrame:
    """Assign anonymized respondent IDs, trim whitespace, and fix age-group typo."""
    df = df.withColumn("row_num", F.monotonically_increasing_id() + 1)
    df = df.withColumn(
        "respondent_id",
        F.concat(F.lit("RESP_"), F.lpad(F.col("row_num").cast("string"), 3, "0")),
    )
    for col_name in TEXT_CLEAN_COLS:
        df = df.withColumn(
            col_name,
            F.trim(F.regexp_replace(F.coalesce(F.col(col_name), F.lit("")), r"\s+", " ")),
        )
    return df.withColumn(
        "age_group",
        F.when(F.col("age_group_raw") == "24-34", F.lit("25-34")).otherwise(F.col("age_group_raw")),
    )


def encode_survey_items(df: DataFrame) -> DataFrame:
    """Encode Q2 usage frequency and Q3-Q6 5-point Likert items into numeric scores."""
    freq_map = _build_spark_map(FREQ_SCORE_MAP)
    likert_map = _build_spark_map(LIKERT_SCORE_MAP)
    return (
        df.withColumn("usage_freq_n", freq_map[F.lower(F.col("ai_usage_freq"))])
        .withColumn("aware_energy_n", likert_map[F.lower(F.col("aware_energy"))])
        .withColumn("aware_carbon_n", likert_map[F.lower(F.col("aware_carbon"))])
        .withColumn("consider_impact_n", likert_map[F.lower(F.col("consider_impact"))])
        .withColumn("avoid_use_n", likert_map[F.lower(F.col("avoid_unnecessary_use"))])
    )


def audit_open_ended_validity(df: DataFrame) -> DataFrame:
    """Flag blank, placeholder, off-topic, and copied-prompt Q7/Q8 responses."""
    q7_empty = F.col("concern_text") == ""
    q8_empty = F.col("awareness_text") == ""
    both_blank = q7_empty & q8_empty
    placeholder_only = (
        (~both_blank)
        & F.col("concern_text").isin("", ".", "-")
        & F.col("awareness_text").isin("", ".", "-")
    )
    combined_lower = F.lower(F.concat_ws(" ", F.col("concern_text"), F.col("awareness_text")))
    is_off_topic = (~both_blank) & (~placeholder_only) & (~combined_lower.rlike(ENV_TOPIC_REGEX))
    copied_q8_prompt = F.lower(F.col("awareness_text")).contains(
        "what do you think should be done to increase public awareness"
    )

    return (
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


def add_composite_indices(df: DataFrame) -> DataFrame:
    """Compute Awareness_Index, Behavior_Index, gap, usage segment, and tiers."""
    return (
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
        .withColumn("high_behavior", F.when(F.col("Behavior_Index") >= 4.0, F.lit(1)).otherwise(F.lit(0)))
        .withColumn(
            "gap_type",
            F.when(F.col("Awareness_Behavior_Gap") >= 1.0, F.lit("Aware but not acting"))
            .when(F.col("Awareness_Behavior_Gap") <= -1.0, F.lit("Acting beyond awareness"))
            .when(F.col("Awareness_Index") >= 3.0, F.lit("Aware & taking action"))
            .otherwise(F.lit("Unaware & not acting")),
        )
    )


def tag_open_ended_themes(df: DataFrame) -> DataFrame:
    """Apply multi-label PySpark regex coding for Q7 concerns and Q8 solutions."""
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
    return df


def load_and_clean_with_pyspark(
    spark: SparkSession,
    export_csv: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the complete PySpark cleaning workflow on `main_dataset.csv`,
    export `cleaned_main_dataset.csv`, and return `(pdf_all, pdf_clean)`.
    """
    df = read_raw_survey(spark)
    df = scrub_pii_and_normalize(df)
    df = encode_survey_items(df)
    df = audit_open_ended_validity(df)
    df = add_composite_indices(df)
    df = tag_open_ended_themes(df)

    df_all = df.select(*EXPORT_COLS)
    df_clean = df_all.filter(F.col("is_valid") == 1)

    if export_csv:
        write_spark_csv(df_clean, CLEANED_CSV)

    pdf_all = pd.DataFrame([r.asDict() for r in df_all.collect()])
    pdf_clean = pd.DataFrame([r.asDict() for r in df_clean.collect()])
    return pdf_all, pdf_clean


def main() -> None:
    """Standalone runner for PySpark data cleaning and export."""
    spark = create_spark_session()
    try:
        pdf_all, pdf_clean = load_and_clean_with_pyspark(spark, export_csv=True)
    finally:
        spark.stop()
    print(f"Raw records loaded   : {len(pdf_all)}")
    print(f"Cleaned valid records: {len(pdf_clean)} ({len(pdf_clean) * 100.0 / len(pdf_all):.2f}%)")
    print(f"Excluded records     : {len(pdf_all) - len(pdf_clean)}")
    print(f"Saved cleaned CSV    : {os.path.abspath(CLEANED_CSV)}")


if __name__ == "__main__":
    main()

"""
02_data_cleansing.py
====================
RUN ORDER: 2 of 5

Stage 2 — Topic-Aligned Data Cleansing & Unrelated/Invalid Row Removal (PySpark)
--------------------------------------------------------------------------------
Revolves strictly around our study in `docs/DSA - Proposal.pdf` and
`docs/Data Scalability and Analytics - Data transformation (Pyspark) (1).pdf`.

Reads `output/integration/stage1_integrated_dataset.csv` and performs:
  1. Removal of all invalid & off-topic/unrelated rows (`is_valid_record == 1`),
     exporting `output/cleansing/stage2_cleaning_loss_log.csv`
  2. Whitespace trimming & blank-to-NULL normalization across all string columns
  3. Age-group category normalization (`'24-34'` -> `'25-34'`)
  4. `[1.0, 5.0]` domain clamping & `Imputer(strategy="median")` for `usage_freq_n`,
     `aware_energy_n`, `aware_carbon_n`, `consider_impact_n`, `avoid_use_n`, followed by
     regenerating standardized Likert & usage-frequency labels
  5. Open-ended text cleansing on `concern_text` (Q7) and `awareness_text` (Q8)
     (removing junk tokens, verbatim copied prompts, and non-topic fragments, and
     creating `has_concern_text` and `has_awareness_text` flags)

Outputs -> `output/cleansing/`:
  - `stage2_cleansed_dataset.csv` (Cleaned topic-aligned unified dataset)
  - `stage2_cleaning_loss_log.csv` (Per-study cleaning retention & removal log)
"""

import os
from typing import Dict

from pyspark.ml.feature import Imputer
from pyspark.sql import DataFrame, functions as F, types as T

from pipeline_utils import (
    CLEANSING_DIR,
    JUNK_TEXT_TOKENS,
    MAIN_ITEM_SCORE_COLS,
    STAGE1_LOST_SUMMARY_CSV,
    STAGE1_UNIFIED_CSV,
    STAGE2_CLEANING_LOG_CSV,
    STAGE2_UNIFIED_CSV,
    banner,
    clamp_likert,
    create_spark_session,
    log,
    read_csv_spark,
    score_to_frequency_label,
    score_to_likert_label,
    write_spark_df_to_csv,
)


def filter_and_log_lost_rows(df: DataFrame, df_audit: DataFrame) -> DataFrame:
    """Filter out invalid and off-topic/unrelated rows (`is_valid_record == 0`) and export the loss log."""
    raw_total = df.count()
    write_spark_df_to_csv(df_audit, STAGE2_CLEANING_LOG_CSV)

    valid_df = (
        df.filter(F.col("is_valid_record").cast("int") == 1)
        .drop("is_valid_record", "invalid_reason")
        .dropDuplicates(["respondent_id"])
    )
    clean_total = valid_df.count()
    log(
        f"Topic-Aligned Cleaning Filter: {raw_total:,} raw rows -> "
        f"{clean_total:,} cleaned topic-valid rows ({raw_total - clean_total:,} invalid/unrelated rows removed)"
    )
    return valid_df


def trim_strings_and_fix_categories(df: DataFrame) -> DataFrame:
    """Trim whitespace across all string columns, convert empty strings to NULL, and fix age-band typos."""
    str_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, T.StringType)]
    for c in str_cols:
        trimmed = F.trim(F.col(c))
        df = df.withColumn(c, F.when(trimmed == "", None).otherwise(trimmed))

    df = df.withColumn(
        "age_group",
        F.when(F.col("age_group") == "24-34", "25-34")
        .when(F.col("age_group").isNull(), "18-24")
        .otherwise(F.col("age_group")),
    )
    log(f"Trimmed whitespace & normalized NULLs across {len(str_cols)} string columns; fixed '24-34' -> '25-34'")
    return df


def clean_and_impute_q2_to_q6(df: DataFrame) -> DataFrame:
    """Clamp Q2..Q6 numeric scores to `[1.0, 5.0]`, median-impute partial nulls, and refresh labels."""
    for c in MAIN_ITEM_SCORE_COLS:
        df = df.withColumn(c, clamp_likert(F.col(c)))

    null_counts: Dict[str, int] = {
        k: int(v)
        for k, v in (
            df.select([F.count(F.when(F.col(c).isNull() | F.isnan(c), c)).alias(c) for c in MAIN_ITEM_SCORE_COLS])
            .collect()[0]
            .asDict()
            .items()
        )
    }
    log(f"Missing values across Q2..Q6 scores before median imputation: {null_counts}")

    imputed_cols = [f"{c}_imp" for c in MAIN_ITEM_SCORE_COLS]
    imputer = Imputer(inputCols=MAIN_ITEM_SCORE_COLS, outputCols=imputed_cols).setStrategy("median")
    df = imputer.fit(df).transform(df)
    for c in MAIN_ITEM_SCORE_COLS:
        df = df.withColumn(c, F.round(F.col(f"{c}_imp"), 4)).drop(f"{c}_imp")

    return (
        df.withColumn("ai_usage_freq", score_to_frequency_label("usage_freq_n"))
        .withColumn("aware_energy", score_to_likert_label("aware_energy_n"))
        .withColumn("aware_carbon", score_to_likert_label("aware_carbon_n"))
        .withColumn("consider_impact", score_to_likert_label("consider_impact_n"))
        .withColumn("avoid_unnecessary_use", score_to_likert_label("avoid_use_n"))
    )


def clean_open_ended_q7_and_q8(df: DataFrame) -> DataFrame:
    """Clean `concern_text` (Q7) and `awareness_text` (Q8) and attach availability flags."""
    for col_name in ["concern_text", "awareness_text"]:
        norm = F.trim(F.regexp_replace(F.coalesce(F.col(col_name), F.lit("")), r"\s+", " "))
        lower_val = F.lower(norm)
        copied_prompt = (
            lower_val.contains("what do you think should be done to increase public awareness")
            | lower_val.contains("what environmental impact of artificial intelligence concerns you")
        )
        df = df.withColumn(
            col_name,
            F.when(
                (norm == "") | (F.length(norm) < 3) | lower_val.isin(JUNK_TEXT_TOKENS) | copied_prompt,
                None,
            ).otherwise(norm),
        )

    return (
        df.withColumn("has_concern_text", F.col("concern_text").isNotNull().cast("int"))
        .withColumn("has_awareness_text", F.col("awareness_text").isNotNull().cast("int"))
    )


def main() -> None:
    if not os.path.exists(STAGE1_UNIFIED_CSV):
        raise FileNotFoundError("Stage 1 output not found. Please run `01_data_integration.py` first.")

    os.makedirs(CLEANSING_DIR, exist_ok=True)
    spark = create_spark_session("DSA-Final-02-Topic-Aligned-Cleansing")
    banner("STAGE 2: TOPIC-ALIGNED DATA CLEANSING & UNRELATED/INVALID ROW REMOVAL")

    df_raw = read_csv_spark(spark, STAGE1_UNIFIED_CSV)
    df_audit = read_csv_spark(spark, STAGE1_LOST_SUMMARY_CSV)

    df_valid = filter_and_log_lost_rows(df_raw, df_audit)
    df_trimmed = trim_strings_and_fix_categories(df_valid)
    df_imputed = clean_and_impute_q2_to_q6(df_trimmed)
    df_clean = clean_open_ended_q7_and_q8(df_imputed)

    write_spark_df_to_csv(df_clean, STAGE2_UNIFIED_CSV)

    spark.stop()
    log("Stage 2 (Topic-Aligned Data Cleansing) finished successfully.")


if __name__ == "__main__":
    main()

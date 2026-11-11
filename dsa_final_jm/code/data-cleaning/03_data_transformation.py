"""
03_data_transformation.py
=========================
RUN ORDER: 3 of 5

Stage 3 — Feature Engineering, Composite Indices & NLP Theme Coding (PySpark)
-----------------------------------------------------------------------------
Reads `output/data-cleaning/cleansing/stage2_cleansed_dataset.csv` and performs:
  1. Composite Construct Indices:
       * awareness_score = mean(aw_*)
       * preparedness_score = mean(prep_*)
       * awareness_preparedness_gap = awareness_score - preparedness_score
       * aw_warning_index, aw_local_hazard_index, aw_safety_health_index
       * prep_supplies_index, prep_planning_index
  2. Diagnostic Segmentation:
       * awareness_level: "Low", "Moderate", "High"
       * preparedness_level: "Low", "Moderate", "High"
       * gap_typology: "Aligned", "Aware but Unprepared", "Prepared beyond Awareness"
       * risk_tier: "High Risk", "Moderate Risk", "Low Risk / Resilient"
  3. Qualitative NLP Keyword Theme Coding strictly for student challenges and institutional support:
       * Challenges (c_*): c_supplies_food, c_financial_cost, c_evac_access, c_power_comms, c_housing_safety
       * Support (s_*): s_go_bag_kits, s_early_warning_sms, s_evac_shelter, s_drrm_seminars, s_financial_aid
  4. PySpark ML Pipeline (`VectorAssembler`, `StandardScaler`) producing z-standardized scores (*_z).

Outputs -> `output/data-cleaning/transformation/`:
  - `stage3_transformed_dataset.csv` (Transformed unified dataset)
"""

import os
from typing import List

from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, functions as F

from pipeline_utils import (
    ALL_12_ITEM_COLS,
    ALL_QUALITATIVE_THEME_COLS,
    AWARENESS_ITEM_COLS,
    CHALLENGE_THEME_COLS,
    CORE_INDEX_COLS,
    PREPAREDNESS_ITEM_COLS,
    STAGE2_UNIFIED_CSV,
    STAGE3_UNIFIED_CSV,
    SUPPORT_THEME_COLS,
    TRANSFORMATION_DIR,
    banner,
    create_spark_session,
    log,
    read_csv_spark,
    write_spark_df_to_csv,
)


def compute_composite_indices(df: DataFrame) -> DataFrame:
    """Compute awareness, preparedness, gap, and sub-dimension indices."""
    # Cast all 12 items to double
    for c in ALL_12_ITEM_COLS:
        df = df.withColumn(c, F.col(c).cast("double"))

    # Overall indices
    aw_sum = sum(F.col(c) for c in AWARENESS_ITEM_COLS)
    prep_sum = sum(F.col(c) for c in PREPAREDNESS_ITEM_COLS)

    df = (
        df.withColumn("awareness_score", F.round(aw_sum / 6.0, 3))
        .withColumn("preparedness_score", F.round(prep_sum / 6.0, 3))
        .withColumn(
            "awareness_preparedness_gap",
            F.round(F.col("awareness_score") - F.col("preparedness_score"), 3)
        )
        .withColumn(
            "aw_warning_index",
            F.round((F.col("aw_warning_signals") + F.col("aw_monitor_updates")) / 2.0, 3)
        )
        .withColumn(
            "aw_local_hazard_index",
            F.round((F.col("aw_residence_risk") + F.col("aw_emergency_contacts")) / 2.0, 3)
        )
        .withColumn(
            "aw_safety_health_index",
            F.round((F.col("aw_evac_routes") + F.col("aw_health_risks")) / 2.0, 3)
        )
        .withColumn(
            "prep_supplies_index",
            F.round((F.col("prep_go_bag") + F.col("prep_food_water") + F.col("prep_secure_docs")) / 3.0, 3)
        )
        .withColumn(
            "prep_planning_index",
            F.round((F.col("prep_evac_plan") + F.col("prep_follow_protocols") + F.col("prep_breaker_cutoff")) / 3.0, 3)
        )
    )

    return df


def add_behavioral_segmentation(df: DataFrame) -> DataFrame:
    """Add awareness_level, preparedness_level, gap_typology, and risk_tier classifications."""
    df = (
        df.withColumn(
            "awareness_level",
            F.when(F.col("awareness_score") < 2.5, "Low")
            .when(F.col("awareness_score") <= 3.75, "Moderate")
            .otherwise("High")
        )
        .withColumn(
            "preparedness_level",
            F.when(F.col("preparedness_score") < 2.5, "Low")
            .when(F.col("preparedness_score") <= 3.75, "Moderate")
            .otherwise("High")
        )
        .withColumn(
            "gap_typology",
            F.when(F.abs(F.col("awareness_preparedness_gap")) <= 0.5, "Aligned")
            .when(F.col("awareness_preparedness_gap") > 0.5, "Aware but Unprepared")
            .otherwise("Prepared beyond Awareness")
        )
        .withColumn(
            "risk_tier",
            F.when(
                (F.col("flood_history_n").cast("double") >= 3.0) & (F.col("prep_supplies_index") < 3.0),
                "High Risk"
            )
            .when(
                (F.col("flood_history_n").cast("double") >= 3.0) | (F.col("prep_supplies_index") < 3.0),
                "Moderate Risk"
            )
            .otherwise("Low Risk / Resilient")
        )
    )
    return df


def code_qualitative_themes(df: DataFrame) -> DataFrame:
    """Perform keyword theme coding on open-ended text fields using rlike."""
    # Challenge themes
    patterns_c = {
        "c_supplies_food": r"(?i)food|water|rations|canned|grocer|drink|provision|eat|starv",
        "c_financial_cost": r"(?i)money|budget|fund|financial|cost|expense|expensive|afford|allowance|income",
        "c_evac_access": r"(?i)evacuat|center|route|transport|road|distance|access|crowd|vehicle|stranded",
        "c_power_comms": r"(?i)power|electric|signal|battery|charge|light|brownout|blackout|internet|network|wifi",
        "c_housing_safety": r"(?i)dorm|board|rent|roof|leak|second floor|lowland|drainage|inundat|elevat|landlord|room",
    }

    # Support themes
    patterns_s = {
        "s_go_bag_kits": r"(?i)go-bag|go bag|emergency kit|survival kit|flashlight|whistle|aid kit|supplies",
        "s_early_warning_sms": r"(?i)sms|text|alert|warning|broadcast|siren|bulletin|update|notice|announcement",
        "s_evac_shelter": r"(?i)shelter|center|evacuation area|facility|safe zone|room|building|temporary housing",
        "s_drrm_seminars": r"(?i)seminar|training|workshop|drill|educat|course|lecture|orientat|module|webinar",
        "s_financial_aid": r"(?i)financial|aid|relief|fund|support|allowance|free|calamity|subsidy",
    }

    for col_name, pat in patterns_c.items():
        df = df.withColumn(
            col_name,
            F.when(F.col("challenge_text").rlike(pat), 1).otherwise(0)
        )

    for col_name, pat in patterns_s.items():
        df = df.withColumn(
            col_name,
            F.when(F.col("support_text").rlike(pat), 1).otherwise(0)
        )

    return df


def scale_numerical_features(df: DataFrame) -> DataFrame:
    """Z-standardize the 12 items for PCA dimensionality reduction."""
    assembler = VectorAssembler(inputCols=ALL_12_ITEM_COLS, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features_z", withMean=True, withStd=True)

    pipeline = Pipeline(stages=[assembler, scaler])
    model = pipeline.fit(df)
    df_scaled = model.transform(df)

    # Extract individual z-score columns
    z_array = vector_to_array(F.col("features_z"))
    for idx, c in enumerate(ALL_12_ITEM_COLS):
        df_scaled = df_scaled.withColumn(f"{c}_z", F.round(z_array[idx], 4))

    return df_scaled.drop("features_raw", "features_z")


def run_stage3_transformation() -> None:
    banner("STAGE 3: TOPIC-ALIGNED FEATURE ENGINEERING & ML SCALING")

    spark = create_spark_session("Stage3_Flood_Transformation")
    os.makedirs(TRANSFORMATION_DIR, exist_ok=True)

    log(f"Reading Stage 2 Cleansed Dataset: {STAGE2_UNIFIED_CSV}")
    df_clean = read_csv_spark(spark, STAGE2_UNIFIED_CSV)

    # 1. Composite construct indices
    df_indices = compute_composite_indices(df_clean)

    # 2. Behavioral segmentation & diagnostic categories
    df_segmented = add_behavioral_segmentation(df_indices)

    # 3. NLP qualitative theme coding
    df_themed = code_qualitative_themes(df_segmented)

    # 4. Standardized scaling for PCA
    df_final = scale_numerical_features(df_themed)

    # Export Stage 3 Transformed Unified CSV
    log(f"Writing Stage 3 Transformed Dataset to: {STAGE3_UNIFIED_CSV}")
    written = write_spark_df_to_csv(df_final, STAGE3_UNIFIED_CSV)
    log(f"Successfully saved {written:,} rows to Stage 3 Transformed Dataset.")

    spark.stop()
    banner("STAGE 3 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    run_stage3_transformation()


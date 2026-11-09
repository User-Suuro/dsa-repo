"""
03_data_transformation.py
=========================
RUN ORDER: 3 of 5

Stage 3 — Topic-Aligned Feature Engineering, ML Scaling & NLP Theme Coding (PySpark)
------------------------------------------------------------------------------------
Revolves strictly around `docs/DSA - Proposal.pdf` and
`docs/Data Scalability and Analytics - Data transformation (Pyspark) (1).pdf`.

Reads `output/cleansing/stage2_cleansed_dataset.csv` and performs:
  1. Composite indices from `docs/Data transformation (Pyspark).pdf`:
       * `Awareness_Index = (aware_energy_n + aware_carbon_n) / 2`
       * `Behavior_Index = (consider_impact_n + avoid_use_n) / 2`
       * `Awareness_Behavior_Gap = Awareness_Index - Behavior_Index`
  2. Behavioral & Diagnostic Segmentation from `docs/`:
       * `usage_segment`: "Light (rarely)", "Moderate (weekly)", "Heavy (daily+)"
       * `awareness_level`: "Low", "Medium", "High"
       * `behavior_level`: "Low", "Medium", "High"
       * `gap_type`: "Aligned", "Aware but not acting", "Acting beyond awareness"
  3. Open-Ended Keyword Theme Coding (`rlike`) strictly for the 12 environmental
     concerns (`c_*`) and awareness/responsible-use solutions (`s_*`) in `docs/`:
       * Q7 Concerns: `c_water`, `c_energy`, `c_carbon`, `c_ewaste`, `c_noise_heat`, `c_ecosystem`
       * Q8 Solutions: `s_education`, `s_campaign`, `s_policy`, `s_transparency`, `s_tech`, `s_responsible_use`
       * Plus `concern_len_words`, `awareness_len_words`, `concern_theme_count`, `solution_theme_count`
  4. PySpark ML Pipeline (`StringIndexer`, `OneHotEncoder`, `VectorAssembler`, `StandardScaler`)
     producing z-standardized scores (`*_z`) for PCA reduction.

Outputs -> `output/transformation/`:
  - `stage3_transformed_dataset.csv` (Single unified topic-aligned transformed dataset)
"""

import os
from typing import Dict, List

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, functions as F

from pipeline_utils import (
    HEAVY_USAGE_THRESHOLD,
    MAIN_ITEM_SCORE_COLS,
    MODERATE_USAGE_THRESHOLD,
    Q7_CONCERN_THEME_COLS,
    Q8_SOLUTION_THEME_COLS,
    STAGE2_UNIFIED_CSV,
    STAGE3_UNIFIED_CSV,
    TRANSFORMATION_DIR,
    banner,
    create_spark_session,
    has_kw,
    log,
    read_csv_spark,
    write_spark_df_to_csv,
)

CATEGORICAL_FEATURE_COLS: List[str] = [
    "age_group",
    "usage_segment",
]

# Exact 6 Q7 Environmental Concern patterns (`c_*`) from `docs/`
Q7_CONCERN_PATTERNS: Dict[str, str] = {
    "c_water": r"water|drought|drouth|cool|scarcity|fresh water",
    "c_energy": r"energy|electric|power consum|power usage|watt|grid",
    "c_carbon": r"carbon|emission|greenhouse|global warming|climate|co2|pollut",
    "c_ewaste": r"e-waste|e waste|waste|hardware|mining|resource|material|contaminat",
    "c_noise_heat": r"noise|heat|temperature|loud",
    "c_ecosystem": r"flora|fauna|wildlife|ecosystem|biodiversity|habitat|forest|\bland\b|farm|pax silica|clark|deforest|aeta",
}

# Exact 6 Q8 Suggested Solution patterns (`s_*`) from `docs/`
Q8_SOLUTION_PATTERNS: Dict[str, str] = {
    "s_education": r"educat|school|teach|seminar|literacy|workshop|training|learn|inform|curricul|universit|orientat|course|class",
    "s_campaign": r"campaign|social media|advocacy|\bposts?\b|media|poster|seminar|tiktok|infographic|video|platform|flyer|news",
    "s_policy": r"law|regulat|polic|government|legislat|limit|rules|guideline|dict|ntc",
    "s_transparency": r"transparen|disclos|report|open about|accountab|label|company|companies",
    "s_tech": r"efficien|renewable|clean energy|green|sustainab|solar",
    "s_responsible_use": r"responsib|moderat|avoid|mindful|wisely|proper use|unnecessary|minimalism|own mind|last resort|sparingly|not rely|depend",
}


def engineer_docs_composite_indices_and_segments(df: DataFrame) -> DataFrame:
    """
    Compute `Awareness_Index`, `Behavior_Index`, `Awareness_Behavior_Gap`, and
    the exact categorical segments from `docs/Data transformation (Pyspark).pdf`.
    """
    for c in MAIN_ITEM_SCORE_COLS:
        df = df.withColumn(c, F.col(c).cast("double"))

    df = (
        df.withColumn(
            "Awareness_Index",
            F.round((F.col("aware_energy_n") + F.col("aware_carbon_n")) / 2.0, 4),
        )
        .withColumn(
            "Behavior_Index",
            F.round((F.col("consider_impact_n") + F.col("avoid_use_n")) / 2.0, 4),
        )
        .withColumn(
            "Awareness_Behavior_Gap",
            F.round(F.col("Awareness_Index") - F.col("Behavior_Index"), 4),
        )
        .withColumn(
            "usage_segment",
            F.when(F.col("usage_freq_n") >= HEAVY_USAGE_THRESHOLD, "Heavy (daily+)")
            .when(F.col("usage_freq_n") >= MODERATE_USAGE_THRESHOLD, "Moderate (weekly)")
            .otherwise("Light (rarely)"),
        )
        .withColumn(
            "awareness_level",
            F.when(F.col("Awareness_Index") >= 4.0, "High")
            .when(F.col("Awareness_Index") >= 3.0, "Medium")
            .otherwise("Low"),
        )
        .withColumn(
            "behavior_level",
            F.when(F.col("Behavior_Index") >= 4.0, "High")
            .when(F.col("Behavior_Index") >= 3.0, "Medium")
            .otherwise("Low"),
        )
        .withColumn(
            "gap_type",
            F.when(F.col("Awareness_Behavior_Gap") >= 1.0, "Aware but not acting")
            .when(F.col("Awareness_Behavior_Gap") <= -1.0, "Acting beyond awareness")
            .otherwise("Aligned"),
        )
    )
    log("Engineered Awareness_Index, Behavior_Index, Awareness_Behavior_Gap, usage_segment, awareness_level, behavior_level, gap_type")
    return df


def encode_topic_open_ended_themes(df: DataFrame) -> DataFrame:
    """
    Extract the 6 Q7 environmental concern themes (`c_*`) and 6 Q8 solution
    themes (`s_*`) defined in `docs/Data transformation (Pyspark).pdf`.
    """
    for col_name, pattern in Q7_CONCERN_PATTERNS.items():
        df = df.withColumn(col_name, has_kw("concern_text", pattern))

    for col_name, pattern in Q8_SOLUTION_PATTERNS.items():
        df = df.withColumn(col_name, has_kw("awareness_text", pattern))

    concern_sum_expr = sum(F.col(c) for c in Q7_CONCERN_THEME_COLS)
    solution_sum_expr = sum(F.col(c) for c in Q8_SOLUTION_THEME_COLS)

    df = (
        df.withColumn(
            "concern_len_words",
            F.when(F.col("concern_text").isNull(), 0).otherwise(F.size(F.split(F.col("concern_text"), r"\s+"))),
        )
        .withColumn(
            "awareness_len_words",
            F.when(F.col("awareness_text").isNull(), 0).otherwise(F.size(F.split(F.col("awareness_text"), r"\s+"))),
        )
        .withColumn("concern_theme_count", concern_sum_expr)
        .withColumn("solution_theme_count", solution_sum_expr)
    )
    log("Encoded 6 Q7 environmental concern themes (`c_*`) and 6 Q8 awareness solution themes (`s_*`)")
    return df


def apply_ml_scaling_pipeline(df: DataFrame) -> DataFrame:
    """One-hot encode `age_group` & `usage_segment` and z-score standardize Q2..Q6 numeric scores."""
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in CATEGORICAL_FEATURE_COLS
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")
        for c in CATEGORICAL_FEATURE_COLS
    ]
    assembler = VectorAssembler(
        inputCols=MAIN_ITEM_SCORE_COLS,
        outputCol="numeric_vec",
        handleInvalid="keep",
    )
    scaler = StandardScaler(
        inputCol="numeric_vec",
        outputCol="numeric_scaled",
        withMean=True,
        withStd=True,
    )

    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])
    df_scaled = pipeline.fit(df).transform(df)
    df_scaled = df_scaled.withColumn("scaled_arr", vector_to_array(F.col("numeric_scaled")))

    for idx, c in enumerate(MAIN_ITEM_SCORE_COLS):
        df_scaled = df_scaled.withColumn(f"{c}_z", F.round(F.col("scaled_arr")[idx], 4))

    ohe_cols = [f"{c}_ohe" for c in CATEGORICAL_FEATURE_COLS]
    idx_cols = [f"{c}_idx" for c in CATEGORICAL_FEATURE_COLS]
    log(f"One-hot encoded {CATEGORICAL_FEATURE_COLS}; z-score scaled {MAIN_ITEM_SCORE_COLS}")
    return df_scaled.drop("scaled_arr", "numeric_vec", "numeric_scaled", *ohe_cols, *idx_cols)


def main() -> None:
    if not os.path.exists(STAGE2_UNIFIED_CSV):
        raise FileNotFoundError("Stage 2 output not found. Please run `02_data_cleansing.py` first.")

    os.makedirs(TRANSFORMATION_DIR, exist_ok=True)
    spark = create_spark_session("DSA-Final-03-Topic-Aligned-Transformation")
    banner("STAGE 3: TOPIC-ALIGNED TRANSFORMATION (INDICES, GAP TYPOLOGY, ENV THEMES, ML SCALING)")

    df = read_csv_spark(spark, STAGE2_UNIFIED_CSV)
    df = engineer_docs_composite_indices_and_segments(df)
    df = encode_topic_open_ended_themes(df)
    df = apply_ml_scaling_pipeline(df)

    write_spark_df_to_csv(df, STAGE3_UNIFIED_CSV)

    spark.stop()
    log("Stage 3 (Topic-Aligned Data Transformation) finished successfully.")


if __name__ == "__main__":
    main()

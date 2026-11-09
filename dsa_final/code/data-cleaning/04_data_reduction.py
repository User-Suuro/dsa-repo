"""
04_data_reduction.py
====================
RUN ORDER: 4 of 5

Stage 4 — Topic-Aligned Data Reduction (PySpark)
------------------------------------------------
Revolves strictly around `docs/DSA - Proposal.pdf` and
`docs/Data Scalability and Analytics - Data transformation (Pyspark) (1).pdf`.

Reads `output/transformation/stage3_transformed_dataset.csv` and performs:
  1. Zero-Variance Feature Pruning (`F.countDistinct(c) <= 1`)
  2. Dimensionality Reduction via Principal Component Analysis (`PCA(k=3)`)
     on the 5 z-standardized `main_dataset.csv` items (`usage_freq_n_z`,
     `aware_energy_n_z`, `aware_carbon_n_z`, `consider_impact_n_z`, `avoid_use_n_z`
     -> `pca_1`, `pca_2`, `pca_3`)
  3. Numerosity Reduction via 50% Stratified Sampling (`sampleBy("usage_segment", seed=42)`)

Outputs -> `output/reduction/`:
  - `integrated_master_dataset.csv` (Final unified topic-aligned master dataset)
  - `integrated_master_dataset_sampled50.csv` (50% stratified sample)
"""

import os
from typing import List, Sequence

from pyspark.ml.feature import PCA, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame, functions as F

from pipeline_utils import (
    ALL_TOPIC_THEME_COLS,
    CORE_INDEX_COLS,
    MAIN_ITEM_SCORE_COLS,
    MASTER_UNIFIED_CSV,
    PCA_COMPONENTS,
    RANDOM_SEED,
    REDUCTION_DIR,
    SAMPLED_UNIFIED_CSV,
    STAGE3_UNIFIED_CSV,
    STRATIFIED_SAMPLE_FRACTION,
    banner,
    create_spark_session,
    log,
    read_csv_spark,
    write_spark_df_to_csv,
)

Z_SCORE_COLS: List[str] = [f"{c}_z" for c in MAIN_ITEM_SCORE_COLS]


def cast_numeric_columns(df: DataFrame) -> DataFrame:
    """Cast item scores, composite indices, and z-score columns to DoubleType."""
    num_cols = set(MAIN_ITEM_SCORE_COLS + CORE_INDEX_COLS + Z_SCORE_COLS)
    for c in num_cols:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast("double"))
    return df


def prune_zero_variance_columns(df: DataFrame, candidate_cols: Sequence[str]) -> DataFrame:
    """Identify and drop any candidate columns with `<= 1` distinct value."""
    distinct_counts = (
        df.select([F.countDistinct(F.col(c)).alias(c) for c in candidate_cols if c in df.columns])
        .collect()[0]
        .asDict()
    )
    zero_var_cols = [c for c, cnt in distinct_counts.items() if cnt <= 1]
    log(f"Zero-variance columns identified & dropped: {zero_var_cols if zero_var_cols else 'None'}")
    return df.drop(*zero_var_cols) if zero_var_cols else df


def run_pca_reduction(df: DataFrame) -> DataFrame:
    """Fit a 3-component PCA model on the 5 z-standardized Q2..Q6 columns."""
    assembler = VectorAssembler(
        inputCols=Z_SCORE_COLS,
        outputCol="numeric_scaled_vec",
        handleInvalid="keep",
    )
    df_vec = assembler.transform(df)

    pca_model = PCA(
        k=PCA_COMPONENTS,
        inputCol="numeric_scaled_vec",
        outputCol="numeric_pca",
    ).fit(df_vec)
    explained = [round(float(v) * 100, 2) for v in pca_model.explainedVariance]
    log(
        f"PCA(k={PCA_COMPONENTS}) on Q2..Q6 explained variance (%): {explained} "
        f"(Cumulative = {round(sum(explained), 2)}%)"
    )

    df_pca = pca_model.transform(df_vec).withColumn("pca_arr", vector_to_array(F.col("numeric_pca")))
    for i in range(PCA_COMPONENTS):
        df_pca = df_pca.withColumn(f"pca_{i + 1}", F.round(F.col("pca_arr")[i], 4))
    return df_pca.drop("numeric_scaled_vec", "numeric_pca", "pca_arr")


def create_stratified_sample(df: DataFrame, fraction: float) -> DataFrame:
    """Create a reproducible 50% stratified sample across `usage_segment` strata (matching `docs/`)."""
    strata = [r[0] for r in df.select("usage_segment").distinct().collect()]
    fractions = {s: fraction for s in strata}
    return df.sampleBy("usage_segment", fractions=fractions, seed=RANDOM_SEED)


def main() -> None:
    if not os.path.exists(STAGE3_UNIFIED_CSV):
        raise FileNotFoundError("Stage 3 output not found. Please run `03_data_transformation.py` first.")

    os.makedirs(REDUCTION_DIR, exist_ok=True)
    spark = create_spark_session("DSA-Final-04-Topic-Aligned-Reduction")
    banner("STAGE 4: TOPIC-ALIGNED DATA REDUCTION (ZERO-VARIANCE PRUNING, PCA(k=3), SAMPLING)")

    df = cast_numeric_columns(read_csv_spark(spark, STAGE3_UNIFIED_CSV))
    candidate_cols = [
        "source_study",
        "study_role",
        "age_group",
        "ai_usage_freq",
        "usage_segment",
        "awareness_level",
        "behavior_level",
        "gap_type",
    ] + MAIN_ITEM_SCORE_COLS + ALL_TOPIC_THEME_COLS
    df = prune_zero_variance_columns(df, candidate_cols)
    df_pca = run_pca_reduction(df)

    write_spark_df_to_csv(df_pca, MASTER_UNIFIED_CSV)

    df_sampled = create_stratified_sample(df_pca, STRATIFIED_SAMPLE_FRACTION)
    write_spark_df_to_csv(df_sampled, SAMPLED_UNIFIED_CSV)

    spark.stop()
    log("Stage 4 (Topic-Aligned Data Reduction) finished successfully.")


if __name__ == "__main__":
    main()

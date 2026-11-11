"""
pipeline_utils.py
=================
Shared utilities, SparkSession factory, directory paths, domain constants,
data cleaning rules, and Windows-safe CSV writers for the 5-stage DSA Final
PySpark Multi-Source Pipeline in `dsa_final_jm`.

Main Study:
  "Assessing College Students Awareness and Preparedness Toward Flood Risks"
  Authors: John Mark Pardo, Byron Boral, Aerold Steven Reyes (Group 11 - Skywalker)
  Institution: College Students, Camarines Norte, Philippines

Integrated Public Studies:
  - Public S2: Adrian M. Segovia & Joy R. Tungol (2026), Centro Escolar University,
    "Anticipatory mobility, psychological distress, and flood exposure among young adults
    in seven flood-prone communities in Metro Manila and Central Luzon, Philippines"
    (Mendeley Data, DOI: 10.17632/462nvdfvjj.1)
  - Public S3: NOAA / IPPRA University of Oklahoma (2024),
    "The 2024 Flooding and Society Survey (FL24)"
    (Harvard Dataverse, DOI: 10.7910/DVN/I9751R)
  - Public S4: Majah-Leah V. Ravago, Claire Dennis S. Mapa, et al. (2020),
    "Data from a survey of the Philippines' local governments on their risk management
    strategies to natural disasters" (Data in Brief / Mendeley Data: 10.17632/9y2yhw45zt.1)

Canonical 12 Likert Items:
  Part I - Flood Risk Awareness:
    * aw_warning_signals   [1. Understand rainfall & flood warning signals / alerts]
    * aw_monitor_updates   [2. Regularly monitor official weather & disaster updates]
    * aw_residence_risk    [3. Aware of flood history & risk level in residence area]
    * aw_emergency_contacts[4. Know emergency contact numbers of disaster response units]
    * aw_evac_routes       [5. Know locations of evacuation centers & safe routes]
    * aw_health_risks      [6. Aware of waterborne diseases & flood health hazards]
  Part II - Personal Flood Preparedness:
    * prep_go_bag          [1. Packed emergency Go-Bag with essentials]
    * prep_food_water      [2. Maintain 3-day supply of food and drinking water]
    * prep_secure_docs     [3. Secure personal documents in waterproof storage]
    * prep_evac_plan       [4. Clear emergency communication & evacuation plan]
    * prep_follow_protocols[5. Actively follow official disaster protocols]
    * prep_breaker_cutoff  [6. Safety protocols for electrical main breaker cutoff]
"""

import csv
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Sequence

import pyspark
from pyspark.sql import DataFrame, SparkSession, functions as F, types as T

# Configure Windows paths and workers
os.environ["SPARK_HOME"] = os.path.dirname(pyspark.__file__)
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ==============================================================================
# DIRECTORY & FILE PATH CONSTANTS
# ==============================================================================
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CODE_DIR, "..", ".."))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
PUBLIC_DATASET_DIR = os.path.join(DATASET_DIR, "public")

OUTPUT_ROOT = os.path.join(PROJECT_DIR, "output", "data-cleaning")
INTEGRATION_DIR = os.path.join(OUTPUT_ROOT, "integration")
STAGING_DIR = os.path.join(INTEGRATION_DIR, "staging")
CLEANSING_DIR = os.path.join(OUTPUT_ROOT, "cleansing")
TRANSFORMATION_DIR = os.path.join(OUTPUT_ROOT, "transformation")
REDUCTION_DIR = os.path.join(OUTPUT_ROOT, "reduction")
ANALYSIS_DIR = os.path.join(OUTPUT_ROOT, "analysis")
FIG_DIR = os.path.join(ANALYSIS_DIR, "figures")

# Unified dataset handoff paths across the 5 stages
STAGE1_UNIFIED_CSV = os.path.join(INTEGRATION_DIR, "stage1_integrated_dataset.csv")
STAGE1_LOST_SUMMARY_CSV = os.path.join(INTEGRATION_DIR, "stage1_lost_rows_by_study.csv")
STAGE1_LOST_REASONS_CSV = os.path.join(INTEGRATION_DIR, "stage1_lost_rows_reasons_detail.csv")

STAGE2_UNIFIED_CSV = os.path.join(CLEANSING_DIR, "stage2_cleansed_dataset.csv")
STAGE2_CLEANING_LOG_CSV = os.path.join(CLEANSING_DIR, "stage2_cleaning_loss_log.csv")

STAGE3_UNIFIED_CSV = os.path.join(TRANSFORMATION_DIR, "stage3_transformed_dataset.csv")

MASTER_UNIFIED_CSV = os.path.join(REDUCTION_DIR, "integrated_master_dataset.csv")
SAMPLED_UNIFIED_CSV = os.path.join(REDUCTION_DIR, "integrated_master_dataset_sampled50.csv")

ANALYSIS_LOST_ROWS_CSV = os.path.join(ANALYSIS_DIR, "study_rows_lost_and_reasons.csv")

# Input files
MAIN_STUDY_CSV = os.path.join(DATASET_DIR, "Assessing College Students Awareness and Preparedness Toward Flood Risks.csv")
SEGOVIA_DIR = os.path.join(PUBLIC_DATASET_DIR, "Segovia_2026_PH_Young_Adults_Flood_Exposure")
SEGOVIA_SCREENED_CSV = os.path.join(SEGOVIA_DIR, "data_screened_541.csv")
SEGOVIA_ANALYTIC_CSV = os.path.join(SEGOVIA_DIR, "data_analytic_370.csv")
FL24_DIR = os.path.join(PUBLIC_DATASET_DIR, "FL24_Flooding_and_Society_Survey")
FL24_CSV = os.path.join(FL24_DIR, "FL24_data_wtd.csv")
RAVAGO_DIR = os.path.join(PUBLIC_DATASET_DIR, "Supplementary Appendix B DIB Ravago et al 2020")
RAVAGO_CSV = os.path.join(RAVAGO_DIR, "Supplementary Appendix A1 DIB Ravago et al 2020.csv")

# ==============================================================================
# DOMAIN & PIPELINE CONSTANTS
# ==============================================================================
LIKERT_MIN = 1.0
LIKERT_MAX = 5.0
LIKERT_NEUTRAL = 3.0

PCA_COMPONENTS = 3
STRATIFIED_SAMPLE_FRACTION = 0.50
RANDOM_SEED = 42

# The 6 Awareness items
AWARENESS_ITEM_COLS: List[str] = [
    "aw_warning_signals",
    "aw_monitor_updates",
    "aw_residence_risk",
    "aw_emergency_contacts",
    "aw_evac_routes",
    "aw_health_risks",
]

# The 6 Preparedness items
PREPAREDNESS_ITEM_COLS: List[str] = [
    "prep_go_bag",
    "prep_food_water",
    "prep_secure_docs",
    "prep_evac_plan",
    "prep_follow_protocols",
    "prep_breaker_cutoff",
]

ALL_12_ITEM_COLS: List[str] = AWARENESS_ITEM_COLS + PREPAREDNESS_ITEM_COLS

# Derived Composite Index columns
CORE_INDEX_COLS: List[str] = [
    "awareness_score",
    "preparedness_score",
    "awareness_preparedness_gap",
    "aw_warning_index",
    "aw_local_hazard_index",
    "aw_safety_health_index",
    "prep_supplies_index",
    "prep_planning_index",
]

# Qualitative Open-Ended Theme binary columns (rlike keyword coding)
# Challenge Themes (c_*)
CHALLENGE_THEME_COLS: List[str] = [
    "c_supplies_food",
    "c_financial_cost",
    "c_evac_access",
    "c_power_comms",
    "c_housing_safety",
]

# Support Themes (s_*)
SUPPORT_THEME_COLS: List[str] = [
    "s_go_bag_kits",
    "s_early_warning_sms",
    "s_evac_shelter",
    "s_drrm_seminars",
    "s_financial_aid",
]

ALL_QUALITATIVE_THEME_COLS: List[str] = CHALLENGE_THEME_COLS + SUPPORT_THEME_COLS

JUNK_TEXT_TOKENS: List[str] = [
    "n/a", "na", "none", "no", "nothing", "nil", "unsure",
    "not sure", "-", ".", "?", "idk", "null", "nan", "nor neccesary",
]

# Canonical Schema for Stage 1 Unified Dataset
UNIFIED_CANONICAL_SCHEMA: List[str] = [
    "respondent_id",
    "source_study",
    "study_role",
    "age",
    "age_group",
    "sex",
    "year_level",
    "course_discipline",
    "living_arrangement",
    "municipality_residence",
    "flood_history",
    "flood_history_n",
    "aw_warning_signals",
    "aw_monitor_updates",
    "aw_residence_risk",
    "aw_emergency_contacts",
    "aw_evac_routes",
    "aw_health_risks",
    "prep_go_bag",
    "prep_food_water",
    "prep_secure_docs",
    "prep_evac_plan",
    "prep_follow_protocols",
    "prep_breaker_cutoff",
    "challenge_text",
    "support_text",
    "is_valid_record",
    "invalid_reason",
]


# ==============================================================================
# LOGGING & CONSOLE FORMATTING
# ==============================================================================
def banner(title: str) -> None:
    """Print a formatted stage section header."""
    line = "=" * 90
    print(f"\n{line}\n  {title}\n{line}")


def log(msg: str) -> None:
    """Print an indented pipeline status line."""
    print(f"  [Flood-Pipeline] {msg}")


# ==============================================================================
# SPARK SESSION INITIALIZATION
# ==============================================================================
def create_spark_session(app_name: str, shuffle_partitions: int = 8) -> SparkSession:
    """
    Create and configure a Windows-compatible PySpark session with Adaptive
    Query Execution (AQE) enabled.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.driver.memory", "4g")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ==============================================================================
# COLUMN SANITIZATION & SPARK I/O HELPERS
# ==============================================================================
def sanitize_col_name(name: str) -> str:
    """Convert raw header strings into Spark-SQL-safe alphanumeric identifiers."""
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", str(name).strip()).strip("_")
    clean = re.sub(r"_+", "_", clean)
    return clean if clean else "col_unnamed"


def sanitize_df_columns(df: DataFrame) -> DataFrame:
    """Rename all columns in a DataFrame to sanitized unique identifiers."""
    seen: Dict[str, int] = {}
    for c in df.columns:
        base = sanitize_col_name(c)
        cnt = seen.get(base, 0)
        seen[base] = cnt + 1
        final_name = base if cnt == 0 else f"{base}_{cnt}"
        if final_name != c:
            df = df.withColumnRenamed(c, final_name)
    return df


def read_csv_spark(spark: SparkSession, path: str, infer_schema: bool = False) -> DataFrame:
    """Read a UTF-8 CSV file into a Spark DataFrame with multi-line quote support."""
    return (
        spark.read.option("header", "true")
        .option("inferSchema", "true" if infer_schema else "false")
        .option("multiLine", "true")
        .option("quote", '"')
        .option("escape", '"')
        .option("encoding", "UTF-8")
        .csv(path)
    )


def write_spark_df_to_csv(
    df: DataFrame,
    out_path: str,
    exclude_cols: Optional[Iterable[str]] = None,
) -> int:
    """
    Stream a Spark DataFrame to a single UTF-8-SIG CSV file via `toLocalIterator()`.
    Bypasses Hadoop `winutils.exe` on Windows while keeping driver memory bounded.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    exclude = set(exclude_cols or [])
    scalar_cols = [
        f.name
        for f in df.schema.fields
        if f.name not in exclude
        and not isinstance(f.dataType, (T.ArrayType, T.MapType, T.StructType))
    ]

    selected_df = df.select(scalar_cols)
    field_names = [f.name for f in selected_df.schema.fields]

    count = 0
    with open(out_path, mode="w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(field_names)
        for row in selected_df.toLocalIterator():
            row_dict = row.asDict()
            writer.writerow([row_dict.get(c) for c in field_names])
            count += 1

    return count


def clamp_likert(col_name: str) -> F.Column:
    """Clamp numeric score to [LIKERT_MIN, LIKERT_MAX]."""
    c = F.col(col_name).cast("double")
    return (
        F.when(c.isNull(), None)
        .when(c < LIKERT_MIN, F.lit(LIKERT_MIN))
        .when(c > LIKERT_MAX, F.lit(LIKERT_MAX))
        .otherwise(F.round(c, 3))
    )


def score_to_likert_label(col_name: str) -> F.Column:
    """Convert a 1..5 numeric Likert score into a standardized qualitative label."""
    c = F.col(col_name).cast("double")
    return (
        F.when(c.isNull(), F.lit(None))
        .when(c < 1.5, F.lit("Strongly Disagree"))
        .when(c < 2.5, F.lit("Disagree"))
        .when(c < 3.5, F.lit("Neutral"))
        .when(c < 4.5, F.lit("Agree"))
        .otherwise(F.lit("Strongly Agree"))
    )


def row_mean_null_safe(cols: Sequence[str]) -> F.Column:
    """Calculate the row-wise mean across provided numeric columns, ignoring NULLs."""
    valid_terms = [F.when(F.col(c).isNotNull(), F.col(c)).otherwise(F.lit(0.0)) for c in cols]
    valid_counts = [F.when(F.col(c).isNotNull(), F.lit(1.0)).otherwise(F.lit(0.0)) for c in cols]

    sum_col = valid_terms[0]
    count_col = valid_counts[0]
    for term, cnt in zip(valid_terms[1:], valid_counts[1:]):
        sum_col = sum_col + term
        count_col = count_col + cnt

    return F.when(count_col > 0.0, F.round(sum_col / count_col, 3)).otherwise(F.lit(None))


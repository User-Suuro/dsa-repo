"""
pipeline_utils.py
=================
Shared utilities, SparkSession factory, directory paths, domain constants,
zero-dependency XLSX/CSV staging loaders, and Windows-safe CSV writers for the
5-stage DSA Final PySpark Pipeline.

Revolves strictly around our Main Study Topic (see `docs/DSA - Proposal.pdf` and
`docs/Data Scalability and Analytics - Data transformation (Pyspark) (1).pdf`):
  "AI and the Environment: An Analysis of Public Awareness"
  Authors: Renz Dominic Azul, Gabriel de la Cruz, Godwin Reigh Galvez (BSIT 3B)

Canonical Schema Basis (`main_dataset.csv` & `docs/`):
  - `respondent_id`, `source_study`, `study_role`
  - Part I (Respondent Profile):
      * `age_group` ("1. What is your age?": 18-24, 25-34, 35-44, 45-54, 55 and above)
      * `ai_usage_freq` & `usage_freq_n` ("2. How often do you use AI tools?": 1=Rarely .. 5=Several times a day)
  - Part II (Environmental Awareness):
      * `aware_energy` & `aware_energy_n` ("3. Aware using AI consumes significant electricity/energy")
      * `aware_carbon` & `aware_carbon_n` ("4. Aware AI data centers contribute to carbon emissions & env impacts")
  - Part III (Responsible AI Usage):
      * `consider_impact` & `consider_impact_n` ("5. Consider environmental impact when deciding how to use AI")
      * `avoid_unnecessary_use` & `avoid_use_n` ("6. Avoid unnecessary use of AI when task can be completed without it")
  - Part IV (Open-Ended Qualitative Questions):
      * `concern_text` ("7. What environmental impact of AI concerns you the most, and why?")
      * `awareness_text` ("8. What should be done to increase public awareness and encourage responsible use of AI?")
"""

import csv
import os
import re
import shutil
import sys
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
import pyspark
from pyspark.sql import DataFrame, SparkSession, functions as F, types as T

# Ensure Windows worker subprocesses and SPARK_HOME resolve cleanly even when
# the workspace path contains spaces (e.g., "C:\Coding Projects\Python Projects").
os.environ["SPARK_HOME"] = os.path.dirname(pyspark.__file__)
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ==============================================================================
# DIRECTORY & FILE PATH CONSTANTS (SINGLE UNIFIED DATASET PIPELINE)
# ==============================================================================
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CODE_DIR, "..", ".."))
OUTPUT_ROOT = os.path.join(PROJECT_DIR, "output", "data-cleaning")
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
LIVERPOOL_DIR = os.path.join(
    DATASET_DIR,
    "Changing Patterns of Generative AI Use, Acceptable Practice and Institutional Guidance in Higher Education",
)

INTEGRATION_DIR = os.path.join(OUTPUT_ROOT, "integration")
STAGING_DIR = os.path.join(INTEGRATION_DIR, "staging")
CLEANSING_DIR = os.path.join(OUTPUT_ROOT, "cleansing")
TRANSFORMATION_DIR = os.path.join(OUTPUT_ROOT, "transformation")
REDUCTION_DIR = os.path.join(OUTPUT_ROOT, "reduction")
ANALYSIS_DIR = os.path.join(OUTPUT_ROOT, "analysis")
FIG_DIR = os.path.join(ANALYSIS_DIR, "figures")

# Single unified dataset handoff paths across the 5 stages
STAGE1_UNIFIED_CSV = os.path.join(INTEGRATION_DIR, "stage1_integrated_dataset.csv")
STAGE1_LOST_SUMMARY_CSV = os.path.join(INTEGRATION_DIR, "stage1_lost_rows_by_study.csv")
STAGE1_LOST_REASONS_CSV = os.path.join(INTEGRATION_DIR, "stage1_lost_rows_reasons_detail.csv")

STAGE2_UNIFIED_CSV = os.path.join(CLEANSING_DIR, "stage2_cleansed_dataset.csv")
STAGE2_CLEANING_LOG_CSV = os.path.join(CLEANSING_DIR, "stage2_cleaning_loss_log.csv")

STAGE3_UNIFIED_CSV = os.path.join(TRANSFORMATION_DIR, "stage3_transformed_dataset.csv")

MASTER_UNIFIED_CSV = os.path.join(REDUCTION_DIR, "integrated_master_dataset.csv")
SAMPLED_UNIFIED_CSV = os.path.join(REDUCTION_DIR, "integrated_master_dataset_sampled50.csv")

ANALYSIS_LOST_ROWS_CSV = os.path.join(ANALYSIS_DIR, "study_rows_lost_and_reasons.csv")

# ==============================================================================
# DOMAIN & PIPELINE CONSTANTS (Strictly Aligned to `docs/`)
# ==============================================================================
LIKERT_MIN = 1.0
LIKERT_MAX = 5.0
LIKERT_NEUTRAL = 3.0
REVERSE_5PT_OFFSET = 6.0

HEAVY_USAGE_THRESHOLD = 3.5
MODERATE_USAGE_THRESHOLD = 2.0

PCA_COMPONENTS = 3
STRATIFIED_SAMPLE_FRACTION = 0.50
RANDOM_SEED = 42

# The 5 core quantitative items from `docs/Data transformation (Pyspark).pdf` (Q2..Q6)
MAIN_ITEM_SCORE_COLS: List[str] = [
    "usage_freq_n",
    "aware_energy_n",
    "aware_carbon_n",
    "consider_impact_n",
    "avoid_use_n",
]

# Composite construct indices derived directly in `docs/Data transformation (Pyspark).pdf`
CORE_INDEX_COLS: List[str] = [
    "usage_freq_n",
    "Awareness_Index",
    "Behavior_Index",
    "Awareness_Behavior_Gap",
]

# 6 Q7 Environmental Concern binary themes (`c_*`) from `docs/`
Q7_CONCERN_THEME_COLS: List[str] = [
    "c_water",
    "c_energy",
    "c_carbon",
    "c_ewaste",
    "c_noise_heat",
    "c_ecosystem",
]

# 6 Q8 Suggested Solution binary themes (`s_*`) from `docs/`
Q8_SOLUTION_THEME_COLS: List[str] = [
    "s_education",
    "s_campaign",
    "s_policy",
    "s_transparency",
    "s_tech",
    "s_responsible_use",
]

ALL_TOPIC_THEME_COLS: List[str] = Q7_CONCERN_THEME_COLS + Q8_SOLUTION_THEME_COLS

JUNK_TEXT_TOKENS: List[str] = [
    "n/a", "na", "none", "no", "nothing", "nil", "unsure",
    "not sure", "-", ".", "?", "idk", "null", "nan",
]

# Regex pattern to verify whether an open-ended response revolves around our main topic:
# AI environmental impacts (water, energy, carbon, e-waste, heat/noise, land/ecosystem)
# OR responsible AI usage & public awareness solutions.
MAIN_TOPIC_ENV_PATTERN: str = (
    r"water|drought|drouth|cool|scarcity|energy|electric|power|watt|grid|"
    r"carbon|emission|greenhouse|global warming|climate|co2|pollut|"
    r"e-waste|e waste|waste|hardware|mining|resource|material|contaminat|"
    r"noise|heat|temperature|loud|flora|fauna|wildlife|ecosystem|biodiversity|"
    r"habitat|forest|\bland\b|farm|pax silica|clark|deforest|aeta|"
    r"environment|nature|earth|planet|data cent|server|sustainab|ecological|green"
)

MAIN_TOPIC_RESPONSIBLE_USE_PATTERN: str = (
    r"unnecessary|last resort|sparingly|overuse|rely|depend|own brain|own mind|"
    r"think for myself|minimalism|responsib|mindful|wisely|proper use|limit|"
    r"educat|school|seminar|campaign|social media|transparen|polic|guid|regulat|aware"
)


# ==============================================================================
# LOGGING & CONSOLE FORMATTING
# ==============================================================================
def banner(title: str) -> None:
    """Print a formatted stage section header."""
    line = "=" * 90
    print(f"\n{line}\n  {title}\n{line}")


def log(msg: str) -> None:
    """Print an indented pipeline status line."""
    print(f"  [DSA-Pipeline] {msg}")


# ==============================================================================
# SPARK SESSION INITIALIZATION
# ==============================================================================
def create_spark_session(app_name: str, shuffle_partitions: int = 8) -> SparkSession:
    """
    Create and configure a Windows-compatible PySpark session with Adaptive
    Query Execution (AQE) enabled for optimal partition coalescing and join skew.
    """
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
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
        and "Vector" not in type(f.dataType).__name__
    ]
    projected = df.select(*scalar_cols)
    row_count = 0
    with open(out_path, mode="w", newline="", encoding="utf-8-sig") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(scalar_cols)
        for row in projected.toLocalIterator():
            writer.writerow(["" if val is None else val for val in row])
            row_count += 1
    log(f"Saved CSV: {os.path.relpath(out_path, OUTPUT_ROOT)} ({row_count:,} rows x {len(scalar_cols)} cols)")
    return row_count


def spark_to_pandas(df: DataFrame, cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Convert scalar columns of a Spark DataFrame into a Pandas DataFrame safely on Windows."""
    selected = df.select(*cols) if cols else df
    col_names = selected.columns
    rows = [list(r) for r in selected.collect()]
    return pd.DataFrame(rows, columns=col_names)


# ==============================================================================
# ZERO-DEPENDENCY XLSX & LEADING-UNDERSCORE CSV STAGING
# ==============================================================================
def _col_letters_to_index(letters: str) -> int:
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1


def xlsx_sheet_to_csv(xlsx_path: str, sheet_name: str, out_csv_path: str) -> int:
    """
    Parse an Excel `.xlsx` worksheet via Python's built-in `zipfile` + `ElementTree`
    (no `openpyxl` required) and write a sanitized UTF-8 staging CSV for JVM Spark ingestion.
    """
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    ns_main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ns_rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

    with zipfile.ZipFile(xlsx_path, "r") as zf:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            ss_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in ss_root.findall(f"{{{ns_main}}}si"):
                texts = [t.text or "" for t in si.iter(f"{{{ns_main}}}t")]
                shared_strings.append("".join(texts))

        wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
        rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels_root}

        target_path = None
        for sh in wb_root.findall(f".//{{{ns_main}}}sheet"):
            name = sh.attrib.get("name", "")
            if name == sheet_name or name.strip() == sheet_name.strip():
                r_id = sh.attrib.get(f"{{{ns_rel}}}id")
                target = rel_map.get(r_id, "")
                target_path = target.lstrip("/") if target.startswith("/") else f"xl/{target}"
                break

        if not target_path or target_path not in zf.namelist():
            raise ValueError(f"Worksheet '{sheet_name}' not found in '{xlsx_path}'")

        sheet_root = ET.fromstring(zf.read(target_path))
        rows_data: List[List[str]] = []
        max_col = 0

        for row_el in sheet_root.findall(f".//{{{ns_main}}}sheetData/{{{ns_main}}}row"):
            row_dict: Dict[int, str] = {}
            for cell in row_el.findall(f"{{{ns_main}}}c"):
                ref = cell.attrib.get("r", "")
                col_letters = "".join(ch for ch in ref if ch.isalpha())
                col_idx = _col_letters_to_index(col_letters) if col_letters else len(row_dict)
                c_type = cell.attrib.get("t", "")
                v_el = cell.find(f"{{{ns_main}}}v")
                is_el = cell.find(f"{{{ns_main}}}is")

                val = ""
                if c_type == "s" and v_el is not None and v_el.text is not None:
                    s_idx = int(v_el.text)
                    val = shared_strings[s_idx] if 0 <= s_idx < len(shared_strings) else ""
                elif c_type == "inlineStr" and is_el is not None:
                    val = "".join(t.text or "" for t in is_el.iter(f"{{{ns_main}}}t"))
                elif v_el is not None and v_el.text is not None:
                    val = v_el.text

                row_dict[col_idx] = val.strip() if isinstance(val, str) else str(val)
                if col_idx + 1 > max_col:
                    max_col = col_idx + 1

            if row_dict:
                rows_data.append([row_dict.get(i, "") for i in range(max_col)])

    if not rows_data:
        raise ValueError(f"Worksheet '{sheet_name}' in '{xlsx_path}' is empty.")

    raw_headers = rows_data[0]
    seen_headers: Dict[str, int] = {}
    headers: List[str] = []
    for idx, h in enumerate(raw_headers):
        base = sanitize_col_name(h if h else f"col_{idx}")
        cnt = seen_headers.get(base, 0)
        seen_headers[base] = cnt + 1
        headers.append(base if cnt == 0 else f"{base}_{cnt}")

    num_cols = len(headers)
    data_count = 0
    with open(out_csv_path, mode="w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(headers)
        for r in rows_data[1:]:
            padded = (r + [""] * num_cols)[:num_cols]
            if any(cell != "" for cell in padded):
                writer.writerow(padded)
                data_count += 1
    return data_count


def stage_underscore_csv(csv_path: str, staged_csv_path: str) -> None:
    """Copy a CSV whose filename starts with `_` to a clean staging filename for Hadoop."""
    os.makedirs(os.path.dirname(staged_csv_path), exist_ok=True)
    shutil.copyfile(csv_path, staged_csv_path)


# ==============================================================================
# SHARED PYSPARK COLUMN EXPRESSION BUILDERS (Basis: `docs/` & `main_dataset.csv`)
# ==============================================================================
def row_mean_null_safe(col_names: Sequence[str]) -> F.Column:
    """Compute row-wise mean across non-null columns in PySpark, rounded to 4 decimals."""
    valid_sum = F.lit(0.0)
    valid_cnt = F.lit(0.0)
    for c in col_names:
        num_c = F.col(c).cast("double")
        valid_sum = valid_sum + F.coalesce(num_c, F.lit(0.0))
        valid_cnt = valid_cnt + F.when(num_c.isNotNull(), F.lit(1.0)).otherwise(F.lit(0.0))
    return F.when(valid_cnt > 0, F.round(valid_sum / valid_cnt, 4)).otherwise(
        F.lit(None).cast("double")
    )


def clamp_likert(col_expr: F.Column) -> F.Column:
    """Clamp a numeric column expression strictly within `[1.0, 5.0]`."""
    return F.when(
        col_expr.isNotNull(),
        F.round(
            F.least(F.lit(LIKERT_MAX), F.greatest(F.lit(LIKERT_MIN), col_expr.cast("double"))),
            4,
        ),
    ).otherwise(F.lit(None).cast("double"))


def score_to_likert_label(score_col: str) -> F.Column:
    """Convert a `[1.0, 5.0]` numeric score to `main_dataset.csv`'s 5-point Likert label."""
    s = F.col(score_col).cast("double")
    return (
        F.when(s.isNull(), F.lit(None).cast("string"))
        .when(s < 1.5, F.lit("Strongly Disagree"))
        .when(s < 2.5, F.lit("Disagree"))
        .when(s < 3.5, F.lit("Neutral"))
        .when(s < 4.5, F.lit("Agree"))
        .otherwise(F.lit("Strongly Agree"))
    )


def score_to_frequency_label(score_col: str) -> F.Column:
    """Convert a `[1.0, 5.0]` usage score to `main_dataset.csv`'s Q2 usage frequency label."""
    s = F.col(score_col).cast("double")
    return (
        F.when(s.isNull(), F.lit(None).cast("string"))
        .when(s < 1.5, F.lit("Rarely"))
        .when(s < 2.5, F.lit("Once a week"))
        .when(s < 3.5, F.lit("Several times a week"))
        .when(s < 4.5, F.lit("Once a day"))
        .otherwise(F.lit("Several times a day"))
    )


def has_kw(col_name: str, *keywords: str) -> F.Column:
    """Return 1 if lower-cased text column matches any regex keyword pattern, else 0."""
    pattern = "(" + "|".join(keywords) + ")"
    return F.when(
        F.lower(F.coalesce(F.col(col_name), F.lit(""))).rlike(pattern),
        F.lit(1),
    ).otherwise(F.lit(0))

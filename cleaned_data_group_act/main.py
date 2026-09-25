"""
PySpark Data Preprocessing Pipeline
Dataset context: survey data on consumer perceptions of AI/LLM energy use
(data_analysis.csv) integrated with an LLM-based text-classification of the
open-ended "mitigation strategy" question MET1 (data_analysis_strategies.csv),
as used in the ink.springer.com/article/10.1007/s12525-026-00908-w study.

Covers the four classic preprocessing stages:
  1. Data Integration   - combine the two sources into one analytic dataset
  2. Data Cleansing     - dedupe, trim, fix types, validate ranges, fill nulls
  3. Data Transformation - recode, bucket, scale, one-hot encode, feature engineer
  4. Data Reduction      - drop zero-variance/redundant columns, collapse the
                           40 LLM label columns into compact consensus features,
                           and apply PCA to shrink the numeric feature space
"""

import numpy as np
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, PCA, Imputer
)
from pyspark.ml import Pipeline

def banner(title):
    """Print a clearly delimited section header so pipeline stages are easy
    to follow on the console (kept separate from Spark's own log lines)."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

def log(msg):
    print(f"  - {msg}")

spark = (
    SparkSession.builder
    .appName("AI-Energy-Survey-Preprocessing")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
# ERROR only: WARN-level Spark/Hadoop chatter otherwise interleaves with
# our own print statements and makes the console hard to read.
spark.sparkContext.setLogLevel("ERROR")

SURVEY_PATH = "data_analysis.csv"
STRATEGIES_PATH = "data_analysis_strategies.csv"

# Running tally of what happens at each stage, printed as a summary table
# at the end (and easy to paste into a paper's preprocessing section).
summary = {}

# ============================================================================
# 1. DATA INTEGRATION
# ============================================================================
# Both files are semicolon-delimited and share the free-text column MET1
# (the respondent's answer to "how would you mitigate AI's energy impact?").
# data_analysis.csv holds the raw survey (demographics, attitudes, charge
# allocations). data_analysis_strategies.csv holds four LLMs' independent
# classification of that same free-text answer into 10 strategy categories
# (one boolean column per model per category). We integrate them into a
# single respondent-level table keyed on MET1.
banner("STAGE 1: DATA INTEGRATION")

survey = spark.read.csv(SURVEY_PATH, header=True, sep=";", inferSchema=True)
strategies = spark.read.csv(STRATEGIES_PATH, header=True, sep=";", inferSchema=True)

# Model names in the header (e.g. "qwen2.5-vl-72b-instruct_select_provider")
# contain literal dots, which Spark's column-expression parser treats as
# nested-field access. Sanitize to underscores so every column can be
# referenced safely with F.col() / dot notation throughout the pipeline.
def sanitize_columns(df):
    for c in df.columns:
        safe = c.replace(".", "_")
        if safe != c:
            df = df.withColumnRenamed(c, safe)
    return df

survey = sanitize_columns(survey)
strategies = sanitize_columns(strategies)

n_survey, n_survey_cols = survey.count(), len(survey.columns)
n_strat, n_strat_cols = strategies.count(), len(strategies.columns)
log(f"Loaded data_analysis.csv        -> {n_survey} rows x {n_survey_cols} cols")
log(f"Loaded data_analysis_strategies.csv -> {n_strat} rows x {n_strat_cols} cols")
summary["1. Source: survey"] = f"{n_survey} rows x {n_survey_cols} cols"
summary["1. Source: LLM strategies"] = f"{n_strat} rows x {n_strat_cols} cols"

# Normalize the join key on both sides before joining (trailing/leading
# whitespace differences would otherwise silently drop matches).
survey = survey.withColumn("MET1_key", F.trim(F.lower(F.col("MET1"))))
strategies = strategies.withColumn("MET1_key", F.trim(F.lower(F.col("MET1"))))

# Give the strategies table a row id so duplicate free-text answers
# (e.g. many people just typing "Unsure") don't fan out the join.
from pyspark.sql.window import Window
strategies = strategies.withColumn(
    "row_id", F.row_number().over(Window.orderBy(F.monotonically_increasing_id()))
)
survey = survey.withColumn(
    "row_id", F.row_number().over(Window.orderBy(F.monotonically_increasing_id()))
)

integrated = survey.drop("MET1_key").join(
    strategies.drop("MET1", "MET1_key"), on="row_id", how="inner"
)

n_int, n_int_cols = integrated.count(), len(integrated.columns)
log(f"Joined on respondent row id (MET1-aligned) -> {n_int} rows x {n_int_cols} cols")
summary["1. Integrated dataset"] = f"{n_int} rows x {n_int_cols} cols"

# ============================================================================
# 2. DATA CLEANSING
# ============================================================================
banner("STAGE 2: DATA CLEANSING")

# 2a. Exact-duplicate rows
before = integrated.count()
integrated = integrated.dropDuplicates()
n_dupes = before - integrated.count()
log(f"Exact duplicate rows removed: {n_dupes}")
summary["2. Duplicate rows removed"] = n_dupes

# 2b. Trim whitespace / normalize case in the free-text column
integrated = integrated.withColumn("MET1", F.trim(F.col("MET1")))
integrated = integrated.withColumn(
    "MET1_clean",
    F.when(F.lower(F.col("MET1")).isin("unsure", "not sure", "n/a", "na", ""), "Unsure")
     .otherwise(F.col("MET1"))
)
log("Trimmed whitespace and normalized free-text 'unsure' variants in MET1")

# 2c. Enforce correct types on the numeric survey columns (inferSchema can
# mis-type sparse/edge-case columns; be explicit for anything used downstream)
numeric_cols = [
    "duration", "Prof", "Imp1", "Imp2",
    "Charge_PM", "Charge_Vendors", "Charge_Users", "Charge_Develop", "Charge_Noone",
    "Charge_HH_PM", "Charge_HH_Vendor", "Charge_HH_Users", "Charge_HH_Developers", "Charge_HH_Noone",
    "Household", "Income", "PolOrient", "Gender", "Age", "Education",
]
for c in numeric_cols:
    integrated = integrated.withColumn(c, F.col(c).cast(T.DoubleType()))
log(f"Cast {len(numeric_cols)} numeric columns to DoubleType")

# 2d. Null / missing-value audit, then impute
null_counts = integrated.select(
    [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in numeric_cols]
).collect()[0].asDict()
cols_with_nulls = [c for c, n in null_counts.items() if n > 0]
log(f"Columns with missing values: {cols_with_nulls or 'none'}")
summary["2. Columns with missing values"] = len(cols_with_nulls)

if cols_with_nulls:
    imputer = Imputer(inputCols=cols_with_nulls,
                       outputCols=[c + "_imp" for c in cols_with_nulls],
                       strategy="median")
    integrated = imputer.fit(integrated).transform(integrated)
    for c in cols_with_nulls:
        integrated = integrated.drop(c).withColumnRenamed(c + "_imp", c)
    log(f"Imputed missing values in {cols_with_nulls} using column medians")

# 2e. Range / consistency validation:
#   - Charge_* allocations should each sum to 100 (percent split across 5 actors)
#   - Age should be a plausible adult age
integrated = integrated.withColumn(
    "charge_total",
    F.col("Charge_PM") + F.col("Charge_Vendors") + F.col("Charge_Users")
    + F.col("Charge_Develop") + F.col("Charge_Noone"),
)
invalid_charge = integrated.filter(F.col("charge_total") != 100).count()
log(f"Rows where Charge_* allocation doesn't sum to 100: {invalid_charge} (flagged, not dropped)")
summary["2. Rows with invalid Charge_* total"] = invalid_charge

invalid_age = integrated.filter((F.col("Age") < 16) | (F.col("Age") > 100)).count()
integrated = integrated.filter((F.col("Age") >= 16) & (F.col("Age") <= 100))
log(f"Rows with implausible Age (<16 or >100) removed: {invalid_age}")
summary["2. Rows with implausible Age removed"] = invalid_age

# 2f. Standardize the 40 LLM boolean label columns to 0/1 integers (in case
# any were read as strings "True"/"False" rather than native booleans)
llm_label_cols = [c for c in strategies.columns if c not in ("MET1", "MET1_key", "row_id")]
for c in llm_label_cols:
    integrated = integrated.withColumn(
        c, F.when(F.col(c).cast("string").isin("true", "True", "1"), 1).otherwise(0)
    )
log(f"Standardized {len(llm_label_cols)} LLM boolean label columns to 0/1 integers")

n_clean, n_clean_cols = integrated.count(), len(integrated.columns)
log(f"Dataset after cleansing -> {n_clean} rows x {n_clean_cols} cols")
summary["2. After cleansing"] = f"{n_clean} rows x {n_clean_cols} cols"

# ============================================================================
# 3. DATA TRANSFORMATION
# ============================================================================
banner("STAGE 3: DATA TRANSFORMATION")

# 3a. Recode Likert/ordinal-coded categoricals into readable labels
gender_map = {1: "Male", 2: "Female", 3: "Non-binary", 4: "Other/Prefer not to say"}
env_map = {1: "Environmentalist", 2: "Not environmentalist"}
energy_map = {1: "Low consumption", 2: "High consumption"}

def map_col(df, col, mapping, new_col=None):
    new_col = new_col or col
    m = F.create_map([F.lit(x) for pair in mapping.items() for x in pair])
    return df.withColumn(new_col, m[F.col(col).cast("int")])

integrated = map_col(integrated, "Gender", gender_map, "Gender_label")
integrated = map_col(integrated, "Environmentalist", env_map, "Environmentalist_label")
integrated = map_col(integrated, "EnergyConsumption", energy_map, "EnergyConsumption_label")
log("Recoded Gender, Environmentalist, EnergyConsumption into readable labels")

# 3b. Feature engineering: bucket Age into groups
integrated = integrated.withColumn(
    "AgeGroup",
    F.when(F.col("Age") < 25, "18-24")
     .when(F.col("Age") < 35, "25-34")
     .when(F.col("Age") < 45, "35-44")
     .when(F.col("Age") < 55, "45-54")
     .when(F.col("Age") < 65, "55-64")
     .otherwise("65+"),
)
log("Engineered AgeGroup (6 buckets) from continuous Age")

# 3c. Feature engineering: importance/professional-familiarity composite score
integrated = integrated.withColumn(
    "Concern_Index", (F.col("Imp1") + F.col("Imp2")) / F.lit(2.0)
)
log("Engineered Concern_Index = mean(Imp1, Imp2)")

# 3d. Consensus features from the 40 LLM label columns (4 models x 10
# categories). For each strategy category, compute how many of the 4 models
# agree, and derive a majority-vote (>=2 of 4) boolean flag.
models = ["qwen2_5-vl-72b-instruct", "llama-3_3-70b-instruct",
          "mistral-large-instruct", "openai-gpt-oss-120b"]
categories = ["select_provider", "select_model", "select_modality", "optimize_prompt",
              "reduce_usage", "use_alternative", "abstain_from_ai", "not_sure",
              "other", "irrelevant"]

for cat in categories:
    cols = [f"{m}_{cat}" for m in models]
    integrated = integrated.withColumn(f"agree_{cat}", sum(F.col(c) for c in cols))
    integrated = integrated.withColumn(
        f"majority_{cat}", (F.col(f"agree_{cat}") >= 2).cast("int")
    )
log(f"Built {len(categories)} agree_* (0-4 vote count) and "
    f"{len(categories)} majority_* (>=2 of 4) consensus features across the 4 LLMs")

# 3e. One-hot encode + index nominal categoricals for downstream ML use
categorical_cols = ["Gender", "PolOrient", "Education", "Household", "Income", "AgeGroup"]
indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep")
            for c in categorical_cols]
encoders = [OneHotEncoder(inputCol=c + "_idx", outputCol=c + "_ohe")
            for c in categorical_cols]

# 3f. Scale numeric features (z-score standardization)
numeric_for_scaling = ["duration", "Prof", "Imp1", "Imp2", "Age", "Concern_Index"]
assembler_num = VectorAssembler(inputCols=numeric_for_scaling, outputCol="numeric_vec")
scaler = StandardScaler(inputCol="numeric_vec", outputCol="numeric_scaled",
                         withMean=True, withStd=True)

transform_pipeline = Pipeline(stages=indexers + encoders + [assembler_num, scaler])
integrated = transform_pipeline.fit(integrated).transform(integrated)
log(f"One-hot encoded {len(categorical_cols)} categoricals; "
    f"z-score standardized {len(numeric_for_scaling)} numeric features")

n_trans_cols = len(integrated.columns)
log(f"Dataset after transformation -> {integrated.count()} rows x {n_trans_cols} cols")
summary["3. After transformation"] = f"{integrated.count()} rows x {n_trans_cols} cols"

# ============================================================================
# 4. DATA REDUCTION
# ============================================================================
banner("STAGE 4: DATA REDUCTION")

# 4a. Drop zero-variance / redundant columns (no analytic signal)
variance_check = integrated.select(
    [F.countDistinct(F.col(c)).alias(c) for c in ["Consent", "SeriousPart"]]
).collect()[0].asDict()
zero_variance_cols = [c for c, n in variance_check.items() if n <= 1]
integrated = integrated.drop(*zero_variance_cols)
log(f"Distinct-value counts checked for candidate zero-variance columns: {variance_check}")
log(f"Dropped zero-variance columns: {zero_variance_cols}")
summary["4. Zero-variance columns dropped"] = zero_variance_cols

# 4b. Dimensionality reduction of the LLM label space: the 40 raw per-model
# boolean columns are collapsed to the 10 agree_/10 majority_ consensus
# columns already built in step 3d, dropping the 40 originals.
before_cols = len(integrated.columns)
integrated_reduced = integrated.drop(*llm_label_cols)
after_cols = len(integrated_reduced.columns)
log(f"Collapsed {len(llm_label_cols)} raw per-model LLM label columns -> "
    f"{2 * len(categories)} consensus/majority columns "
    f"({before_cols} -> {after_cols} total columns)")
summary["4. LLM label columns collapsed"] = f"{len(llm_label_cols)} -> {2 * len(categories)}"

# 4c. PCA on the standardized numeric features to further compress dimensionality
pca = PCA(k=3, inputCol="numeric_scaled", outputCol="numeric_pca")
pca_model = pca.fit(integrated_reduced)
integrated_reduced = pca_model.transform(integrated_reduced)
explained = pca_model.explainedVariance.toArray()
log(f"PCA on {len(numeric_for_scaling)} numeric features -> 3 components")
log(f"Explained variance ratios: {np.round(explained, 3).tolist()}, "
    f"cumulative = {round(float(explained.sum()), 3)}")
summary["4. PCA cumulative variance explained"] = round(float(explained.sum()), 3)

# 4d. Numerosity reduction example: a stratified sample (here 50%) for
# faster downstream experimentation while preserving the Gender distribution
sample_fractions = {row["Gender"]: 0.5 for row in integrated_reduced.select("Gender").distinct().collect()}
sampled = integrated_reduced.sampleBy("Gender", fractions=sample_fractions, seed=42)
n_sampled = sampled.count()
log(f"Stratified 50% sample by Gender -> {n_sampled} rows (from {integrated_reduced.count()})")
summary["4. Stratified 50% sample size"] = n_sampled

n_final, n_final_cols = integrated_reduced.count(), len(integrated_reduced.columns)
log(f"Final reduced dataset -> {n_final} rows x {n_final_cols} cols")
summary["4. Final reduced dataset"] = f"{n_final} rows x {n_final_cols} cols"

# ============================================================================
# 5. DESCRIPTIVE STATISTICS / FINDINGS
# ============================================================================
# Exploratory statistics computed on the cleaned, integrated dataset. These
# are descriptive/correlational only (no hypothesis testing), intended to
# characterize the sample and the LLM-classification consensus patterns.
banner("STAGE 5: DESCRIPTIVE STATISTICS / FINDINGS")

# 5a. Sample demographics
log("Demographic composition:")
for col in ["Gender_label", "AgeGroup", "Environmentalist_label", "EnergyConsumption_label"]:
    dist = integrated_reduced.groupBy(col).count().orderBy(F.desc("count")).collect()
    dist_str = ", ".join(f"{r[col]}={r['count']} ({100*r['count']/n_final:.1f}%)" for r in dist)
    log(f"  {col}: {dist_str}")

# 5b. Majority-vote strategy frequency across the sample (how often each
# mitigation strategy was the LLM-consensus classification of MET1)
log("Majority-vote (>=2 of 4 LLMs) strategy frequency, most to least common:")
maj_freq = []
for cat in categories:
    c = integrated_reduced.filter(F.col(f"majority_{cat}") == 1).count()
    maj_freq.append((cat, c))
maj_freq.sort(key=lambda x: -x[1])
for cat, c in maj_freq:
    log(f"  {cat:<18} {c:4d} ({100*c/n_final:.1f}%)")
summary["5. Top majority-vote strategy"] = f"{maj_freq[0][0]} ({100*maj_freq[0][1]/n_final:.1f}%)"

# 5c. Inter-model agreement: how often do all 4 LLMs agree vs. split?
agree_expr = sum(F.col(f"agree_{cat}") for cat in categories)
unanimous_counts = {
    cat: integrated_reduced.filter(F.col(f"agree_{cat}") == 4).count() for cat in categories
}
log(f"Full 4/4-model unanimous agreement (by category): {unanimous_counts}")
total_unanimous = sum(unanimous_counts.values())
summary["5. Total unanimous (4/4) classifications"] = f"{total_unanimous} of {n_final * len(categories)} category-respondent pairs"

# 5d. Correlations (Pearson) between key continuous variables
r_pol_concern = integrated_reduced.stat.corr("PolOrient", "Concern_Index")
r_income_concern = integrated_reduced.stat.corr("Income", "Concern_Index")
r_age_concern = integrated_reduced.stat.corr("Age", "Concern_Index")
r_prof_concern = integrated_reduced.stat.corr("Prof", "Concern_Index")
log("Pearson correlations with Concern_Index (mean of Imp1, Imp2):")
log(f"  PolOrient (political orientation): r = {r_pol_concern:.3f}")
log(f"  Income:                            r = {r_income_concern:.3f}")
log(f"  Age:                               r = {r_age_concern:.3f}")
log(f"  Prof (self-rated AI familiarity):  r = {r_prof_concern:.3f}")
summary["5. r(PolOrient, Concern_Index)"] = round(r_pol_concern, 3)
summary["5. r(Prof familiarity, Concern_Index)"] = round(r_prof_concern, 3)

# 5e. Group comparisons: mean Concern_Index and reduce-usage endorsement by
# environmentalist self-identification
log("Concern_Index and majority_reduce_usage rate by Environmentalist_label:")
grp_stats = (
    integrated_reduced.groupBy("Environmentalist_label")
    .agg(
        F.count("*").alias("n"),
        F.round(F.mean("Concern_Index"), 2).alias("mean_concern"),
        F.round(F.stddev("Concern_Index"), 2).alias("sd_concern"),
        F.round(F.mean("majority_reduce_usage") * 100, 1).alias("pct_reduce_usage"),
    )
    .collect()
)
for r in grp_stats:
    log(f"  {r['Environmentalist_label']:<22} n={r['n']:4d}  "
        f"mean Concern_Index={r['mean_concern']}  sd={r['sd_concern']}  "
        f"reduce_usage majority={r['pct_reduce_usage']}%")
summary["5. Concern_Index by Environmentalist status"] = {
    r["Environmentalist_label"]: r["mean_concern"] for r in grp_stats
}

# ============================================================================
# Persist outputs
# ============================================================================
banner("PERSISTING OUTPUT")

output_cols = [
    "row_id", "MET1_clean", "AgeGroup", "Gender_label", "Environmentalist_label",
    "EnergyConsumption_label", "Concern_Index", "PolOrient", "Education", "Income",
    "Household",
] + [f"agree_{c}" for c in categories] + [f"majority_{c}" for c in categories]

final_output = integrated_reduced.select(*output_cols)

# Write with plain csv (no pandas / Hadoop-writer dependency, so this runs
# the same on Windows without needing winutils.exe or HADOOP_HOME set).
import csv
rows = final_output.collect()
out_path = "output_preprocessed_dataset.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(final_output.columns)
    for row in rows:
        writer.writerow(row)
log(f"Wrote {len(rows)} rows x {len(final_output.columns)} cols -> {out_path}")

# ============================================================================
# Final summary (paper-friendly)
# ============================================================================
banner("PREPROCESSING SUMMARY")
label_width = max(len(k) for k in summary) + 2
for k, v in summary.items():
    print(f"  {k:<{label_width}} {v}")

print("\nSample of final preprocessed dataset:")
final_output.show(5, truncate=60)

spark.stop()
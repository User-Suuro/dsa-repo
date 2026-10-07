"NOTE: ANY DATASET IS NOT INCLUDED IN THIS REPOSITORY (contains personal data). Please request the dataset from the author if you wish to run this pipeline."

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI needed; figures are saved to disk
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, PCA
)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
CSV_PATH = "dataset.csv"
OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def log(msg):
    print(f"  - {msg}")


spark = (
    SparkSession.builder
    .appName("AI-Environment-Awareness-Survey")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "4")
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY")  # lenient timestamp parsing
    .config("spark.sql.execution.arrow.pyspark.enabled", "false")  # avoid PyArrow dependency
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

summary = {}  # running tally printed at the end

# ============================================================================
# 0. LOAD & INSPECT
# ============================================================================
banner("STAGE 0: LOAD & INSPECT")

# multiLine=True is required: some open-ended answers contain line breaks
raw = spark.read.csv(
    CSV_PATH, header=True, inferSchema=False,
    multiLine=True, quote='"', escape='"', encoding="UTF-8",
)
n_raw = raw.count()
log(f"Raw shape: {n_raw} rows x {len(raw.columns)} cols")
summary["0. Raw shape"] = f"{n_raw} rows x {len(raw.columns)} cols"

# Short, safe column names (the originals are full survey questions)
new_names = [
    "timestamp_raw", "username", "consent", "age_group_raw", "ai_usage_freq",
    "aware_energy", "aware_carbon", "consider_impact", "avoid_unnecessary_use",
    "concern_text", "awareness_text",
]
assert len(new_names) == len(raw.columns), "Unexpected number of columns in CSV"
raw = raw.toDF(*new_names)
raw.printSchema()

# ============================================================================
# 1. DATA INTEGRATION
# ============================================================================
banner("STAGE 1: DATA INTEGRATION")
# The questionnaire is one table, but the two open-ended questions (Q7, Q8)
# cannot be analysed as-is. We build a second source - a "text coding" table
# of keyword themes - and integrate it with the structured table on a
# respondent id (same idea as joining the LLM-classification table in the
# reference pipeline).

raw = raw.withColumn(
    "respondent_id", F.row_number().over(Window.orderBy(F.monotonically_increasing_id()))
)

# --- Source A: structured questionnaire responses
source_a = raw.select(
    "respondent_id", "timestamp_raw", "username", "consent", "age_group_raw",
    "ai_usage_freq", "aware_energy", "aware_carbon", "consider_impact",
    "avoid_unnecessary_use",
)


# --- Source B: keyword-coded themes from the open-ended answers
def has_kw(col, *kws):
    """1 if lower-cased text contains any keyword (regex alternation)."""
    pattern = "(" + "|".join(kws) + ")"
    return F.when(F.lower(F.coalesce(F.col(col), F.lit(""))).rlike(pattern), 1).otherwise(0)


source_b = (
    raw.select("respondent_id", "concern_text", "awareness_text")
    # Q7: environmental concerns
    .withColumn("c_water", has_kw("concern_text", "water", "drought", "cooling"))
    .withColumn("c_energy", has_kw("concern_text", "energy", "electric", "power consum", "power usage"))
    .withColumn("c_carbon", has_kw("concern_text", "carbon", "emission", "greenhouse",
                                   "global warming", "climate", "co2", "pollut"))
    .withColumn("c_ewaste", has_kw("concern_text", "e-waste", "e waste", "waste", "hardware",
                                   "mining", "resource", "material"))
    .withColumn("c_noise_heat", has_kw("concern_text", "noise", "heat", "temperature", "loud"))
    .withColumn("c_ecosystem", has_kw("concern_text", "flora", "fauna", "wildlife", "ecosystem",
                                      "biodiversity", "habitat", "forest", r"\bland\b"))
    # Q8: suggested solutions
    .withColumn("s_education", has_kw("awareness_text", "educat", "school", "teach", "seminar",
                                      "literacy", "workshop", "training", "learn", "inform"))
    .withColumn("s_campaign", has_kw("awareness_text", "campaign", "social media", "advocacy",
                                     r"\bposts?\b", "media", "poster", "seminar"))
    .withColumn("s_policy", has_kw("awareness_text", "law", "regulat", "polic", "government",
                                   "legislat", "limit", "rules", "guideline"))
    .withColumn("s_transparency", has_kw("awareness_text", "transparen", "disclos", "report",
                                         "open about", "accountab"))
    .withColumn("s_tech", has_kw("awareness_text", "efficien", "renewable", "clean energy",
                                 "green", "sustainab", "solar"))
    .withColumn("s_responsible_use", has_kw("awareness_text", "responsib", "moderat", "avoid",
                                            "mindful", "wisely", "proper use"))
)
theme_cols = [c for c in source_b.columns if c.startswith(("c_", "s_"))]
source_b_flags = source_b.select("respondent_id", "concern_text", "awareness_text", *theme_cols)

log(f"Source A (structured survey): {source_a.count()} rows x {len(source_a.columns)} cols")
log(f"Source B (text-coded themes): {source_b_flags.count()} rows x {len(source_b_flags.columns)} cols")

integrated = source_a.join(source_b_flags, on="respondent_id", how="inner")
n_int = integrated.count()
log(f"Integrated on respondent_id -> {n_int} rows x {len(integrated.columns)} cols")
summary["1. Source A (structured)"] = f"{source_a.count()} rows x {len(source_a.columns)} cols"
summary["1. Source B (text themes)"] = f"{source_b_flags.count()} rows x {len(source_b_flags.columns)} cols"
summary["1. Integrated dataset"] = f"{n_int} rows x {len(integrated.columns)} cols"

# ============================================================================
# 2. DATA CLEANSING
# ============================================================================
banner("STAGE 2: DATA CLEANSING")

# 2a. Trim whitespace in every string column; empty strings -> NULL
str_cols = [f.name for f in integrated.schema.fields if isinstance(f.dataType, T.StringType)]
for c in str_cols:
    integrated = integrated.withColumn(c, F.trim(F.col(c)))
    integrated = integrated.withColumn(c, F.when(F.col(c) == "", None).otherwise(F.col(c)))
log(f"Trimmed whitespace / blanks->NULL in {len(str_cols)} string columns")

# 2b. Exact duplicate check (ignoring the generated id)
before = integrated.count()
integrated = integrated.dropDuplicates([c for c in integrated.columns if c != "respondent_id"])
n_dupes = before - integrated.count()
log(f"Exact duplicate responses removed: {n_dupes}")
summary["2. Duplicate rows removed"] = n_dupes

# 2c. Parse the timestamp ("2026/10/01 1:07:01 PM GMT+8")
integrated = (
    integrated
    .withColumn("ts_tmp", F.regexp_replace("timestamp_raw", r"\s*GMT[+-]\d+", ""))
    .withColumn("submitted_at", F.to_timestamp("ts_tmp", "yyyy/MM/dd h:mm:ss a"))
    .drop("ts_tmp")
)
n_bad_ts = integrated.filter(F.col("submitted_at").isNull()).count()
log(f"Timestamp parsed to TimestampType (unparseable: {n_bad_ts})")
summary["2. Unparseable timestamps"] = n_bad_ts

# 2d. Normalise capitalisation ('Strongly Agree' vs 'Strongly agree')
likert_cols = ["aware_energy", "aware_carbon", "consider_impact", "avoid_unnecessary_use"]
for c in likert_cols:
    integrated = integrated.withColumn(c, F.initcap(F.lower(F.col(c))))
integrated = integrated.withColumn("ai_usage_freq", F.initcap(F.lower(F.col("ai_usage_freq"))))
log("Normalised capitalisation of Likert + usage-frequency columns")

# 2e. Fix age-band typo: '24-34' overlaps '18-24' -> '25-34'
n_age_fix = integrated.filter(F.col("age_group_raw") == "24-34").count()
integrated = integrated.withColumn(
    "age_group",
    F.when(F.col("age_group_raw") == "24-34", "25-34").otherwise(F.col("age_group_raw")),
)
log(f"Fixed age-band label '24-34' -> '25-34' in {n_age_fix} rows (overlapping-band typo)")
summary["2. Age-band labels corrected"] = n_age_fix


# 2f. Open-ended answers: junk (".", "n/a", very short) -> missing
def clean_text(col):
    t = F.col(col)
    return F.when(
        t.isNull() | (F.length(t) < 3)
        | F.lower(t).isin("n/a", "na", "none", "no", "nothing", "-", "."),
        None,
    ).otherwise(t)


for c in ["concern_text", "awareness_text"]:
    integrated = integrated.withColumn(c, clean_text(c))
integrated = (
    integrated
    .withColumn("has_concern_text", F.col("concern_text").isNotNull().cast("int"))
    .withColumn("has_awareness_text", F.col("awareness_text").isNotNull().cast("int"))
)
n_miss_c = integrated.filter("has_concern_text = 0").count()
n_miss_a = integrated.filter("has_awareness_text = 0").count()
log(f"Missing/junk open-ended answers -> Q7: {n_miss_c}, Q8: {n_miss_a}")
summary["2. Missing/junk Q7 answers"] = n_miss_c
summary["2. Missing/junk Q8 answers"] = n_miss_a

# 2g. Validate categorical domains
valid_likert = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
valid_freq = ["Rarely", "Once A Week", "Several Times A Week", "Once A Day", "Several Times A Day"]
bad_likert = integrated.filter(
    ~F.col("aware_energy").isin(valid_likert) | ~F.col("aware_carbon").isin(valid_likert)
    | ~F.col("consider_impact").isin(valid_likert) | ~F.col("avoid_unnecessary_use").isin(valid_likert)
).count()
bad_freq = integrated.filter(~F.col("ai_usage_freq").isin(valid_freq)).count()
log(f"Out-of-domain Likert rows: {bad_likert}; out-of-domain usage-frequency rows: {bad_freq}")
summary["2. Out-of-domain Likert / frequency rows"] = f"{bad_likert} / {bad_freq}"

# 2h. Consent filter (research ethics)
before = integrated.count()
integrated = integrated.filter(F.lower(F.col("consent")).startswith("yes"))
log(f"Rows without consent removed: {before - integrated.count()}")

# 2i. Anonymise: drop the e-mail (PII), keep only a flag
n_email = integrated.filter(F.col("username").isNotNull()).count()
integrated = integrated.withColumn("gave_email", F.col("username").isNotNull().cast("int")).drop("username")
log(f"Dropped 'username' (PII; {n_email} respondents gave an e-mail); kept 'gave_email' flag only")

n_clean = integrated.count()
log(f"Dataset after cleansing -> {n_clean} rows x {len(integrated.columns)} cols")
summary["2. After cleansing"] = f"{n_clean} rows x {len(integrated.columns)} cols"

# ============================================================================
# 3. DATA TRANSFORMATION
# ============================================================================
banner("STAGE 3: DATA TRANSFORMATION")

# 3a. Ordinal encoding (Likert 1-5, usage frequency 1-5)
likert_map = {"Strongly Disagree": 1, "Disagree": 2, "Neutral": 3, "Agree": 4, "Strongly Agree": 5}
freq_map = {"Rarely": 1, "Once A Week": 2, "Several Times A Week": 3,
            "Once A Day": 4, "Several Times A Day": 5}


def map_col(df, col, mapping, new_col):
    m = F.create_map([F.lit(x) for kv in mapping.items() for x in kv])
    return df.withColumn(new_col, m[F.col(col)].cast("double"))


integrated = map_col(integrated, "aware_energy", likert_map, "aware_energy_n")
integrated = map_col(integrated, "aware_carbon", likert_map, "aware_carbon_n")
integrated = map_col(integrated, "consider_impact", likert_map, "consider_impact_n")
integrated = map_col(integrated, "avoid_unnecessary_use", likert_map, "avoid_use_n")
integrated = map_col(integrated, "ai_usage_freq", freq_map, "usage_freq_n")
log("Ordinal-encoded 4 Likert items (1-5) and AI usage frequency (1-5)")

# 3b. Composite indices
integrated = (
    integrated
    .withColumn("Awareness_Index", (F.col("aware_energy_n") + F.col("aware_carbon_n")) / 2.0)
    .withColumn("Behavior_Index", (F.col("consider_impact_n") + F.col("avoid_use_n")) / 2.0)
    .withColumn("Awareness_Behavior_Gap", F.col("Awareness_Index") - F.col("Behavior_Index"))
)
log("Engineered Awareness_Index = mean(Q3,Q4); Behavior_Index = mean(Q5,Q6); "
    "Awareness_Behavior_Gap = Awareness - Behavior")

# 3c. Categorical segments
integrated = (
    integrated
    .withColumn("usage_segment",
                F.when(F.col("usage_freq_n") >= 4, "Heavy (daily+)")
                 .when(F.col("usage_freq_n") >= 2, "Moderate (weekly)")
                 .otherwise("Light (rarely)"))
    .withColumn("awareness_level",
                F.when(F.col("Awareness_Index") >= 4, "High")
                 .when(F.col("Awareness_Index") >= 3, "Medium").otherwise("Low"))
    .withColumn("behavior_level",
                F.when(F.col("Behavior_Index") >= 4, "High")
                 .when(F.col("Behavior_Index") >= 3, "Medium").otherwise("Low"))
    .withColumn("gap_type",
                F.when(F.col("Awareness_Behavior_Gap") >= 1, "Aware but not acting")
                 .when(F.col("Awareness_Behavior_Gap") <= -1, "Acting beyond awareness")
                 .otherwise("Aligned"))
)
log("Engineered usage_segment, awareness_level, behavior_level, gap_type")

# 3d. Text-derived numeric features
c_themes = [c for c in theme_cols if c.startswith("c_")]
s_themes = [c for c in theme_cols if c.startswith("s_")]
integrated = (
    integrated
    .withColumn("concern_len_words",
                F.when(F.col("concern_text").isNull(), 0)
                 .otherwise(F.size(F.split(F.col("concern_text"), r"\s+"))))
    .withColumn("awareness_len_words",
                F.when(F.col("awareness_text").isNull(), 0)
                 .otherwise(F.size(F.split(F.col("awareness_text"), r"\s+"))))
    .withColumn("concern_theme_count", sum(F.col(c) for c in c_themes))
    .withColumn("solution_theme_count", sum(F.col(c) for c in s_themes))
)
log("Engineered word counts and theme counts for the open-ended answers")

# 3e. One-hot encode nominal categoricals + z-score scale numeric features
cat_cols = ["age_group", "usage_segment"]
indexers = [StringIndexer(inputCol=c, outputCol=c + "_idx", handleInvalid="keep") for c in cat_cols]
encoders = [OneHotEncoder(inputCol=c + "_idx", outputCol=c + "_ohe") for c in cat_cols]
numeric_for_scaling = ["aware_energy_n", "aware_carbon_n", "consider_impact_n",
                       "avoid_use_n", "usage_freq_n"]
assembler = VectorAssembler(inputCols=numeric_for_scaling, outputCol="numeric_vec")
scaler = StandardScaler(inputCol="numeric_vec", outputCol="numeric_scaled",
                        withMean=True, withStd=True)
integrated = Pipeline(stages=indexers + encoders + [assembler, scaler]).fit(integrated).transform(integrated)
log(f"One-hot encoded {cat_cols}; z-score scaled {numeric_for_scaling}")

integrated = integrated.cache()
summary["3. After transformation"] = f"{integrated.count()} rows x {len(integrated.columns)} cols"
log(f"Dataset after transformation -> {summary['3. After transformation']}")

# ============================================================================
# 4. DATA REDUCTION
# ============================================================================
banner("STAGE 4: DATA REDUCTION")

# 4a. Zero-variance columns (e.g. consent: everyone said yes)
check_cols = ["consent", "gave_email"] + theme_cols
distinct = integrated.agg(*[F.countDistinct(c).alias(c) for c in check_cols]).collect()[0].asDict()
zero_var = [c for c, n in distinct.items() if n <= 1]
log(f"Zero-variance columns found: {zero_var}")
integrated = integrated.drop(*zero_var)
summary["4. Zero-variance columns dropped"] = zero_var

# 4b. Redundant raw / intermediate columns
redundant = ["timestamp_raw", "age_group_raw", "numeric_vec", "age_group_idx", "usage_segment_idx"]
integrated_reduced = integrated.drop(*[c for c in redundant if c in integrated.columns])
log(f"Dropped redundant raw/intermediate columns: {redundant}")

# 4c. PCA on the 5 standardised numeric features
pca_model = PCA(k=3, inputCol="numeric_scaled", outputCol="numeric_pca").fit(integrated_reduced)
integrated_reduced = pca_model.transform(integrated_reduced)
explained = pca_model.explainedVariance.toArray()
log(f"PCA (5 numeric features -> 3 components) explained variance: "
    f"{np.round(explained, 3).tolist()}, cumulative = {explained.sum():.3f}")
summary["4. PCA cumulative variance (k=3)"] = round(float(explained.sum()), 3)

# 4d. Numerosity reduction: stratified 50% sample by usage_segment
segs = [r["usage_segment"] for r in integrated_reduced.select("usage_segment").distinct().collect()]
sampled = integrated_reduced.sampleBy("usage_segment", {s: 0.5 for s in segs}, seed=42)
n_samp = sampled.count()
log(f"Stratified 50% sample by usage_segment -> {n_samp} rows (of {integrated_reduced.count()})")
summary["4. Stratified sample size"] = n_samp

integrated_reduced = integrated_reduced.cache()
n_final = integrated_reduced.count()
summary["4. Final reduced dataset"] = f"{n_final} rows x {len(integrated_reduced.columns)} cols"
log(f"Final reduced dataset -> {summary['4. Final reduced dataset']}")

# ============================================================================
# 5. ANALYSIS (basic statistics with PySpark DataFrame operations)
# ============================================================================
banner("STAGE 5: ANALYSIS & FINDINGS")
df = integrated_reduced
insights = []
tables = {}


def pct(n, d):
    return 100.0 * n / d if d else 0.0


# --- Analysis 1: descriptive statistics --------------------------------------
log("ANALYSIS 1: Descriptive statistics (mean, sd, median, min, max)")
desc_vars = ["aware_energy_n", "aware_carbon_n", "consider_impact_n", "avoid_use_n",
             "usage_freq_n", "Awareness_Index", "Behavior_Index", "Awareness_Behavior_Gap"]
agg_exprs = []
for c in desc_vars:
    agg_exprs += [F.round(F.mean(c), 2).alias(f"{c}__mean"),
                  F.round(F.stddev(c), 2).alias(f"{c}__sd"),
                  F.expr(f"percentile_approx({c}, 0.5)").alias(f"{c}__median"),
                  F.min(c).alias(f"{c}__min"), F.max(c).alias(f"{c}__max")]
d = df.agg(*agg_exprs).collect()[0].asDict()
desc_rows = [{"variable": c, "mean": d[f"{c}__mean"], "sd": d[f"{c}__sd"],
              "median": round(d[f"{c}__median"], 2), "min": d[f"{c}__min"], "max": d[f"{c}__max"]}
             for c in desc_vars]
desc_pdf = pd.DataFrame(desc_rows)
tables["analysis1_descriptive_stats"] = desc_pdf
print(desc_pdf.to_string(index=False))
m = {r["variable"]: r for r in desc_rows}
_aw, _bh, _gp = (m["Awareness_Index"]["mean"], m["Behavior_Index"]["mean"],
                 m["Awareness_Behavior_Gap"]["mean"])
_rel = ("slightly higher than" if _gp > 0.1 else "slightly lower than" if _gp < -0.1
        else "about the same as")
insights.append(
    f"Overall awareness is {_rel} self-reported behaviour: mean Awareness_Index = {_aw} vs "
    f"Behavior_Index = {_bh} (1-5 scale); average gap = {_gp} points. Both sit in the 'Agree' range."
)

# --- Analysis 2: Likert distributions ----------------------------------------
log("ANALYSIS 2: Likert distributions and % agreement (Agree + Strongly Agree)")
likert_items = {
    "aware_energy": "Aware AI uses lots of energy (Q3)",
    "aware_carbon": "Aware data centres add emissions (Q4)",
    "consider_impact": "Consider env. impact when using AI (Q5)",
    "avoid_unnecessary_use": "Avoid unnecessary AI use (Q6)",
}
order = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
likert_rows = []
for col, label in likert_items.items():
    counts = {r[col]: r["count"] for r in df.groupBy(col).count().collect()}
    row = {"item": label, **{k: counts.get(k, 0) for k in order}}
    row["pct_agree"] = round(pct(row["Agree"] + row["Strongly Agree"], n_final), 1)
    row["pct_disagree"] = round(pct(row["Disagree"] + row["Strongly Disagree"], n_final), 1)
    likert_rows.append(row)
likert_pdf = pd.DataFrame(likert_rows)
tables["analysis2_likert_distribution"] = likert_pdf
print(likert_pdf.to_string(index=False))
top = max(likert_rows, key=lambda r: r["pct_agree"])
low = min(likert_rows, key=lambda r: r["pct_agree"])
insights.append(
    f"Highest agreement: '{top['item']}' ({top['pct_agree']}% agree). "
    f"Lowest agreement: '{low['item']}' ({low['pct_agree']}%)."
)

# --- Analysis 3: Awareness-behaviour gap -------------------------------------
log("ANALYSIS 3: Awareness-behaviour gap types")
gap = (df.groupBy("gap_type").count()
       .withColumn("pct", F.round(100.0 * F.col("count") / n_final, 1))
       .orderBy(F.desc("count")).toPandas())
tables["analysis3_gap_types"] = gap
print(gap.to_string(index=False))
g = gap.set_index("gap_type")["pct"].to_dict()
insights.append(
    f"{g.get('Aligned', 0.0)}% of respondents are aligned (awareness and behaviour within 1 point), "
    f"{g.get('Aware but not acting', 0.0)}% are 'aware but not acting' and "
    f"{g.get('Acting beyond awareness', 0.0)}% act more than their awareness score suggests."
)

# --- Analysis 4: Group comparison by usage segment ---------------------------
log("ANALYSIS 4: Mean indices by AI usage segment")
by_seg = (df.groupBy("usage_segment")
          .agg(F.count("*").alias("n"),
               F.round(F.mean("Awareness_Index"), 2).alias("mean_awareness"),
               F.round(F.mean("Behavior_Index"), 2).alias("mean_behavior"),
               F.round(F.mean("Awareness_Behavior_Gap"), 2).alias("mean_gap"),
               F.round(F.stddev("Behavior_Index"), 2).alias("sd_behavior"))
          .toPandas())
seg_order = ["Light (rarely)", "Moderate (weekly)", "Heavy (daily+)"]
by_seg["usage_segment"] = pd.Categorical(by_seg["usage_segment"], seg_order, ordered=True)
by_seg = by_seg.sort_values("usage_segment").reset_index(drop=True)
tables["analysis4_by_usage_segment"] = by_seg
print(by_seg.to_string(index=False))
bs = by_seg.set_index("usage_segment")
if "Heavy (daily+)" in bs.index and "Light (rarely)" in bs.index:
    insights.append(
        f"Heavy users (daily+, n={int(bs.loc['Heavy (daily+)', 'n'])}) have Behavior_Index "
        f"{bs.loc['Heavy (daily+)', 'mean_behavior']} vs {bs.loc['Light (rarely)', 'mean_behavior']} for light "
        f"users (n={int(bs.loc['Light (rarely)', 'n'])}); awareness is {bs.loc['Heavy (daily+)', 'mean_awareness']} "
        f"vs {bs.loc['Light (rarely)', 'mean_awareness']}."
    )

# --- Analysis 5: Pearson correlations ----------------------------------------
log("ANALYSIS 5: Pearson correlations")
pairs = [("aware_energy_n", "aware_carbon_n"), ("Awareness_Index", "Behavior_Index"),
         ("consider_impact_n", "avoid_use_n"), ("usage_freq_n", "Awareness_Index"),
         ("usage_freq_n", "Behavior_Index"), ("aware_energy_n", "consider_impact_n"),
         ("aware_carbon_n", "avoid_use_n")]
corr_pdf = pd.DataFrame([{"var_1": a, "var_2": b, "pearson_r": round(df.stat.corr(a, b), 3)}
                         for a, b in pairs])
tables["analysis5_correlations"] = corr_pdf
print(corr_pdf.to_string(index=False))


def strength(r):
    a = abs(r)
    return "strong" if a >= 0.6 else "moderate" if a >= 0.3 else "weak"


def get_r(a, b):
    return float(corr_pdf.loc[(corr_pdf.var_1 == a) & (corr_pdf.var_2 == b), "pearson_r"].iloc[0])


r_ab = get_r("Awareness_Index", "Behavior_Index")
r_uf = get_r("usage_freq_n", "Behavior_Index")
insights.append(
    f"Awareness and behaviour show a {strength(r_ab)} correlation (r = {r_ab}); "
    f"usage frequency vs behaviour is {strength(r_uf)} (r = {r_uf}). "
    f"Correlation does not imply causation (n = {n_final})."
)

# --- Analysis 6: Open-ended themes -------------------------------------------
log("ANALYSIS 6: Open-ended themes (keyword coding of Q7 and Q8)")
n_c = df.filter("has_concern_text = 1").count()
n_s = df.filter("has_awareness_text = 1").count()
theme_labels = {
    "c_water": "Water use / scarcity", "c_energy": "Energy / electricity use",
    "c_carbon": "Carbon / climate / pollution", "c_ewaste": "E-waste / resources",
    "c_noise_heat": "Noise / heat", "c_ecosystem": "Ecosystem / land impact",
    "s_education": "Education & training", "s_campaign": "Awareness campaigns / media",
    "s_policy": "Policy / regulation", "s_transparency": "Transparency / reporting",
    "s_tech": "Efficient / renewable tech", "s_responsible_use": "Responsible / mindful use",
}
present = [c for c in theme_labels if c in df.columns]
sums_c = df.filter("has_concern_text = 1").agg(
    *[F.sum(c).alias(c) for c in present if c.startswith("c_")]).collect()[0].asDict()
sums_s = df.filter("has_awareness_text = 1").agg(
    *[F.sum(c).alias(c) for c in present if c.startswith("s_")]).collect()[0].asDict()
theme_rows = []
for c in present:
    isq7 = c.startswith("c_")
    val = int((sums_c if isq7 else sums_s)[c] or 0)
    theme_rows.append({"question": "Q7 concern" if isq7 else "Q8 solution",
                       "theme": theme_labels[c], "mentions": val,
                       "pct_of_answers": round(pct(val, n_c if isq7 else n_s), 1)})
theme_pdf = pd.DataFrame(theme_rows).sort_values(["question", "mentions"], ascending=[True, False])
tables["analysis6_open_ended_themes"] = theme_pdf
print(theme_pdf.to_string(index=False))
q7 = theme_pdf[theme_pdf.question == "Q7 concern"].iloc[0]
q8 = theme_pdf[theme_pdf.question == "Q8 solution"].iloc[0]
insights.append(
    f"Most-cited environmental concern (Q7): {q7.theme} ({q7.pct_of_answers}% of {n_c} valid answers). "
    f"Most-suggested solution (Q8): {q8.theme} ({q8.pct_of_answers}% of {n_s} valid answers). "
    f"Themes are keyword-coded, so treat percentages as approximate."
)

# --- Analysis 7: Age-group profile -------------------------------------------
log("ANALYSIS 7: Age-group profile")
by_age = (df.groupBy("age_group")
          .agg(F.count("*").alias("n"),
               F.round(F.mean("Awareness_Index"), 2).alias("mean_awareness"),
               F.round(F.mean("Behavior_Index"), 2).alias("mean_behavior"),
               F.round(F.mean("usage_freq_n"), 2).alias("mean_usage_freq"))
          .orderBy("age_group").toPandas())
by_age["pct"] = (100 * by_age.n / n_final).round(1)
tables["analysis7_by_age_group"] = by_age
print(by_age.to_string(index=False))
share_young = float(by_age.loc[by_age.age_group == "18-24", "pct"].sum())
insights.append(
    f"The sample is young: {share_young}% are aged 18-24, so age-group comparisons are not reliable "
    f"(other bands have very few respondents) and results do not generalise to the wider public."
)

# --- Analysis 8: Behaviour by water-concern mention --------------------------
log("ANALYSIS 8: Behaviour by whether respondent mentioned water as a concern")
by_water = (df.filter("has_concern_text = 1")
            .groupBy("c_water")
            .agg(F.count("*").alias("n"),
                 F.round(F.mean("Awareness_Index"), 2).alias("mean_awareness"),
                 F.round(F.mean("Behavior_Index"), 2).alias("mean_behavior"))
            .orderBy("c_water").toPandas())
tables["analysis8_water_concern_vs_behavior"] = by_water
print(by_water.to_string(index=False))

banner("KEY INSIGHTS")
for i, s in enumerate(insights, 1):
    print(f"  {i}. {s}")

# ============================================================================
# 6. VISUALISATIONS
# ============================================================================
banner("STAGE 6: VISUALISATIONS")
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})

# --- Fig 1: diverging stacked bar --------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 3.8))
colors = ["#b2182b", "#ef8a62", "#d9d9d9", "#67a9cf", "#2166ac"]
tot = likert_pdf[order].sum(axis=1)
share = likert_pdf[order].div(tot, axis=0) * 100
labels = likert_pdf["item"].tolist()[::-1]
share = share.iloc[::-1].reset_index(drop=True)
start = (-(share["Strongly Disagree"] + share["Disagree"] + share["Neutral"] / 2)).values.copy()
for cat, colr in zip(order, colors):
    ax.barh(labels, share[cat], left=start, color=colr, label=cat, edgecolor="white")
    start = start + share[cat].values
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("% of respondents (centred on Neutral)")
ax.set_title(f"Awareness vs. behaviour: Likert responses (n = {n_final})")
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=8)
plt.tight_layout()
f1 = os.path.join(FIG_DIR, "fig1_likert_diverging.png")
plt.savefig(f1, bbox_inches="tight"); plt.close()
log(f"Saved {f1}")

# --- Fig 2: awareness vs behaviour by usage segment ---------------------------
fig, ax = plt.subplots(figsize=(6.5, 3.9))
bsg = by_seg.copy()
x = np.arange(len(bsg)); w = 0.36
b1 = ax.bar(x - w / 2, bsg.mean_awareness, w, label="Awareness Index", color="#2166ac")
b2 = ax.bar(x + w / 2, bsg.mean_behavior, w, label="Behavior Index", color="#ef8a62")
ax.bar_label(b1, fmt="%.2f", padding=2); ax.bar_label(b2, fmt="%.2f", padding=2)
ax.set_xticks(x)
ax.set_xticklabels([f"{s}\n(n={int(n)})" for s, n in zip(bsg.usage_segment, bsg.n)])
ax.set_ylim(0, 5.6); ax.set_ylabel("Mean score (1-5)")
ax.set_title("Awareness vs. behaviour by AI-usage segment")
ax.legend(frameon=False, loc="upper left", ncol=2)
plt.tight_layout()
f2 = os.path.join(FIG_DIR, "fig2_index_by_usage.png")
plt.savefig(f2); plt.close()
log(f"Saved {f2}")

# --- Fig 3: open-ended themes -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
for ax, q, colr, ttl in zip(axes, ["Q7 concern", "Q8 solution"], ["#b2182b", "#1a9850"],
                            [f"Environmental concerns (Q7, n={n_c})",
                             f"Suggested solutions (Q8, n={n_s})"]):
    t = theme_pdf[theme_pdf.question == q].sort_values("pct_of_answers")
    ax.barh(t.theme, t.pct_of_answers, color=colr)
    ax.bar_label(ax.containers[0], fmt="%.0f%%", padding=2)
    ax.set_xlim(0, max(100, t.pct_of_answers.max() + 12))
    ax.set_title(ttl); ax.set_xlabel("% of valid open-ended answers")
plt.tight_layout()
f3 = os.path.join(FIG_DIR, "fig3_open_ended_themes.png")
plt.savefig(f3); plt.close()
log(f"Saved {f3}")

# --- Fig 4: correlation heatmap + gap-type pie --------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
hm_vars = ["aware_energy_n", "aware_carbon_n", "consider_impact_n", "avoid_use_n", "usage_freq_n"]
hm_lbl = ["Q3 energy\naware", "Q4 carbon\naware", "Q5 consider\nimpact", "Q6 avoid\nuse", "Usage\nfreq"]
cm = np.array([[1.0 if a == b else df.stat.corr(a, b) for b in hm_vars] for a in hm_vars])
im = axes[0].imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
axes[0].set_xticks(range(5)); axes[0].set_xticklabels(hm_lbl, fontsize=7)
axes[0].set_yticks(range(5)); axes[0].set_yticklabels(hm_lbl, fontsize=7)
for i in range(5):
    for j in range(5):
        axes[0].text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if abs(cm[i, j]) > 0.6 else "black")
axes[0].set_title("Pearson correlation matrix")
plt.colorbar(im, ax=axes[0], fraction=0.046)
gp = gap.set_index("gap_type")["count"]
pie_cols = {"Aligned": "#67a9cf", "Aware but not acting": "#ef8a62", "Acting beyond awareness": "#999999"}
axes[1].pie(gp, labels=gp.index, autopct="%1.0f%%", startangle=90,
            colors=[pie_cols[k] for k in gp.index], wedgeprops={"edgecolor": "white"})
axes[1].set_title("Awareness-behaviour alignment")
plt.tight_layout()
f4 = os.path.join(FIG_DIR, "fig4_correlation_and_gap.png")
plt.savefig(f4); plt.close()
log(f"Saved {f4}")

# ============================================================================
# 7. PERSIST OUTPUTS
# ============================================================================
banner("STAGE 7: PERSISTING OUTPUTS")

output_cols = [
    "respondent_id", "submitted_at", "age_group", "ai_usage_freq", "usage_segment",
    "aware_energy", "aware_carbon", "consider_impact", "avoid_unnecessary_use",
    "aware_energy_n", "aware_carbon_n", "consider_impact_n", "avoid_use_n", "usage_freq_n",
    "Awareness_Index", "Behavior_Index", "Awareness_Behavior_Gap",
    "awareness_level", "behavior_level", "gap_type",
    "concern_text", "awareness_text", "concern_len_words", "awareness_len_words",
] + [c for c in theme_cols if c in df.columns] + ["concern_theme_count", "solution_theme_count"]

final = df.select(*output_cols).orderBy("respondent_id").toPandas()
processed_path = os.path.join(OUT_DIR, "processed_dataset.csv")
final.to_csv(processed_path, index=False, encoding="utf-8-sig")
log(f"Wrote processed dataset: {processed_path} ({final.shape[0]} rows x {final.shape[1]} cols)")

for name, t in tables.items():
    p = os.path.join(OUT_DIR, f"{name}.csv")
    t.to_csv(p, index=False)
    log(f"Wrote {p}")

with open(os.path.join(OUT_DIR, "key_insights.txt"), "w", encoding="utf-8") as fh:
    fh.write("KEY INSIGHTS\n" + "=" * 12 + "\n")
    for i, s in enumerate(insights, 1):
        fh.write(f"{i}. {s}\n")

banner("PREPROCESSING SUMMARY")
w = max(len(k) for k in summary) + 2
for k, v in summary.items():
    print(f"  {k:<{w}} {v}")

print("\nSample of processed dataset:")
print(final[["respondent_id", "age_group", "usage_segment", "Awareness_Index",
             "Behavior_Index", "gap_type"]].head(5).to_string(index=False))

spark.stop()
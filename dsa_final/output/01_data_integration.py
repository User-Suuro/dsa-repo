"""
01_data_integration.py
======================
RUN ORDER: 1 of 5

Stage 1 — Topic-Aligned Multi-Source Integration Based on `main_dataset.csv` & `docs/` (PySpark)
------------------------------------------------------------------------------------------------
Revolves strictly around our study topic in `docs/DSA - Proposal.pdf` and
`docs/Data Scalability and Analytics - Data transformation (Pyspark) (1).pdf`:
  "AI and the Environment: An Analysis of Public Awareness"

Integrates `main_dataset.csv` (Main Study Basis) with the public datasets into ONE
unified dataset using ONLY the features related to our study (Q1 through Q8),
while removing unrelated columns and flagging invalid or off-topic/unrelated rows:

  - Identifiers: `respondent_id`, `source_study`, `study_role`
  - Part I (Respondent Profile):
      * `age_group` ("1. What is your age?": 18-24, 25-34, 35-44, 45-54, 55 and above)
      * `ai_usage_freq` & `usage_freq_n` ("2. How often do you use AI tools?": 1..5)
  - Part II (Environmental Awareness):
      * `aware_energy` & `aware_energy_n` ("3. Aware AI consumes significant electricity/energy")
      * `aware_carbon` & `aware_carbon_n` ("4. Aware AI data centers contribute to carbon/env impacts")
  - Part III (Responsible AI Usage):
      * `consider_impact` & `consider_impact_n` ("5. Consider impact when deciding how to use AI")
      * `avoid_unnecessary_use` & `avoid_use_n` ("6. Avoid unnecessary use of AI when task can be done without it")
  - Part IV (Open-Ended Qualitative Questions — Topic-Filtered):
      * `concern_text` ("7. What environmental impact of AI concerns you the most, and why?")
      * `awareness_text` ("8. What should be done to increase public awareness and encourage responsible use of AI?")
  - Audit Columns:
      * `is_valid_record` (1 = Valid & Topic-Related, 0 = Invalid or Not Related to Main Topic)
      * `invalid_reason` (Exact reason why the row was lost/removed during cleaning & integration)

Outputs -> `output/integration/`:
  - `stage1_integrated_dataset.csv` (Single unified dataset: 3,766 raw rows x 18 cols)
  - `stage1_lost_rows_by_study.csv` (Per-study summary of Raw, Cleaned, and Lost/Removed rows)
  - `stage1_lost_rows_reasons_detail.csv` (Detailed breakdown of why rows were lost/removed per study)
"""

import os
from typing import List

from pyspark.sql import DataFrame, SparkSession, Window, functions as F

from pipeline_utils import (
    DATASET_DIR,
    INTEGRATION_DIR,
    JUNK_TEXT_TOKENS,
    LIVERPOOL_DIR,
    MAIN_TOPIC_ENV_PATTERN,
    MAIN_TOPIC_RESPONSIBLE_USE_PATTERN,
    REVERSE_5PT_OFFSET,
    STAGE1_LOST_REASONS_CSV,
    STAGE1_LOST_SUMMARY_CSV,
    STAGE1_UNIFIED_CSV,
    STAGING_DIR,
    banner,
    create_spark_session,
    log,
    read_csv_spark,
    row_mean_null_safe,
    sanitize_df_columns,
    score_to_frequency_label,
    score_to_likert_label,
    stage_underscore_csv,
    write_spark_df_to_csv,
    xlsx_sheet_to_csv,
)

# Strictly aligned to `docs/Data Scalability and Analytics - Data transformation (Pyspark) (1).pdf`
# Unrelated columns (country, respondent_role, academic_level, discipline_faculty) are removed.
UNIFIED_MAIN_SCHEMA_COLS: List[str] = [
    "respondent_id",
    "source_study",
    "study_role",
    "age_group",
    "ai_usage_freq",
    "usage_freq_n",
    "aware_energy",
    "aware_energy_n",
    "aware_carbon",
    "aware_carbon_n",
    "consider_impact",
    "consider_impact_n",
    "avoid_unnecessary_use",
    "avoid_use_n",
    "concern_text",
    "awareness_text",
    "is_valid_record",
    "invalid_reason",
]


def _likert_5pt_text_map() -> F.Column:
    """Map 5-point Likert text responses to `[1.0, 5.0]`."""
    return F.create_map(
        F.lit("strongly disagree"), F.lit(1.0),
        F.lit("disagree"), F.lit(2.0),
        F.lit("neutral"), F.lit(3.0),
        F.lit("agree"), F.lit(4.0),
        F.lit("strongly agree"), F.lit(5.0),
    )


def attach_standardized_labels_and_project(df: DataFrame) -> DataFrame:
    """Attach `main_dataset.csv` Q2 frequency and Q3..Q6 Likert labels and project to the unified schema."""
    df = (
        df.withColumn("ai_usage_freq", score_to_frequency_label("usage_freq_n"))
        .withColumn("aware_energy", score_to_likert_label("aware_energy_n"))
        .withColumn("aware_carbon", score_to_likert_label("aware_carbon_n"))
        .withColumn("consider_impact", score_to_likert_label("consider_impact_n"))
        .withColumn("avoid_unnecessary_use", score_to_likert_label("avoid_use_n"))
    )
    return df.select(*UNIFIED_MAIN_SCHEMA_COLS)


def integrate_main_study_ph(spark: SparkSession) -> DataFrame:
    """
    Load our Main Basis Dataset (`main_dataset.csv`):
    "AI and the Environment: An Analysis of Public Awareness" (Philippines).
    Removes unrelated columns (`Timestamp`, `Username`) and dynamically flags:
      - rows where mandatory Part IV open-ended Q7 & Q8 are both blank
      - rows where Q7 & Q8 are placeholder "." only
      - rows where Q7 ("What environmental impact of AI concerns you the most?")
        is off-topic / not related to the environment.
    """
    raw_df = sanitize_df_columns(
        read_csv_spark(spark, os.path.join(DATASET_DIR, "main_dataset.csv"))
    )
    df = raw_df.withColumn("local_id", F.monotonically_increasing_id() + 1)

    likert_map = _likert_5pt_text_map()
    freq_map = F.create_map(
        F.lit("rarely"), F.lit(1.0),
        F.lit("once a week"), F.lit(2.0),
        F.lit("several times a week"), F.lit(3.0),
        F.lit("once a day"), F.lit(4.0),
        F.lit("several times a day"), F.lit(5.0),
    )

    q7_col = "7_What_environmental_impact_of_Artificial_Intelligence_concerns_you_the_most_and_why"
    q8_col = "8_What_do_you_think_should_be_done_to_increase_public_awareness_and_encourage_responsible_use_of_AI"
    consent_col = [c for c in df.columns if c.startswith("I_have_read_and_understood")][0]

    q7_trim = F.trim(F.coalesce(F.col(q7_col), F.lit("")))
    q8_trim = F.trim(F.coalesce(F.col(q8_col), F.lit("")))
    both_blank = (q7_trim == "") & (q8_trim == "")
    both_dot_or_punct = (~both_blank) & (q7_trim.isin("", ".", "-")) & (q8_trim.isin("", ".", "-"))
    non_consent = ~F.lower(F.coalesce(F.col(consent_col), F.lit(""))).contains("yes")

    # Verify Q7 & Q8 actually address our main topic (AI environmental impacts)
    combined_q7_q8_lower = F.lower(F.concat_ws(" ", q7_trim, q8_trim))
    is_off_topic_q7 = (
        (~both_blank)
        & (~both_dot_or_punct)
        & (~combined_q7_q8_lower.rlike(MAIN_TOPIC_ENV_PATTERN))
    )

    df = (
        df.withColumn("respondent_id", F.concat(F.lit("MAIN_PH_"), F.lpad(F.col("local_id").cast("string"), 4, "0")))
        .withColumn("source_study", F.lit("Main_Study_PH_Env_Awareness"))
        .withColumn("study_role", F.lit("Main Study (PH)"))
        .withColumn(
            "age_group",
            F.when(F.trim(F.col("1_What_is_your_age")) == "24-34", "25-34")
            .when(F.trim(F.col("1_What_is_your_age")).isin("18-24", "25-34", "35-44", "45-54", "55+"), F.trim(F.col("1_What_is_your_age")))
            .otherwise("18-24"),
        )
        .withColumn("usage_freq_n", freq_map[F.lower(F.trim(F.col("2_How_often_do_you_use_Artificial_Intelligence_tools")))])
        .withColumn("aware_energy_n", likert_map[F.lower(F.trim(F.col("3_I_am_aware_that_using_AI_systems_consumes_significant_amounts_of_electricity_or_energy")))])
        .withColumn("aware_carbon_n", likert_map[F.lower(F.trim(F.col("4_I_am_aware_that_AI_data_centers_can_contribute_to_carbon_emissions_and_other_environmental_impacts")))])
        .withColumn("consider_impact_n", likert_map[F.lower(F.trim(F.col("5_I_consider_the_environmental_impact_of_AI_when_deciding_how_I_use_AI_tools")))])
        .withColumn("avoid_use_n", likert_map[F.lower(F.trim(F.col("6_I_avoid_unnecessary_use_of_AI_when_the_task_can_be_completed_without_it")))])
        .withColumn("concern_text", F.col(q7_col))
        .withColumn("awareness_text", F.col(q8_col))
        .withColumn(
            "is_valid_record",
            F.when(non_consent | both_blank | both_dot_or_punct | is_off_topic_q7, F.lit(0)).otherwise(F.lit(1)),
        )
        .withColumn(
            "invalid_reason",
            F.when(non_consent, F.lit("Did Not Consent to Participate"))
            .when(both_blank, F.lit("Missing Mandatory Open-Ended Q7 & Q8 (Both Blank)"))
            .when(both_dot_or_punct, F.lit("Non-Substantive Open-Ended Q7 & Q8 (Placeholder '.' Only)"))
            .when(is_off_topic_q7, F.lit("Not Related to Main Topic (Off-Topic Q7: Hacking, Privacy, or AGI Instead of Environment)"))
            .otherwise(F.lit("Valid")),
        )
    )
    log("Harmonized Main Basis Dataset: Main_Study_PH_Env_Awareness (main_dataset.csv)")
    return attach_standardized_labels_and_project(df)


def integrate_public_bd_cognitive(spark: SparkSession) -> DataFrame:
    """
    Map Public Study S2 (`_(AI_Cognitive_Dependence_Survey_Data_BD_StudentsResponses) - Sheet1.csv`)
    onto our `docs/` Q1–Q8 schema.
    Removes unrelated columns (`Gender`, `University Type`, `Academic Level`,
    `Department/Discipline`, `Trust in AI` grading, `Academic Decision-Making`)
    and dynamically flags duplicate and straight-line uniform survey submissions.
    """
    s2_raw = os.path.join(DATASET_DIR, "_(AI_Cognitive_Dependence_Survey_Data_BD_StudentsResponses) - Sheet1.csv")
    s2_staged = os.path.join(STAGING_DIR, "cognitive_dependence_bd.csv")
    stage_underscore_csv(s2_raw, s2_staged)

    raw_df = sanitize_df_columns(read_csv_spark(spark, s2_staged))
    raw_cols = raw_df.columns
    df = raw_df.withColumn("local_id", F.monotonically_increasing_id() + 1)

    dup_win = Window.partitionBy(*[F.col(c) for c in raw_cols]).orderBy("local_id")
    df = df.withColumn("dup_rank", F.row_number().over(dup_win))

    likert_map = _likert_5pt_text_map()
    all_25_likert_cols = [
        c for c in raw_cols
        if c.startswith(("AI_Usage_", "Cognitive_Dependence_", "Trust_in_AI_", "Critical_Thinking_", "Academic_Decision_Making_"))
    ]
    usage_cols = [c for c in raw_cols if c.startswith("AI_Usage_")]
    dep_cols = [c for c in raw_cols if c.startswith("Cognitive_Dependence_")]
    crit_cols = [c for c in raw_cols if c.startswith("Critical_Thinking_")]

    # Detect straight-line uniform responses across all 25 items
    first_item = F.lower(F.trim(F.col(all_25_likert_cols[0])))
    is_straight_liner = F.lit(True)
    for c in all_25_likert_cols[1:]:
        is_straight_liner = is_straight_liner & (F.lower(F.trim(F.col(c))) == first_item)

    for idx, c in enumerate(usage_cols):
        df = df.withColumn(f"u_num_{idx}", likert_map[F.lower(F.trim(F.col(c)))])
    for idx, c in enumerate(dep_cols):
        df = df.withColumn(f"d_num_{idx}", likert_map[F.lower(F.trim(F.col(c)))])
        df = df.withColumn(
            f"d_inv_{idx}",
            F.when(F.col(f"d_num_{idx}").isNotNull(), REVERSE_5PT_OFFSET - F.col(f"d_num_{idx}")).otherwise(None),
        )
    for idx, c in enumerate(crit_cols):
        df = df.withColumn(f"c_num_{idx}", likert_map[F.lower(F.trim(F.col(c)))])
        df = df.withColumn(
            f"c_inv_{idx}",
            F.when(F.col(f"c_num_{idx}").isNotNull(), REVERSE_5PT_OFFSET - F.col(f"c_num_{idx}")).otherwise(None),
        )

    age_num = F.col("Age").cast("double")
    df = (
        df.withColumn("respondent_id", F.concat(F.lit("PUB_BD_"), F.lpad(F.col("local_id").cast("string"), 4, "0")))
        .withColumn("source_study", F.lit("Public_S2_BD_Cognitive_Dependence"))
        .withColumn("study_role", F.lit("Public Study"))
        .withColumn(
            "age_group",
            F.when(age_num < 18, "18-24")
            .when((age_num >= 18) & (age_num <= 24), "18-24")
            .when((age_num >= 25) & (age_num <= 34), "25-34")
            .when(age_num >= 35, "35-44")
            .otherwise("18-24"),
        )
        # Only use items directly related to AI usage frequency, impact awareness, and avoiding unnecessary AI dependency
        .withColumn("usage_freq_n", row_mean_null_safe([f"u_num_{i}" for i in range(len(usage_cols))]))
        .withColumn("aware_energy_n", row_mean_null_safe(["c_inv_0", "c_inv_1", "c_inv_2"]))
        .withColumn("aware_carbon_n", row_mean_null_safe(["c_inv_2", "c_inv_3", "c_inv_4"]))
        .withColumn("consider_impact_n", row_mean_null_safe(["c_inv_0", "c_inv_1", "c_inv_3", "d_inv_1", "d_inv_3"]))
        .withColumn("avoid_use_n", row_mean_null_safe(["d_inv_0", "d_inv_1", "d_inv_2", "d_inv_3", "d_inv_4"]))
        .withColumn("concern_text", F.lit(None).cast("string"))
        .withColumn("awareness_text", F.lit(None).cast("string"))
        .withColumn(
            "is_valid_record",
            F.when((F.col("dup_rank") > 1) | is_straight_liner, F.lit(0)).otherwise(F.lit(1)),
        )
        .withColumn(
            "invalid_reason",
            F.when(F.col("dup_rank") > 1, F.lit("Exact Duplicate Respondent Record (All 30 Columns Identical)"))
            .when(is_straight_liner, F.lit("Invalid Straight-Line Uniform Response (All 25 Items Identical)"))
            .otherwise(F.lit("Valid")),
        )
    )
    log("Harmonized Public Study S2: Public_S2_BD_Cognitive_Dependence (unrelated academic cols removed)")
    return attach_standardized_labels_and_project(df)


def integrate_public_utaut_factors(spark: SparkSession) -> DataFrame:
    """
    Map Public Study S3 (`Dataset_Factors Affecting Responsible Generative AI Use Among University Students.xlsx`)
    onto our `docs/` Q1–Q8 schema.
    Removes unrelated UTAUT marketing/adoption columns (`PE`, `IQ`, `SN`, `HB`, `TR`, `BI`)
    and keeps only actual usage (`UB1..UB3`), AI literacy/awareness (`AL1..AL7`), and
    moral/responsible AI obligation (`MO1..MO3`).
    """
    s3_xlsx = os.path.join(DATASET_DIR, "Dataset_Factors Affecting Responsible Generative AI Use Among University Students.xlsx")
    s3_staged = os.path.join(STAGING_DIR, "factors_responsible_genai.csv")
    xlsx_sheet_to_csv(s3_xlsx, "TEST1", s3_staged)

    raw_df = read_csv_spark(spark, s3_staged)
    raw_cols = raw_df.columns
    df = raw_df.withColumn("local_id", F.monotonically_increasing_id() + 1)
    dup_win = Window.partitionBy(*[F.col(c) for c in raw_cols]).orderBy("local_id")
    df = df.withColumn("dup_rank", F.row_number().over(dup_win))

    df = (
        df.withColumn("respondent_id", F.concat(F.lit("PUB_UTAUT_"), F.lpad(F.col("local_id").cast("string"), 4, "0")))
        .withColumn("source_study", F.lit("Public_S3_UTAUT_Responsible_GenAI"))
        .withColumn("study_role", F.lit("Public Study"))
        .withColumn("age_group", F.lit("18-24"))
        # Only retain items related to AI Usage Frequency (UB), AI Awareness/Literacy (AL), and Responsible AI Use (MO)
        .withColumn("usage_freq_n", row_mean_null_safe(["UB1", "UB2", "UB3"]))
        .withColumn("aware_energy_n", row_mean_null_safe(["AL1", "AL2", "AL3", "AL4"]))
        .withColumn("aware_carbon_n", row_mean_null_safe(["AL5", "AL6", "AL7"]))
        .withColumn("consider_impact_n", row_mean_null_safe(["MO1", "MO2", "MO3"]))
        .withColumn("avoid_use_n", row_mean_null_safe(["MO1", "MO2", "MO3", "AL1", "AL2"]))
        .withColumn("concern_text", F.lit(None).cast("string"))
        .withColumn("awareness_text", F.lit(None).cast("string"))
        .withColumn("is_valid_record", F.when(F.col("dup_rank") > 1, F.lit(0)).otherwise(F.lit(1)))
        .withColumn(
            "invalid_reason",
            F.when(F.col("dup_rank") > 1, F.lit("Exact Duplicate Respondent Record (All 31 Columns Identical)")).otherwise(F.lit("Valid")),
        )
    )
    log("Harmonized Public Study S3: Public_S3_UTAUT_Responsible_GenAI (unrelated UTAUT cols removed)")
    return attach_standardized_labels_and_project(df)


def _prepare_topic_related_liverpool_freetext(spark: SparkSession) -> tuple[DataFrame, DataFrame]:
    """
    Filter `Anon Feee text.xlsx` (`Student text`, `Staff text`)
    so ONLY open-ended responses related to our main topic (AI environmental impacts
    or responsible AI restraint / public awareness guidelines) are kept.
    Unrelated tool lists (`S1`: "ChatGPT"), homework tasks (`S2`: "flashcards"),
    everyday hobbies (`S4`: "recipes"), and exam grading rules (`S5`, `T1..T8`) are removed.
    """
    s5_xlsx = os.path.join(LIVERPOOL_DIR, "Anon Feee text.xlsx")
    s5_stu_csv = os.path.join(STAGING_DIR, "freetext_students.csv")
    s5_stf_csv = os.path.join(STAGING_DIR, "freetext_staff.csv")
    xlsx_sheet_to_csv(s5_xlsx, "Student text", s5_stu_csv)
    xlsx_sheet_to_csv(s5_xlsx, "Staff text", s5_stf_csv)

    combined_topic_regex = f"({MAIN_TOPIC_ENV_PATTERN}|{MAIN_TOPIC_RESPONSIBLE_USE_PATTERN})"

    def _filter_topic_text(csv_path: str, allowed_codes: List[str], label: str) -> DataFrame:
        raw_t = read_csv_spark(spark, csv_path).withColumn("seq_id", F.monotonically_increasing_id() + 1)
        raw_cnt = raw_t.count()
        norm = F.trim(F.regexp_replace(F.coalesce(F.col("Response"), F.lit("")), r"\s+", " "))
        lower_norm = F.lower(norm)

        topic_t = (
            raw_t.filter(
                (norm != "")
                & (F.length(norm) >= 3)
                & (~lower_norm.isin(JUNK_TEXT_TOKENS))
                & F.col("QuestionCode").isin(*allowed_codes)
                & lower_norm.rlike(combined_topic_regex)
            )
            .withColumn("clean_resp", norm)
            .withColumn("is_env_concern", lower_norm.rlike(MAIN_TOPIC_ENV_PATTERN))
        )
        kept_cnt = topic_t.count()
        log(
            f"Filtered {label} Free-Text for Main Topic: {raw_cnt} raw -> {kept_cnt} topic-related kept "
            f"({raw_cnt - kept_cnt} unrelated tool/homework/grading responses removed)"
        )

        w_seq = Window.orderBy("seq_id")
        return topic_t.withColumn("topic_idx", F.row_number().over(w_seq)).select(
            "topic_idx",
            F.when(F.col("is_env_concern"), F.col("clean_resp")).otherwise(F.lit(None).cast("string")).alias("mapped_concern_text"),
            F.when(~F.col("is_env_concern"), F.col("clean_resp")).otherwise(F.col("clean_resp")).alias("mapped_awareness_text"),
        )

    stu_text_df = _filter_topic_text(s5_stu_csv, ["S3", "S6"], "Liverpool Student")
    stf_text_df = _filter_topic_text(s5_stf_csv, ["T2", "T3", "T4", "T9"], "Liverpool Staff")
    return stu_text_df, stf_text_df


def integrate_public_liverpool_students(spark: SparkSession, stu_text_df: DataFrame) -> DataFrame:
    """
    Map Public Study S4A (`Survey Data.xlsx -> Students`) + topic-filtered
    `Student text` onto our `docs/` Q1–Q8 schema.
    Removes all unrelated university exam/plagiarism/faculty columns and dynamically flags:
      - rows not related to main topic (exam-cheating items only; missing `SR_SelfRegulationMean`)
      - rows that failed the embedded survey Attention Check (`AttentionCheckPass = 0`)
      - rows that are incomplete / early drop-outs (`AI_FrequencyMean` & `Guidance_Read` null).
    """
    s4_xlsx = os.path.join(LIVERPOOL_DIR, "Survey Data.xlsx")
    s4_stu_csv = os.path.join(STAGING_DIR, "liverpool_students.csv")
    xlsx_sheet_to_csv(s4_xlsx, "Students ", s4_stu_csv)

    df = read_csv_spark(spark, s4_stu_csv).withColumn("case_idx", F.col("CaseID").cast("int"))

    failed_attn = F.coalesce(F.col("AttentionCheckPass").cast("double") == 0.0, F.lit(False))
    abandoned_stu = (
        (~failed_attn)
        & F.col("AI_FrequencyMean").cast("double").isNull()
        & F.col("Guidance_Read").cast("double").isNull()
    )
    unrelated_exam_only = (
        (~failed_attn)
        & (~abandoned_stu)
        & F.col("SR_SelfRegulationMean").cast("double").isNull()
    )

    df = df.withColumn(
        "is_valid_record",
        F.when(failed_attn | abandoned_stu | unrelated_exam_only, F.lit(0)).otherwise(F.lit(1)),
    )

    w_valid = Window.partitionBy("is_valid_record").orderBy("case_idx")
    df = df.withColumn(
        "valid_rank",
        F.when(F.col("is_valid_record") == 1, F.row_number().over(w_valid)).otherwise(F.lit(-1)),
    )
    df = df.join(stu_text_df, df["valid_rank"] == stu_text_df["topic_idx"], how="left").drop("topic_idx", "valid_rank", "case_idx")

    df = df.withColumn(
        "outside_inv",
        F.when(F.col("AI_Outside").cast("double").between(1, 5), REVERSE_5PT_OFFSET - F.col("AI_Outside").cast("double")).otherwise(None),
    )

    df = (
        df.withColumn("guide_read_5pt", F.when(F.col("Guidance_Read").cast("int") == 1, 5.0).otherwise(2.0))
        .withColumn("guide_known_5pt", F.when(F.col("Guidance_ContentKnown").cast("int") == 1, 5.0).otherwise(2.0))
        .withColumn(
            "dept_guide_5pt",
            F.when(F.col("DeptGuidance").cast("int") == 1, 5.0)
            .when(F.col("DeptGuidance").cast("int") == 2, 3.5)
            .when(F.col("DeptGuidance").cast("int") == 3, 2.0)
            .otherwise(None),
        )
    )

    df = (
        df.withColumn("respondent_id", F.concat(F.lit("PUB_LIV_STU_"), F.lpad(F.col("CaseID").cast("string"), 4, "0")))
        .withColumn("source_study", F.lit("Public_S4A_UK_Liverpool_Students"))
        .withColumn("study_role", F.lit("Public Study"))
        .withColumn("age_group", F.lit("18-24"))
        .withColumn("usage_freq_n", row_mean_null_safe(["AI_FrequencyMean", "outside_inv"]))
        .withColumn("aware_energy_n", row_mean_null_safe(["guide_read_5pt", "guide_known_5pt", "SR_SelfRegulationMean"]))
        .withColumn("aware_carbon_n", row_mean_null_safe(["dept_guide_5pt", "AIProhibited_Understood", "SR_SelfRegulationMean"]))
        .withColumn("consider_impact_n", row_mean_null_safe(["SR_SelfRegulationMean", "AIProhibited_Understood"]))
        .withColumn("avoid_use_n", row_mean_null_safe(["SR_SelfRegulationMean", "guide_known_5pt"]))
        .withColumn("concern_text", F.col("mapped_concern_text"))
        .withColumn("awareness_text", F.col("mapped_awareness_text"))
        .withColumn(
            "invalid_reason",
            F.when(failed_attn, F.lit("Failed Embedded Survey Attention Check (AttentionCheckPass = 0)"))
            .when(abandoned_stu, F.lit("Incomplete Questionnaire / Early Drop-out (Missing Usage & Awareness)"))
            .when(unrelated_exam_only, F.lit("Not Related to Main Topic (Exam-Cheating Items Only; Missing Responsible Self-Regulation)"))
            .otherwise(F.lit("Valid")),
        )
    )
    log("Harmonized Public Study S4A: Public_S4A_UK_Liverpool_Students")
    return attach_standardized_labels_and_project(df)


def integrate_public_liverpool_staff(spark: SparkSession, stf_text_df: DataFrame) -> DataFrame:
    """
    Map Public Study S4B (`Survey Data.xlsx -> Staff`) + topic-filtered
    `Staff text` onto our `docs/` Q1–Q8 schema.
    Dynamically flags:
      - incomplete / early drop-out rows (all core policy/guidance items null)
      - rows not related to our main topic (staff who only answered student-grading
        and plagiarism observation checklists without any topic-relevant environmental
        or responsible AI guidance response).
    """
    s4_xlsx = os.path.join(LIVERPOOL_DIR, "Survey Data.xlsx")
    s4_stf_csv = os.path.join(STAGING_DIR, "liverpool_staff.csv")
    xlsx_sheet_to_csv(s4_xlsx, "Staff", s4_stf_csv)

    df = read_csv_spark(spark, s4_stf_csv).withColumn("case_idx", F.col("CaseID").cast("int"))

    abandoned_stf = (
        F.col("PolicyAware").isNull()
        & F.col("GuidanceAware").isNull()
        & F.col("AcceptableExtent").isNull()
    )
    df = df.withColumn("is_non_abandoned", F.when(abandoned_stf, F.lit(0)).otherwise(F.lit(1)))

    w_non_aband = Window.partitionBy("is_non_abandoned").orderBy("case_idx")
    df = df.withColumn(
        "non_aband_rank",
        F.when(F.col("is_non_abandoned") == 1, F.row_number().over(w_non_aband)).otherwise(F.lit(-1)),
    )
    df = df.join(stf_text_df, df["non_aband_rank"] == stf_text_df["topic_idx"], how="left")

    has_topic_text = F.col("topic_idx").isNotNull()
    unrelated_grading_only = (~abandoned_stf) & (~has_topic_text)

    agree_unacc_cols = [c for c in df.columns if c.startswith("AgreeUnacc_")]
    df = (
        df.withColumn("agree_unacc_5pt", F.lit(1.0) + F.lit(4.0) * row_mean_null_safe(agree_unacc_cols))
        .withColumn("instructed_use_5pt", F.when(F.col("InstructedAIUse").cast("int") == 1, 4.0).otherwise(2.0))
        .withColumn("accept_ext_5pt", F.lit(1.0) + F.col("AcceptableExtent").cast("double"))
        .withColumn("pol_aware_5pt", F.when(F.col("PolicyAware").cast("int") == 1, 4.5).otherwise(2.0))
        .withColumn("pol_help_5pt", F.when(F.col("PolicyHelpful").cast("int") == 1, 4.5).otherwise(2.5))
        .withColumn("guide_aware_5pt", F.when(F.col("GuidanceAware").cast("int") == 1, 4.5).otherwise(2.0))
        .withColumn("guide_help_5pt", F.when(F.col("GuidanceHelpful").cast("int") == 1, 4.5).otherwise(2.5))
    )

    df = (
        df.withColumn("respondent_id", F.concat(F.lit("PUB_LIV_STF_"), F.lpad(F.col("CaseID").cast("string"), 4, "0")))
        .withColumn("source_study", F.lit("Public_S4B_UK_Liverpool_Staff"))
        .withColumn("study_role", F.lit("Public Study"))
        .withColumn("age_group", F.lit("35-44"))
        .withColumn("usage_freq_n", row_mean_null_safe(["instructed_use_5pt", "accept_ext_5pt"]))
        .withColumn("aware_energy_n", row_mean_null_safe(["pol_aware_5pt", "guide_aware_5pt"]))
        .withColumn("aware_carbon_n", row_mean_null_safe(["pol_help_5pt", "guide_help_5pt"]))
        .withColumn("consider_impact_n", row_mean_null_safe(["agree_unacc_5pt", "pol_aware_5pt"]))
        .withColumn("avoid_use_n", row_mean_null_safe(["agree_unacc_5pt", "guide_aware_5pt"]))
        .withColumn("concern_text", F.col("mapped_concern_text"))
        .withColumn("awareness_text", F.col("mapped_awareness_text"))
        .withColumn(
            "is_valid_record",
            F.when(abandoned_stf | unrelated_grading_only, F.lit(0)).otherwise(F.lit(1)),
        )
        .withColumn(
            "invalid_reason",
            F.when(abandoned_stf, F.lit("Incomplete Questionnaire / Early Drop-out (All Core Items Null)"))
            .when(unrelated_grading_only, F.lit("Not Related to Main Topic (Staff Grading & Plagiarism Checklist Only)"))
            .otherwise(F.lit("Valid")),
        )
    )
    log("Harmonized Public Study S4B: Public_S4B_UK_Liverpool_Staff")
    return attach_standardized_labels_and_project(df)


def export_lost_rows_audit(df_unified: DataFrame) -> None:
    """
    Compute, display, and save how many rows were cleaned vs. lost/removed in each study
    during integration/cleaning and the exact reason why.
    """
    reasons_df = (
        df_unified.filter(F.col("is_valid_record") == 0)
        .groupBy("source_study", "study_role", "invalid_reason")
        .agg(F.count("*").alias("rows_lost_for_reason"))
        .orderBy("source_study", F.desc("rows_lost_for_reason"))
    )
    write_spark_df_to_csv(reasons_df, STAGE1_LOST_REASONS_CSV)

    reasons_collapsed = (
        reasons_df.withColumn(
            "reason_with_count",
            F.concat(F.col("invalid_reason"), F.lit(" (n="), F.col("rows_lost_for_reason").cast("string"), F.lit(")")),
        )
        .groupBy("source_study")
        .agg(F.concat_ws("; ", F.collect_list("reason_with_count")).alias("why_rows_lost"))
    )

    summary_df = (
        df_unified.groupBy("source_study", "study_role")
        .agg(
            F.count("*").alias("raw_rows"),
            F.sum(F.col("is_valid_record")).alias("cleaned_rows"),
            F.sum(F.when(F.col("is_valid_record") == 0, 1).otherwise(0)).alias("rows_lost"),
        )
        .withColumn("cleaned_pct", F.round(F.col("cleaned_rows") * 100.0 / F.col("raw_rows"), 2))
        .withColumn("lost_pct", F.round(F.col("rows_lost") * 100.0 / F.col("raw_rows"), 2))
        .join(reasons_collapsed, on="source_study", how="left")
        .fillna({"why_rows_lost": "No rows lost (100% valid)"})
        .orderBy("source_study")
    )
    write_spark_df_to_csv(summary_df, STAGE1_LOST_SUMMARY_CSV)

    banner("TOPIC-ALIGNED INTEGRATION & CLEANING AUDIT: ROWS LOST/REMOVED PER STUDY AND WHY")
    for r in summary_df.collect():
        print(
            f"  * {r['source_study']} [{r['study_role']}]:\n"
            f"      Raw = {r['raw_rows']:,} | Cleaned = {r['cleaned_rows']:,} ({r['cleaned_pct']:.2f}%) | "
            f"Lost/Removed = {r['rows_lost']:,} ({r['lost_pct']:.2f}%)\n"
            f"      Why Lost/Removed: {r['why_rows_lost']}"
        )


def main() -> None:
    os.makedirs(INTEGRATION_DIR, exist_ok=True)
    os.makedirs(STAGING_DIR, exist_ok=True)
    spark = create_spark_session("DSA-Final-01-Topic-Aligned-Integration")

    banner("STAGE 1: TOPIC-ALIGNED MULTI-SOURCE INTEGRATION (BASIS: `main_dataset.csv` & `docs/`)")

    stu_text_df, stf_text_df = _prepare_topic_related_liverpool_freetext(spark)

    df_main = integrate_main_study_ph(spark)
    df_s2 = integrate_public_bd_cognitive(spark)
    df_s3 = integrate_public_utaut_factors(spark)
    df_s4a = integrate_public_liverpool_students(spark, stu_text_df)
    df_s4b = integrate_public_liverpool_staff(spark, stf_text_df)

    df_unified = (
        df_main.unionByName(df_s2)
        .unionByName(df_s3)
        .unionByName(df_s4a)
        .unionByName(df_s4b)
    )
    write_spark_df_to_csv(df_unified, STAGE1_UNIFIED_CSV)
    export_lost_rows_audit(df_unified)

    spark.stop()
    log("Stage 1 (Topic-Aligned Single Unified Dataset Integration) finished successfully.")


if __name__ == "__main__":
    main()

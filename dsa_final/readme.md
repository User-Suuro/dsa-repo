# DSA Final — "AI and the Environment: An Analysis of Public Awareness" (Topic-Aligned Unified PySpark Pipeline)

**Authors:** Renz Dominic Azul, Gabriel de la Cruz, Godwin Reigh Galvez (BSIT 3B, IT 119 - Data Scalability and Analytics)

This directory implements a 5-stage **PySpark Data Integration, Cleansing, Transformation, Reduction, and Analytics** pipeline strictly revolved around our study proposal ([DSA - Proposal.pdf](file:///c:/Coding%20Projects/Python%20Projects/dsa_final/docs/DSA%20-%20Proposal.pdf)) and transformation specification ([Data Scalability and Analytics - Data transformation (Pyspark)  (1).pdf](file:///c:/Coding%20Projects/Python%20Projects/dsa_final/docs/Data%20Scalability%20and%20Analytics%20-%20Data%20transformation%20(Pyspark)%20%20(1).pdf)), using our Primary Philippine Community Survey ([main_dataset.csv](file:///c:/Coding%20Projects/Python%20Projects/dsa_final/dataset/main_dataset.csv), `Q1`–`Q8`) as the canonical schema basis.

All unrelated columns, non-environmental open-ended themes, and off-topic rows across both `main_dataset.csv` and the public datasets are removed during integration and cleaning.

## 1. Canonical Schema (`docs/` & `main_dataset.csv` Basis)

Every record in the single unified dataset is mapped strictly to the 8 questionnaire items and composite constructs defined in `docs/`:
- **`age_group` (`Q1`)**: Respondent age band (`Under 18`, `18-24`, `25-34`, `35 and above`).
- **`ai_usage_freq` / `usage_freq_n` (`Q2`)**: AI usage frequency on a `[1.0, 5.0]` scale (`1=Rarely` to `5=Several times a day`).
- **`aware_energy` / `aware_energy_n` (`Q3`)**: Awareness that AI consumes significant electricity/energy (`[1.0, 5.0]`).
- **`aware_carbon` / `aware_carbon_n` (`Q4`)**: Awareness that AI data centers contribute to carbon emissions and environmental impacts (`[1.0, 5.0]`).
- **`consider_impact` / `consider_impact_n` (`Q5`)**: Considering environmental impact when using AI tools (`[1.0, 5.0]`).
- **`avoid_unnecessary_use` / `avoid_use_n` (`Q6`)**: Willingness to avoid unnecessary AI usage to reduce environmental impact (`[1.0, 5.0]`).
- **`concern_text` (`Q7`)**: Open-ended environmental impact concern (`c_water`, `c_energy`, `c_carbon`, `c_ewaste`, `c_noise_heat`, `c_ecosystem`).
- **`awareness_text` (`Q8`)**: Open-ended suggestion for improving public awareness and responsible AI usage (`s_education`, `s_campaign`, `s_policy`, `s_transparency`, `s_tech`, `s_responsible_use`).
- **Composite Indices & Segments (`docs/`)**: `Awareness_Index = (Q3 + Q4) / 2`, `Behavior_Index = (Q5 + Q6) / 2`, `Awareness_Behavior_Gap = Awareness_Index - Behavior_Index`, `usage_segment`, `awareness_level`, `behavior_level`, and `gap_type` (`Unaware & not acting`, `Aware & taking action`, `Aware but not acting`, `Acting beyond awareness`).

## 2. Rows Cleaned vs. Lost / Removed as Unrelated Per Study

| Study ID | Source File(s) | Role | Raw Rows | Cleaned Topic-Valid Rows | Rows Lost / Removed | Retention % | Loss % | Detailed Reasons Why Rows Were Lost or Removed as Unrelated |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Main_Study_PH_Env_Awareness`** | `main_dataset.csv` | **Main Study Basis (PH)** | **84** | **72** | **12** | **85.71%** | **14.29%** | `7` rows: Not Related to Main Topic (Off-Topic Q7: Hacking, Data Privacy, Sci-Fi AGI Sentience, or Misinformation Instead of Environmental Impact)<br>`4` rows: Missing Mandatory Open-Ended Q7 & Q8 (Both Blank)<br>`1` row: Non-Substantive Open-Ended Q7 & Q8 (Placeholder `"."` Only) |
| `Public_S2_BD_Cognitive_Dependence` | `_(AI_Cognitive_Dependence_Survey_Data_BD_StudentsResponses) - Sheet1.csv` | Public Study (BD) | 2,613 | 2,087 | 526 | 79.87% | 20.13% | `451` rows: Exact Duplicate Respondent Record (All 30 Columns Identical)<br>`75` rows: Invalid Straight-Line Uniform Response (All 25 Items Identical) |
| `Public_S3_UTAUT_Responsible_GenAI` | `Dataset_Factors Affecting Responsible Generative AI Use Among University Students.xlsx` | Public Study (Global) | 400 | 399 | 1 | 99.75% | 0.25% | `1` row: Exact Duplicate Respondent Record (All 31 Columns Identical) |
| `Public_S4A_UK_Liverpool_Students` | `Survey Data.xlsx [Students]` + `Anon Feee text.xlsx [Student text]` | Public Study (UK) | 519 | 262 | 257 | 50.48% | 49.52% | `189` rows: Not Related to Main Topic (Answered Exam-Cheating/Assessment Items Only; Missing Responsible Self-Regulation `SR_SelfRegulationMean`)<br>`41` rows: Failed Embedded Survey Attention Check (`AttentionCheckPass = 0`)<br>`27` rows: Incomplete Questionnaire / Early Drop-out (Missing Usage & Awareness) |
| `Public_S4B_UK_Liverpool_Staff` | `Survey Data.xlsx [Staff]` + `Anon Feee text.xlsx [Staff text]` | Public Study (UK) | 150 | 22 | 128 | 14.67% | 85.33% | `92` rows: Not Related to Main Topic (Staff Grading & Plagiarism Checklist Only — No Environmental or Responsible AI Use Data)<br>`36` rows: Incomplete Questionnaire / Early Drop-out (All Core Items Null) |
| **TOTAL UNIFIED DATASET** | **All 5 Studies Integrated into One Dataset** | **Combined** | **3,766** | **2,842** | **924** | **75.46%** | **24.54%** | See `output/analysis/study_rows_lost_and_reasons.csv` |

## 3. Pipeline Execution Order (`output/`)

```powershell
cd "c:\Coding Projects\Python Projects\dsa_final"

.\.venv\Scripts\python.exe output\01_data_integration.py
.\.venv\Scripts\python.exe output\02_data_cleansing.py
.\.venv\Scripts\python.exe output\03_data_transformation.py
.\.venv\Scripts\python.exe output\04_data_reduction.py
.\.venv\Scripts\python.exe output\05_data_analysis_and_visualization.py
```

## 4. Output Folder Structure (`output/`)

| Stage / Module | Script | Output Folder | Generated Files |
| :--- | :--- | :--- | :--- |
| **Shared Utilities** | `output/pipeline_utils.py` | — | Shared `SparkSession` factory, single-dataset path constants, `MAIN_ITEM_SCORE_COLS` (`usage_freq_n..avoid_use_n`), `Q7_CONCERN_THEME_COLS` (`c_*`), `Q8_SOLUTION_THEME_COLS` (`s_*`), `.xlsx` XML staging, `_` CSV staging, and Windows-safe `toLocalIterator()` CSV writers |
| **Stage 1: Data Integration** | `output/01_data_integration.py` | `output/integration/` | `stage1_integrated_dataset.csv` (`3,766 × 18`), `stage1_lost_rows_by_study.csv`, `stage1_lost_rows_reasons_detail.csv`, and `staging/*.csv` |
| **Stage 2: Data Cleansing** | `output/02_data_cleansing.py` | `output/cleansing/` | `stage2_cleansed_dataset.csv` (`2,842 × 18`), `stage2_cleaning_loss_log.csv` |
| **Stage 3: Data Transformation** | `output/03_data_transformation.py` | `output/transformation/` | `stage3_transformed_dataset.csv` (`2,842 × 46`, composite indices, segments, z-scores, and 12 environmental/responsible-use NLP themes) |
| **Stage 4: Data Reduction** | `output/04_data_reduction.py` | `output/reduction/` | `integrated_master_dataset.csv` (`2,842 × 49`, PCA 85.81% variance), `integrated_master_dataset_sampled50.csv` (`1,454 × 49`) |
| **Stage 5: Data Analysis & Visualization** | `output/05_data_analysis_and_visualization.py` | `output/analysis/` | `study_rows_lost_and_reasons.csv`, `integration_summary_by_source.csv`, `analysis1..8_*.csv`, `key_insights.txt`, and `analysis/figures/fig1..6_*.png` |

## 5. Generated Figures (`output/analysis/figures/`)

1. **`fig1_data_cleaning_summary.png`** — Data Integration & Cleaning Summary (`2,842` Cleaned vs. `924` Removed out of `3,766` Raw Rows across the 5 studies, plus simplified removal reasons).
2. **`fig2_unified_indices_by_study.png`** — Unified Dataset: Mean Construct Indices by Study (`Q2` Usage Frequency, `Awareness_Index`, and `Behavior_Index` across all 5 integrated studies, `N = 2,842`).
3. **`fig3_likert_diverging.png`** — Awareness vs. Behavior: Diverging Likert Responses (`Q3`–`Q6` across the unified dataset, `N = 2,842`).
4. **`fig4_index_by_usage.png`** — Awareness Index vs. Behavior Index by AI-Usage Segment (`Light n=58`, `Moderate n=875`, `Heavy n=1,909`, `N = 2,842`).
5. **`fig5_open_ended_themes.png`** — Open-Ended Qualitative Themes Across the Unified Dataset (`Q7` Environmental Concerns `n=123` text answers and `Q8` Suggested Solutions `n=151` text answers out of `N = 2,842`).
6. **`fig6_correlation_and_gap.png`** — Construct Correlations (`Q2`–`Q6`, `r = +0.7728` between Awareness & Behavior) & Awareness–Behavior Gap Alignment (`N = 2,842`).

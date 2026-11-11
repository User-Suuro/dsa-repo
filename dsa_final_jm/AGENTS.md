# AGENTS.md — `dsa_final_jm` (Assessing College Students Awareness and Preparedness Toward Flood Risks)

## Purpose
This directory implements the complete 5-stage PySpark Data Scalability & Analytics pipeline for the study:
**"Assessing College Students Awareness and Preparedness Toward Flood Risks"** (John Mark Pardo, Byron Boral, Aerold Steven Reyes — Group 11 - Skywalker).

It integrates the primary Camarines Norte college student survey ($N=128$) with three topically-aligned public datasets ($N=1,934$), achieving a unified dataset of $2,062$ raw records and **$1,891$ clean topic-valid records** (screened out $171$ inattentive/careless responses). The pipeline mirrors the exact architectural design, structure, and standards of `dsa_final`.

## Repository Map
- `code/data-cleaning/`: 5-stage PySpark pipeline + utilities:
  - `pipeline_utils.py`: Shared SparkSession factory, Windows path overrides, domain schema constants, and Windows-safe CSV persistence.
  - `01_data_integration.py`: Multi-source data ingestion, schema canonicalization, and loss logging.
  - `02_data_cleansing.py`: Invalid row screening (`is_valid_record == 1`), whitespace trimming, Likert clamping to $[1.0, 5.0]$, median imputation, and junk text stripping.
  - `03_data_transformation.py`: Composite construct indices (`awareness_score`, `preparedness_score`, `awareness_preparedness_gap`, sub-dimensions), behavioral segmentation (`gap_typology`, `risk_tier`), qualitative regex coding (`c_*`, `s_*`), and PySpark ML StandardScaler (`*_z`).
  - `04_data_reduction.py`: Zero-variance pruning, PySpark `PCA(k=3)` dimensionality reduction, and 50% stratified numerosity reduction.
  - `05_data_analysis_and_visualization.py`: 8 empirical analysis tables, loss audits, `key_insights.txt`, and 6 publication-grade figures.
- `code/supervised-ml-algo/`: 6 supervised models (`slr.py`, `mlr.py`, `decision_tree.py`, `knn.py`, `naive_bayes.py`, `svm.py`, `main.py`).
- `code/unsupervised-ml-algo/`: Unsupervised clustering models (`kmeans.py`, `mean_shift.py`, `main.py`).
- `docs/`: Academic documentation and proposal materials:
  - `Chapter_1_Introduction.md` (and `docs/output/Chapter_1_Introduction.md`): Complete Chapter 1 thesis introduction (Background, SOP, Objectives, IPO Conceptual Framework, Significance, Scope & Delimitations, Definition of Terms).
  - `Chapter_2_Review_of_Related_Literature.md` (and `docs/output/Chapter_2_Review_of_Related_Literature.md`): Complete Chapter 2 literature review (Thematic Literature, Foreign/Local Studies, Public Dataset Integration, Synthesis Table, Research Gaps, APA 7 References).
  - `Chapter_3_Methodology.md` (and `docs/output/Chapter_3_Methodology.md`): Complete Chapter 3 research methodology (Design, Sampling, 5-Stage PySpark Pipeline, Variables Table, Likert Scales, ML Modeling Framework, Tools & Reproducibility).
  - `Chapter_4_Results_and_Discussion.md` (and `docs/output/Chapter_4_Results_and_Discussion.md`): Complete Chapter 4 results and discussion (Data Cleansing Funnel, Demographic Profiles, Awareness & Preparedness Levels, Gap Typology, Cohort Analysis, Pearson Correlations, Open-Ended Themes, Unsupervised Personas, Supervised Predictive Models, Literature Synthesis, and 17 explicit figure callouts).
  - `dsa sa baha - Revised.pdf` (Original research proposal by Group 11 - Skywalker).
  - `Data_Transformation_and_Analysis_GROUP11-SKYWALKER (2).pdf` (Course specification and mapping).
- `dataset/`: Primary study data:
  - `Assessing College Students Awareness and Preparedness Toward Flood Risks.csv` (Camarines Norte college students, $N=128$).
- `dataset/public/`: Integrated public flood risk & DRRM datasets:
  - `Segovia_2026_PH_Young_Adults_Flood_Exposure/` (Mendeley Data DOI: 10.17632/462nvdfvjj.1, $N=541$ raw $\to$ $370$ clean young adults, mean age 21.0).
  - `FL24_Flooding_and_Society_Survey/` (Harvard Dataverse DOI: 10.7910/DVN/I9751R, $N=1,200$).
  - `Supplementary Appendix B DIB Ravago et al 2020/` (Data in Brief / Mendeley Data DOI: 10.17632/9y2yhw45zt.1, $N=193$).
  - `citation.txt`: Complete APA 7 citations and study mappings.
- `output/`: Modular stage output folders:
  - `output/data-cleaning/`:
    - `integration/`: Stage 1 unified dataset (`stage1_integrated_dataset.csv`, $2,062 \times 26$) and loss audits.
    - `cleansing/`: Stage 2 cleansed unified dataset (`stage2_cleansed_dataset.csv`, $1,891 \times 26$) and `stage2_cleaning_loss_log.csv`.
    - `transformation/`: Stage 3 transformed unified dataset (`stage3_transformed_dataset.csv`, $1,891 \times 44$).
    - `reduction/`: Stage 4 master datasets (**one big dataset for final output**):
      - `integrated_master_dataset.csv` ($1,891 \times 63$)
      - `integrated_master_dataset_sampled50.csv` ($976 \times 63$)
    - `analysis/`: Stage 5 statistical tables (`analysis1`..`analysis8`), audits (`study_rows_lost_and_reasons.csv`, `integration_summary_by_source.csv`), `key_insights.txt`, and `figures/` (`fig1` through `fig6`).
  - `output/supervised-ml-algo/`: Evaluation metrics (`slr_metrics.csv`, `mlr_metrics.csv`, `decision_tree_metrics.csv`, `knn_metrics.csv`, `naive_bayes_metrics.csv`, `svm_metrics.csv`, `regression_evaluation.csv`, `classification_evaluation.csv`, `feature_importances.csv`, `confusion_matrix.csv`), `supervised_key_insights.txt`, and `figures/` (6 figures).
  - `output/unsupervised-ml-algo/`: Clustering metrics (`kmeans_evaluation_k2_k6.csv`, `kmeans_profiles.csv`, `kmeans_vs_gap_typology.csv`, `mean_shift_metrics.csv`, `mean_shift_profiles.csv`), `unsupervised_key_insights.txt`, and `figures/` (5 figures).
  - `analysis/`: Stage 5 statistical tables (`analysis1`..`analysis8`), audits (`study_rows_lost_and_reasons.csv`, `integration_summary_by_source.csv`), and `key_insights.txt`.
  - `analysis/figures/`: Publication visualizations (`fig1` through `fig6`).
- `.venv/`: Dedicated Python 3.14 virtual environment with `pyspark`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, and `scikit-learn`.

## Setup and Commands
Always run commands using `dsa_final_jm\.venv\Scripts\python.exe`:

```powershell
cd "c:\Coding Projects\Python Projects\dsa_final_jm"

# Run the 5-Stage Data Cleaning & Analysis Pipeline
.\.venv\Scripts\python.exe code\data-cleaning\01_data_integration.py
.\.venv\Scripts\python.exe code\data-cleaning\02_data_cleansing.py
.\.venv\Scripts\python.exe code\data-cleaning\03_data_transformation.py
.\.venv\Scripts\python.exe code\data-cleaning\04_data_reduction.py
.\.venv\Scripts\python.exe code\data-cleaning\05_data_analysis_and_visualization.py

# Run Supervised Machine Learning (SLR, MLR, Decision Tree, KNN, Naive Bayes, SVM)
.\.venv\Scripts\python.exe code\supervised-ml-algo\main.py

# Run Unsupervised Machine Learning (K-Means, Mean-Shift)
.\.venv\Scripts\python.exe code\unsupervised-ml-algo\main.py
```

## Boundaries and Gotchas
- **Spaces in SPARK_HOME**: The workspace path contains spaces. Always ensure `pipeline_utils.py` sets `os.environ["SPARK_HOME"] = os.path.dirname(pyspark.__file__)` before initializing `SparkSession`.
- **Windows-Safe CSV Writing**: Do not call `df.write.csv()` on Windows without Hadoop binaries. Use `write_spark_df_to_csv()` from `pipeline_utils.py` which streams partitions cleanly via Python.
- **Git Boundaries**: Never commit `.venv/` directories.


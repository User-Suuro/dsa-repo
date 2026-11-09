# AGENTS.md — Python & PySpark Data Scalability Workspace

## Purpose
Monorepo of Python and PySpark data engineering, survey NLP, statistical analytics, and machine learning projects.

## Repository Map
- `dsa_final/`: Capstone 5-stage PySpark multi-source survey integration and analytics pipeline (see `dsa_final/AGENTS.md` for detailed commands and rules).
- `study_processed_data/`: 4-stage PySpark preprocessing & statistical analysis for student AI tools survey (`main.py`, `analysis.py`).
- `cleaned_data_group_act/`: PySpark multi-LLM consensus classification & survey integration (`cleaned_data.py`, `analysis.py`).
- `handling_missing_values_pyspark/`, `individual_activity_pyspark/`, `py_spark_group_act/`: Targeted PySpark missing-value imputation, feature engineering, and group preprocessing exercises.
- `dsa_assignment_02/`: Regex extraction & NLTK stopword preprocessing (`main.py`).
- `supervised_ml_algo/` & `unsupervised_ml_algo/`: Scikit-Learn classification, regression, PCA, K-Means, and Mean-Shift clustering scripts.
- `.agents/skills/`: Verified workspace skills (`pyspark-large-dataset-processing`, `multi-source-data-integration`, `survey-nlp-and-text-mining`, `ml-and-statistical-analytics`, `clean-code`, `agents-md`, `antigravity-skill-orchestrator`, `agent-memory-discipline`).

## Setup and Commands
```powershell
# Run the dsa_final 5-stage pipeline using its dedicated virtual environment
.\dsa_final\.venv\Scripts\python.exe dsa_final\code\data-cleaning\01_data_integration.py
.\dsa_final\.venv\Scripts\python.exe dsa_final\code\data-cleaning\02_data_cleansing.py
.\dsa_final\.venv\Scripts\python.exe dsa_final\code\data-cleaning\03_data_transformation.py
.\dsa_final\.venv\Scripts\python.exe dsa_final\code\data-cleaning\04_data_reduction.py
.\dsa_final\.venv\Scripts\python.exe dsa_final\code\data-cleaning\05_data_analysis_and_visualization.py
```

## Boundaries and Gotchas
- Never commit `.venv/` directories (`dsa_final/.venv` and `study_processed_data/.venv` are git-ignored).
- Always set `os.environ["SPARK_HOME"] = os.path.dirname(pyspark.__file__)` before creating a `SparkSession` on Windows because the parent path `C:\Coding Projects\Python Projects` contains spaces.


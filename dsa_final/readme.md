# AI and the Environment: An Analysis of Public Awareness and Responsible AI Usage

A simplified, reproducible Apache PySpark and Scikit-Learn analytics project analyzing our primary community survey (`dataset/main_dataset.csv`: **84 raw responses $\to$ 72 cleaned topic-valid records**, with all **12 excluded/unrelated responses** explicitly audited and marked).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
# Step 1: PySpark Data Cleaning (84 -> 72), Q7/Q8 Categorization & Descriptive Analysis (Figs 1-4)
.\.venv\Scripts\python.exe code\01_clean_and_analyze.py

# Step 2: Supervised Machine Learning with 70/30 Train-Test Split (Figs 5-6)
.\.venv\Scripts\python.exe code\02_machine_learning.py
```

## Outputs

- **Cleaned Data & Audits (`output/`)**: `cleaned_main_dataset.csv` (`N = 72`), `cleaning_audit_all84.csv`, `q7_q8_categorization_all84.csv` (marking all unrelated/invalid entries), `q7_q8_category_summary.csv`, descriptive/correlation CSVs, and supervised ML evaluation tables (`ml_summary.txt`).
- **Figures (`output/figures/`)**:
  - `fig1_data_cleaning_audit.png`
  - `fig2_likert_and_usage.png`
  - `fig3_q7_q8_categorization.png`
  - `fig4_correlation_and_gap.png`
  - `fig5_ml_regression.png`
  - `fig6_ml_classification.png`
- **Paper (`docs/`)**: `IEEE-AI-and-the-Environment-Finals-Rev5.docx` and `IEEE-AI-and-the-Environment-Finals-Rev5.pdf`.

## Requirements

- Python 3.14+ & Java JDK 17
- `pyspark==4.2.0`, `pandas==3.0.6`, `numpy==2.5.3`, `matplotlib==3.11.2`, `scikit-learn==1.9.1`, `scipy==1.18.1`

"""
main.py
=======
Coordinator for Stage 1: PySpark Data Cleaning, Qualitative Coding,
Statistical Analysis, and Figures 1-4.
"""

import os
import sys
from pathlib import Path

_CODE_DIR = str(Path(__file__).resolve().parents[1])
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from data_cleaning.cleaning import load_and_clean_with_pyspark
from data_cleaning.config import FIG_DIR, OUTPUT_DIR
from data_cleaning.qualitative_coding import build_q7_q8_audit_and_summary
from data_cleaning.spark_utils import create_spark_session
from data_cleaning.statistics import compute_statistics
from data_cleaning.visualizations import generate_all_cleaning_figures


def main() -> None:
    """Execute the full Stage 1 cleaning, qualitative coding, stats, and plotting pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 78)
    print("  STEP 1: MAIN DATASET PYSPARK CLEANING, Q7/Q8 CATEGORIZATION & ANALYSIS")
    print("=" * 78)

    spark = create_spark_session()
    try:
        pdf_all, pdf_clean = load_and_clean_with_pyspark(spark, export_csv=True)
    finally:
        spark.stop()

    print(f"Raw records loaded   : {len(pdf_all)}")
    print(f"Cleaned valid records: {len(pdf_clean)} ({len(pdf_clean)*100.0/len(pdf_all):.2f}%)")
    print(f"Excluded records     : {len(pdf_all) - len(pdf_clean)}")

    audit_df, q7q8_summary = build_q7_q8_audit_and_summary(pdf_all)
    stats_out = compute_statistics(pdf_all, pdf_clean)

    print("\n--- DESCRIPTIVE STATISTICS (N = 72) ---")
    print(stats_out["desc"][["label", "mean", "sd", "median", "mode"]].to_string(index=False))

    print("\n--- Q7 & Q8 CATEGORY SUMMARY (ALL 84 RAW, UNRELATED MARKED) ---")
    print(q7q8_summary.to_string(index=False))

    print("\n--- KEY CORRELATIONS (N = 72) ---")
    print(stats_out["corr"].to_string(index=False))

    generate_all_cleaning_figures(pdf_all, pdf_clean, audit_df, q7q8_summary, stats_out)
    print(f"\nSaved cleaned data, CSV summaries, and 4 figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

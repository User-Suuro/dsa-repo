"""
data_cleaning package
=====================
Modular Apache PySpark cleaning, qualitative coding, statistical analysis,
and figure generation for the primary community survey.
"""

from data_cleaning.main import main as run_cleaning_pipeline

__all__ = ["run_cleaning_pipeline"]

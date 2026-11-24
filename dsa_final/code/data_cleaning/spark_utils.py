"""
spark_utils.py
==============
Windows-safe Apache PySpark session factory and CSV streaming writer.
"""

import csv
import os
import sys

import pyspark
from pyspark.sql import DataFrame, SparkSession

# Configure Windows-safe PySpark paths (handles spaces in workspace path)
os.environ["SPARK_HOME"] = os.path.dirname(pyspark.__file__)
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def create_spark_session(
    app_name: str = "DSA-Final-MainDataset-Cleaning-Analysis",
) -> SparkSession:
    """Initialize a lightweight local PySpark session."""
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def write_spark_csv(df: DataFrame, path: str) -> None:
    """Write Spark DataFrame directly to a single UTF-8 CSV without winutils.exe."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = df.columns
    with open(path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for row in df.toLocalIterator():
            writer.writerow(["" if v is None else v for v in row])

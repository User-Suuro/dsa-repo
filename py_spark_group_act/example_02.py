from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import os
import sys

# Use the Python interp reter from the current virtual environment
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Create SparkSession
spark = (
    SparkSession.builder
    .appName("Test")
    .master("local[*]")
    .getOrCreate()
)

# Read CSV
df = spark.read.csv(
    "ecomerce.csv",
    header=True,
    escape='"',
    inferSchema=True
)

# Count customers per country
df.groupBy("Country") \
  .agg(count("CustomerID").alias("country_count")) \
  .show()

spark.stop()
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import os
import sys

# Use the Python interpreter from the current virtual environment
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

# Display first 5 rows
print("Preview (first 5 rows):")
df.show(5, 0)

#  Original 
print("Customer count per country:")
df.groupBy('Country').agg(count('CustomerID').alias('country_count')).show()

# 1. Display the first 10 customers only 
print("1. First 10 customers:")
(df.select('CustomerID')
   .where(col('CustomerID').isNotNull())
   .distinct()
   .orderBy('CustomerID')
   .limit(10)
   .show())

# 2. Display the total number of customers 
print("2. Total number of customers:")
df.select(count_distinct('CustomerID').alias('total_customers')).show()

# 3. First 10 customers and their total purchase 
print("3. First 10 customers with total purchase amount:")
(df.where(col('CustomerID').isNotNull())
   .withColumn('purchase_amount', col('Quantity') * col('UnitPrice'))
   .groupBy('CustomerID')
   .agg(round(sum('purchase_amount'), 2).alias('total_purchase'))
   .orderBy('CustomerID')
   .limit(10)
   .show())

spark.stop()
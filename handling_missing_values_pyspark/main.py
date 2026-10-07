from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when

# Step 1: Create SparkSession and DataFrame
spark = SparkSession.builder.appName("MissingValues").getOrCreate()

data = [
    ("Alice", 20, "Female", 25000),
    ("Bob", None, "Male", 30000),
    ("Charlie", 22, None, 28000),
    ("Diana", 21, "Female", None),
    ("Edward", None, "Male", 35000),
    ("Fiona", 24, None, 32000),
    ("George", 23, "Male", None),
    ("Hannah", None, "Female", 40000),
    ("Ian", 25, None, 38000),
    ("Julia", 22, "Female", None)
]
columns = ["Name", "Age", "Gender", "Income"]

df = spark.createDataFrame(data, columns)

# Step 2: Display the original DataFrame
print("=== Original DataFrame ===")
df.show()

# Step 3: Count missing values in each column
print("=== Missing Values per Column ===")
df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# Step 4: Handle missing values
df_clean = df.fillna({"Age": 0, "Gender": "Unknown", "Income": 0})

# Step 5: Display the cleaned DataFrame
print("=== Cleaned DataFrame ===")
df_clean.show()

# Step 6: Verify no missing values remain
print("=== Missing Values After Cleaning ===")
df_clean.select([count(when(col(c).isNull(), c)).alias(c) for c in df_clean.columns]).show()
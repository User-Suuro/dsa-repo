# Perform Data Transformation and Analysis Using Python and Apache Spark

## NOTE: ANY DATASET IS NOT INCLUDED IN THIS REPOSITORY (contains personal data). Please request the dataset from the author if you wish to run this pipeline."

## Setup

1. Create a virtual environment:

```powershell
python -m venv .venv
```

2. Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Set the PySpark Python environment:

```powershell
$env:PYSPARK_PYTHON = (Get-Command python).Source
$env:PYSPARK_DRIVER_PYTHON = (Get-Command python).Source
```

## Requirements

* Python 3.14.7
* Java JDK 17
* PySpark 4.2.0
* Pandas 3.0.6
* NumPy 2.5.3
* Matplotlib 3.11.2
* NLTK 3.10.3
* Joblib 1.6.0
* Regex 2026.9.3

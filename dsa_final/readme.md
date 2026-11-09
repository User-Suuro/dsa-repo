# AI and the Environment: An Analysis of Public Awareness

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

## Run

```powershell
python code\data-cleaning\01_data_integration.py
python code\data-cleaning\02_data_cleansing.py
python code\data-cleaning\03_data_transformation.py
python code\data-cleaning\04_data_reduction.py
python code\data-cleaning\05_data_analysis_and_visualization.py
```

## Requirements

* Python 3.14.7
* Java JDK 17
* PySpark 4.2.0
* Pandas 3.0.6
* NumPy 2.5.3
* Matplotlib 3.11.2

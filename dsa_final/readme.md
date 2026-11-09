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

### 1. Data Cleaning & Integration Pipeline

```powershell
python code\data-cleaning\01_data_integration.py
python code\data-cleaning\02_data_cleansing.py
python code\data-cleaning\03_data_transformation.py
python code\data-cleaning\04_data_reduction.py
python code\data-cleaning\05_data_analysis_and_visualization.py
```

### 2. Supervised Machine Learning (6 Algorithms)

Run all models or individual algorithms:

```powershell
# Run all 6 supervised algorithms sequentially
python code\supervised-ml-algo\main.py

# Or run individual algorithms:
python code\supervised-ml-algo\slr.py             # Simple Linear Regression
python code\supervised-ml-algo\mlr.py             # Multiple Linear Regression
python code\supervised-ml-algo\decision_tree.py   # Decision Tree Classifier
python code\supervised-ml-algo\knn.py             # K-Nearest Neighbors Classifier
python code\supervised-ml-algo\naive_bayes.py     # Gaussian Naive Bayes Classifier
python code\supervised-ml-algo\svm.py             # Support Vector Machine Classifier
```

### 3. Unsupervised Machine Learning (2 Algorithms)

Run all models or individual algorithms:

```powershell
# Run both K-Means and Mean-Shift sequentially
python code\unsupervised-ml-algo\main.py

# Or run individual algorithms:
python code\unsupervised-ml-algo\kmeans.py        # K-Means Clustering & Personas (k=4)
python code\unsupervised-ml-algo\mean_shift.py   # Mean-Shift Density Clustering
```

## Requirements

* Python 3.14.7
* Java JDK 17
* PySpark 4.2.0
* Pandas 3.0.6
* NumPy 2.5.3
* Matplotlib 3.11.2

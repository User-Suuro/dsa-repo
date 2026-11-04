# Machine Learning Algorithms: Supervised Learning Examples

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

## Run

Run any example on its own:

```powershell
python 01_decision_tree.py
python 02_svm.py
python 03_naive_bayes.py
python 04_knn.py
python 05_simple_linear_regression.py
python 06_multiple_linear_regression.py
```

Notes:

* Examples 01, 02, 04, 05 and 06 open a plot window. Close it to finish the script.
* Example 06 downloads the `mpg` dataset through seaborn the first time it runs, so an internet connection is needed.

## Files

| File | Algorithm | Dataset |
|---|---|---|
| `01_decision_tree.py` | Decision Tree | Iris |
| `02_svm.py` | Support Vector Machine | Iris |
| `03_naive_bayes.py` | Naive Bayes | Iris |
| `04_knn.py` | K-Nearest Neighbors | Iris |
| `05_simple_linear_regression.py` | Simple Linear Regression | Diabetes |
| `06_multiple_linear_regression.py` | Multiple Linear Regression | mpg |

## Requirements

* Python 3.10 or newer
* scikit-learn 1.9.1
* Matplotlib 3.11.2
* NumPy 2.5.4
* Pandas 3.0.6
* Seaborn 0.13.2
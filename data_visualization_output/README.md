# Visualization Examples

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

### 1. Bar Graphs (4 Variants)

```powershell
python example_01.py   # Bar graph
python example_02.py   # Horizontal bar graph
python example_03.py   # Grouped bar graph
python example_04.py   # Stacked bar graph
```

### 2. Pie Charts (2 Variants)

```powershell
python example_05.py   # Exploded pie chart
python example_06.py   # Percentage-labeled pie chart
```

### 3. Scatter Plot (1 Variant)

```powershell
python example_07.py   # Colored and sized scatter plot
```

### Run All Examples

Each chart opens in its own window. Close a window to move on to the next example:

```powershell
Get-ChildItem example_*.py | ForEach-Object { python $_.Name }
```

## Requirements

* Python 3.8 or newer
* Matplotlib 3.7 or newer
* NumPy 1.24 or newer
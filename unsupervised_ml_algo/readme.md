# Unsupervised Learning: Clustering Examples

Python examples taken from the "Machine Learning Algorithm (Unsupervised Learning)" slides.

## Files

| File | Algorithm | Dataset |
|------|-----------|---------|
| `kmeans_example.py` | K-Means Clustering (K=3) | Wine |
| `mean_shift_example.py` | Mean-Shift Clustering | Digits |

Both scripts use PCA to reduce the data to 2 dimensions for plotting.

## Requirements

- Python 3.8+
- numpy, pandas, matplotlib, scikit-learn (listed in `requirements.txt`)

## Setup

```bash
# (optional) create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

Or install directly:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Run

```bash
python kmeans_example.py
python mean_shift_example.py
```

Each script prints some information to the console and opens matplotlib plot windows. Close a window to continue to the next plot.

## Notes

- Both datasets (`load_wine`, `load_digits`) are bundled with scikit-learn, so no download is needed.
- `mean_shift_example.py` shows two plots: the PCA-reduced data colored by true digit labels, then the Mean-Shift clusters. In the slides' output Mean-Shift finds only one cluster, which is expected with the chosen bandwidth on this data.
- Hierarchical Clustering is covered in the slides but has no code example.
"""
knn.py
======
K-Nearest Neighbors Classifier (k=5 with StandardScaler) predicting High vs.
Low/Moderate Responsible Behavior from Q2, Q3, and Q4.
"""

import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_CODE_DIR = str(Path(__file__).resolve().parents[1])
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from machine_learning.data_loader import (
    evaluate_classification_predictions,
    get_classification_split,
    load_cleaned_dataset,
)

MODEL_NAME = "K-Nearest Neighbors (k=5)"


def train_and_evaluate(df: pd.DataFrame) -> Dict[str, Any]:
    """Train and evaluate KNN (k=5) on the stratified 70/30 split."""
    X_train, X_test, y_train, y_test = get_classification_split(df)
    clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return evaluate_classification_predictions(MODEL_NAME, len(X_train), y_test, y_pred)


def main() -> None:
    """Standalone runner for K-Nearest Neighbors Classifier."""
    df = load_cleaned_dataset()
    res = train_and_evaluate(df)
    print(f"{MODEL_NAME} (N = {len(df)}, Stratified 70/30 Split):")
    print(f"  Metrics         : {res['row']}")
    print(f"  Confusion Matrix: {res['confusion_matrix'].tolist()}")


if __name__ == "__main__":
    main()

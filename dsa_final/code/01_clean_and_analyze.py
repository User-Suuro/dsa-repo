"""
01_clean_and_analyze.py
=======================
Thin backwards-compatible runner for Stage 1 (`data_cleaning.main`).
"""

import os
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from data_cleaning.main import main

if __name__ == "__main__":
    main()

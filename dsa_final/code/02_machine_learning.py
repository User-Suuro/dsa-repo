"""
02_machine_learning.py
======================
Thin backwards-compatible runner for Stage 2 (`machine_learning.main`).
"""

import os
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from machine_learning.main import main

if __name__ == "__main__":
    main()

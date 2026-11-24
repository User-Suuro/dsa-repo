"""
qualitative_coding.py
=====================
Row-level Q7/Q8 categorization (including off-topic subtypes) and category
percentage summaries across all 84 raw and 72/71 valid survey entries.
"""

import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

_CODE_DIR = str(Path(__file__).resolve().parents[1])
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from data_cleaning.config import OUTPUT_DIR, Q7_LABELS, Q8_LABELS


def classify_off_topic_subtype(q7_text: str) -> str:
    """Assign a clear human-readable subtype to unrelated (non-environmental) Q7 answers."""
    text = q7_text.lower()
    if "privacy" in text or "store the information" in text:
        return "Unrelated: Data Privacy"
    if "hack" in text:
        return "Unrelated: Cybersecurity / Hacking"
    if "creditibility" in text or "credibility" in text:
        return "Unrelated: Information Credibility"
    if "depending" in text or "depend" in text:
        return "Unrelated: AI Over-Dependence"
    if "reality" in text or "not real" in text:
        return "Unrelated: Synthetic / Fake Reality"
    if "thinking for itself" in text or "like a human" in text:
        return "Unrelated: AI Autonomy / Sentience"
    return "Unrelated: Vague Non-Env Remark"


def _build_row_audit_df(pdf_all: pd.DataFrame) -> pd.DataFrame:
    """Build row-level Q7 and Q8 audit table across all 84 submissions."""
    rows_audit = []
    for _, row in pdf_all.iterrows():
        if row["is_valid"] == 1:
            q7_cats = [Q7_LABELS[c] for c in Q7_LABELS if row[c] == 1]
            q7_label = "; ".join(q7_cats)
            if row["q8_status"] == "Valid":
                q8_cats = [Q8_LABELS[c] for c in Q8_LABELS if row[c] == 1]
                q8_label = "; ".join(q8_cats) if q8_cats else "General Awareness"
            else:
                q8_label = row["q8_status"]
        elif row["record_status"] == "Excluded: Unrelated / Off-Topic Q7":
            q7_label = classify_off_topic_subtype(str(row["concern_text"]))
            q8_label = "Excluded (Respondent Q7 Unrelated)"
        else:
            q7_label = row["record_status"]
            q8_label = row["record_status"]

        rows_audit.append({
            "respondent_id": row["respondent_id"],
            "age_group": row["age_group"],
            "usage_segment": row["usage_segment"],
            "record_status": row["record_status"],
            "q7_categories": q7_label,
            "q8_categories": q8_label,
            "concern_text": row["concern_text"],
            "awareness_text": row["awareness_text"],
        })

    return pd.DataFrame(rows_audit)


def _build_category_summary_df(pdf_all: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Q7 and Q8 theme counts, % of valid, and % of all 84 raw responses."""
    n_raw = len(pdf_all)
    n_valid_q7 = int((pdf_all["is_valid"] == 1).sum())
    n_valid_q8 = int((pdf_all["q8_status"] == "Valid").sum())
    n_off_topic = int((pdf_all["record_status"] == "Excluded: Unrelated / Off-Topic Q7").sum())
    n_blank_dot = int(
        pdf_all["record_status"].isin(["Excluded: Blank Q7 & Q8", "Excluded: Placeholder ('.')"]).sum()
    )
    n_copied_q8 = int((pdf_all["q8_status"] == "Invalid Q8: Copied Prompt").sum())

    summary_rows = []
    for col, lbl in Q7_LABELS.items():
        cnt = int(pdf_all[col].sum())
        summary_rows.append({
            "question": "Q7",
            "group": "Valid Environmental Concern",
            "category": lbl,
            "count": cnt,
            "pct_of_valid": round(cnt * 100.0 / n_valid_q7, 1),
            "pct_of_raw84": round(cnt * 100.0 / n_raw, 1),
        })
    summary_rows.append({
        "question": "Q7", "group": "Unrelated / Excluded",
        "category": "Unrelated / Off-Topic (Non-Env)",
        "count": n_off_topic, "pct_of_valid": np.nan, "pct_of_raw84": round(n_off_topic * 100.0 / n_raw, 1),
    })
    summary_rows.append({
        "question": "Q7", "group": "Unrelated / Excluded",
        "category": "Blank / Placeholder ('.')",
        "count": n_blank_dot, "pct_of_valid": np.nan, "pct_of_raw84": round(n_blank_dot * 100.0 / n_raw, 1),
    })

    for col, lbl in Q8_LABELS.items():
        cnt = int(pdf_all[col].sum())
        summary_rows.append({
            "question": "Q8",
            "group": "Valid Suggested Solution",
            "category": lbl,
            "count": cnt,
            "pct_of_valid": round(cnt * 100.0 / n_valid_q8, 1),
            "pct_of_raw84": round(cnt * 100.0 / n_raw, 1),
        })
    summary_rows.append({
        "question": "Q8", "group": "Unrelated / Invalid",
        "category": "Excluded w/ Off-Topic Q7",
        "count": n_off_topic, "pct_of_valid": np.nan, "pct_of_raw84": round(n_off_topic * 100.0 / n_raw, 1),
    })
    summary_rows.append({
        "question": "Q8", "group": "Unrelated / Invalid",
        "category": "Blank / Placeholder ('.')",
        "count": n_blank_dot, "pct_of_valid": np.nan, "pct_of_raw84": round(n_blank_dot * 100.0 / n_raw, 1),
    })
    summary_rows.append({
        "question": "Q8", "group": "Unrelated / Invalid",
        "category": "Invalid: Copied Prompt Text",
        "count": n_copied_q8, "pct_of_valid": np.nan, "pct_of_raw84": round(n_copied_q8 * 100.0 / n_raw, 1),
    })

    return pd.DataFrame(summary_rows)


def build_q7_q8_audit_and_summary(pdf_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create and export row-level Q7/Q8 categorization (84 rows) and summary CSVs."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    audit_df = _build_row_audit_df(pdf_all)
    audit_df.to_csv(os.path.join(OUTPUT_DIR, "q7_q8_categorization_all84.csv"), index=False, encoding="utf-8-sig")

    summary_df = _build_category_summary_df(pdf_all)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "q7_q8_category_summary.csv"), index=False, encoding="utf-8-sig")
    return audit_df, summary_df

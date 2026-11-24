"""
visualizations.py
=================
Publication-grade figures (Figures 1 to 4) for data cleaning audit, Likert & usage
distributions, Q7/Q8 qualitative coding, and correlation & alignment typology.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

_CODE_DIR = str(Path(__file__).resolve().parents[1])
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from data_cleaning.config import FIG_DIR, configure_plot_style


def plot_fig1_cleaning_audit(pdf_all: pd.DataFrame, audit_df: pd.DataFrame) -> None:
    """Figure 1: Data Cleaning Audit of Main Survey Dataset (84 Raw -> 72 Cleaned, 12 Excluded)."""
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.3), gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.78})

    n_valid = int((pdf_all["is_valid"] == 1).sum())
    n_off = int((pdf_all["record_status"] == "Excluded: Unrelated / Off-Topic Q7").sum())
    n_blank = int((pdf_all["record_status"] == "Excluded: Blank Q7 & Q8").sum())
    n_dot = int((pdf_all["record_status"] == "Excluded: Placeholder ('.')").sum())

    cats_a = [
        "Cleaned & Kept\n(Valid Env. Data)",
        "Unrelated / Off-Topic\n(Non-Env. Q7)",
        "Blank Form\n(No Q7 & Q8)",
        "Placeholder Only\n('.' Entered)",
    ]
    vals_a = [n_valid, n_off, n_blank, n_dot]
    cols_a = ["#16a34a", "#dc2626", "#94a3b8", "#64748b"]

    y_a = np.arange(len(cats_a))[::-1]
    bars_a = axes[0].barh(y_a, vals_a, height=0.58, color=cols_a, edgecolor="white", linewidth=0.9)
    for bar, v in zip(bars_a, vals_a):
        pct = v * 100.0 / len(pdf_all)
        axes[0].text(v + 1.5, bar.get_y() + bar.get_height() / 2, f"{v} ({pct:.1f}%)", va="center", fontweight="bold", fontsize=8.5)
    axes[0].set_yticks(y_a)
    axes[0].set_yticklabels(cats_a, fontweight="bold", fontsize=8.5)
    axes[0].set_xlim(0, 102)
    axes[0].set_xlabel("Number of Survey Responses (Total Raw = 84)")
    axes[0].set_title("A. Raw Survey Cleaning Outcome (N = 84)", fontweight="bold", pad=10)

    excl_df = audit_df[audit_df["record_status"] != "Valid"].copy()
    sub_counts = excl_df["q7_categories"].value_counts().sort_values(ascending=True)
    y_b = np.arange(len(sub_counts))
    cols_b = ["#dc2626" if "Unrelated:" in idx else "#64748b" for idx in sub_counts.index]
    bars_b = axes[1].barh(y_b, sub_counts.values, height=0.62, color=cols_b, alpha=0.9, edgecolor="white", hatch="//")
    for bar, v in zip(bars_b, sub_counts.values):
        axes[1].text(v + 0.12, bar.get_y() + bar.get_height() / 2, f"n = {int(v)}", va="center", fontweight="bold", fontsize=8.2)
    axes[1].set_yticks(y_b)
    axes[1].set_yticklabels(sub_counts.index, fontsize=8.2, fontweight="bold")
    axes[1].set_xlim(0, max(sub_counts.values) + 1.3)
    axes[1].set_xticks(range(0, int(max(sub_counts.values)) + 2))
    axes[1].set_xlabel("Excluded Responses (Total Removed = 12)")
    axes[1].set_title("B. Exact Breakdown of 12 Excluded & Unrelated Rows", fontweight="bold", pad=10)

    fig.suptitle("Data Cleaning Audit: Main Community Survey (84 Raw -> 72 Clean Records)", fontweight="bold", fontsize=11.5, y=1.03)
    fig.savefig(os.path.join(FIG_DIR, "fig1_data_cleaning_audit.png"), bbox_inches="tight")
    plt.close(fig)


def plot_fig2_likert_and_usage(likert_df: pd.DataFrame, seg_df: pd.DataFrame) -> None:
    """Figure 2: 100% Stacked Likert Responses (Q3-Q6) and Mean Indices by Usage Segment (N = 72)."""
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), gridspec_kw={"width_ratios": [1.3, 1.05], "wspace": 0.32})

    order = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]
    colors = ["#b2182b", "#ef8a62", "#d9d9d9", "#67a9cf", "#2166ac"]
    tot = likert_df[order].sum(axis=1)
    share = likert_df[order].div(tot, axis=0) * 100.0
    labels = likert_df["item"].tolist()[::-1]
    share = share.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(labels))
    left = np.zeros(len(labels))

    for cat, colr in zip(order, colors):
        w = share[cat].values
        axes[0].barh(y_pos, w, left=left, height=0.58, color=colr, label=cat, edgecolor="white", linewidth=0.8)
        txt_col = "white" if cat in ("Strongly Disagree", "Strongly Agree") else "#1e293b"
        for i, (st, wv) in enumerate(zip(left, w)):
            if wv >= 6.0:
                axes[0].text(st + wv / 2, i, f"{wv:.0f}%", va="center", ha="center", color=txt_col, fontweight="bold", fontsize=7.8)
        left += w

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels, fontsize=8.4, fontweight="bold")
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Percentage of Cleaned Respondents (%)")
    axes[0].set_title("A. Likert Distribution: Awareness & Behavior (N = 72)", fontweight="bold", pad=10)
    axes[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(0.48, -0.18), frameon=False, fontsize=7.8)

    x = np.arange(len(seg_df))
    bw = 0.35
    b1 = axes[1].bar(x - bw / 2, seg_df["mean_awareness"], bw, label="Awareness Index (Q3-Q4)", color="#2166ac", edgecolor="white")
    b2 = axes[1].bar(x + bw / 2, seg_df["mean_behavior"], bw, label="Behavior Index (Q5-Q6)", color="#ef8a62", edgecolor="white")
    axes[1].bar_label(b1, fmt="%.2f", padding=2, fontsize=8.3, fontweight="bold")
    axes[1].bar_label(b2, fmt="%.2f", padding=2, fontsize=8.3, fontweight="bold")
    seg_x_labels = [
        f"{s.split(' (')[0]}\n({s.split(' (')[1]}\n(n={int(n)})"
        for s, n in zip(seg_df["usage_segment"], seg_df["n"])
    ]
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(seg_x_labels, fontsize=8.2, fontweight="bold")
    axes[1].set_ylim(0, 5.5)
    axes[1].set_ylabel("Mean Score (1-5 Scale)")
    axes[1].set_title("B. Mean Indices by AI Usage Segment (N = 72)", fontweight="bold", pad=10)
    axes[1].legend(frameon=False, loc="upper right", fontsize=8.0)

    fig.savefig(os.path.join(FIG_DIR, "fig2_likert_and_usage.png"), bbox_inches="tight")
    plt.close(fig)


def plot_fig3_q7_q8_categorization(summary_df: pd.DataFrame) -> None:
    """Figure 3: Complete Q7 & Q8 Qualitative Categorization (N = 84 Raw)."""
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), gridspec_kw={"wspace": 0.62})

    for ax, q, valid_col, title in [
        (axes[0], "Q7", "#0f766e", "A. Q7 Environmental Concerns (N = 84 Raw)\n[72 Valid Multi-Theme | 12 Excluded Marked]"),
        (axes[1], "Q8", "#1d4ed8", "B. Q8 Suggested Solutions (N = 84 Raw)\n[71 Valid Multi-Theme | 13 Excluded/Invalid Marked]"),
    ]:
        sub = summary_df[summary_df["question"] == q].copy()
        valid_part = sub[sub["group"].str.startswith("Valid")].sort_values("count", ascending=True)
        excl_part = sub[~sub["group"].str.startswith("Valid")].sort_values("count", ascending=True)

        n_excl = len(excl_part)
        n_val = len(valid_part)
        y_excl = np.arange(n_excl)
        y_val = np.arange(n_excl + 0.75, n_excl + 0.75 + n_val)

        bars_v = ax.barh(y_val, valid_part["count"], height=0.62, color=valid_col, alpha=0.92, edgecolor="white", label="Valid Theme (Included)")
        for bar, (_, r) in zip(bars_v, valid_part.iterrows()):
            ax.text(
                r["count"] + 0.9, bar.get_y() + bar.get_height() / 2,
                f"n={int(r['count'])} ({r['pct_of_valid']:.1f}% valid)",
                va="center", fontsize=8.0, fontweight="bold", color="#0f172a",
            )

        excl_colors = ["#dc2626" if ("Off-Topic" in c or "Copied" in c) else "#64748b" for c in excl_part["category"]]
        bars_e = ax.barh(y_excl, excl_part["count"], height=0.62, color=excl_colors, alpha=0.88, edgecolor="white", hatch="///", label="Marked Unrelated / Invalid (Excluded)")
        for bar, (_, r) in zip(bars_e, excl_part.iterrows()):
            ax.text(
                r["count"] + 0.9, bar.get_y() + bar.get_height() / 2,
                f"n={int(r['count'])} ({r['pct_of_raw84']:.1f}% raw) [EXCLUDED]",
                va="center", fontsize=7.9, fontweight="bold", color="#991b1b",
            )

        sep_y = n_excl - 0.12
        ax.axhline(sep_y, color="#dc2626", linestyle="--", linewidth=1.2, alpha=0.85)
        x_lim_max = sub["count"].max() * 1.62
        ax.text(
            x_lim_max * 0.98, sep_y + 0.10,
            "Excluded / Unrelated below line",
            ha="right", va="bottom", fontsize=7.4, style="italic", color="#991b1b",
        )

        all_y = np.concatenate([y_excl, y_val])
        all_labels = list(excl_part["category"]) + list(valid_part["category"])
        ax.set_yticks(all_y)
        ax.set_yticklabels(all_labels, fontsize=8.3, fontweight="bold")
        for idx_tick, lbl_t in enumerate(ax.get_yticklabels()):
            if idx_tick < n_excl:
                lbl_t.set_color("#991b1b")

        ax.set_xlim(0, x_lim_max)
        ax.set_xlabel("Number of Mentions / Responses")
        ax.set_title(title, fontweight="bold", fontsize=9.3, pad=10)

    legend_handles = [
        Patch(facecolor="#0f766e", edgecolor="white", label="Q7 Valid Environmental Concern (Included)"),
        Patch(facecolor="#1d4ed8", edgecolor="white", label="Q8 Valid Suggested Solution (Included)"),
        Patch(facecolor="#dc2626", edgecolor="white", hatch="///", label="Unrelated / Off-Topic / Copied Prompt (Excluded)"),
        Patch(facecolor="#64748b", edgecolor="white", hatch="///", label="Blank / Placeholder '.' (Excluded)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=True, facecolor="white", edgecolor="#cbd5e1", fontsize=8.2)
    fig.suptitle(
        "Qualitative Categorization of Open-Ended Responses (Q7 & Q8) with Unrelated/Invalid Entries Marked",
        fontweight="bold", fontsize=11.3, y=1.02,
    )
    fig.savefig(os.path.join(FIG_DIR, "fig3_q7_q8_categorization.png"), bbox_inches="tight")
    plt.close(fig)


def plot_fig4_correlation_and_gap(pdf_clean: pd.DataFrame, gap_df: pd.DataFrame) -> None:
    """Figure 4: Pearson Correlation Matrix (Q2-Q6) & Awareness-Behavior Alignment Typology (N = 72)."""
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), gridspec_kw={"wspace": 0.30})

    hm_vars = ["aware_energy_n", "aware_carbon_n", "consider_impact_n", "avoid_use_n", "usage_freq_n"]
    hm_lbls = ["Q3\nEnergy", "Q4\nCarbon", "Q5\nConsider", "Q6\nAvoid", "Q2\nUsage"]
    cm = pdf_clean[hm_vars].astype(float).corr().values

    im = axes[0].imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_xticks(range(5))
    axes[0].set_xticklabels(hm_lbls, fontsize=8.2, fontweight="bold")
    axes[0].set_yticks(range(5))
    axes[0].set_yticklabels(hm_lbls, fontsize=8.2, fontweight="bold")
    for i in range(5):
        for j in range(5):
            axes[0].text(
                j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8.2,
                color="white" if abs(cm[i, j]) > 0.5 else "black", fontweight="bold",
            )
    axes[0].set_title("A. Pearson Correlation Matrix (N = 72)", fontweight="bold", pad=10)
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    pie_colors = {
        "Aware & taking action": "#2166ac",
        "Unaware & not acting": "#94a3b8",
        "Aware but not acting": "#ef8a62",
        "Acting beyond awareness": "#16a34a",
    }
    gp = gap_df.set_index("gap_type")["count"]
    _, _, autotexts = axes[1].pie(
        gp.values,
        labels=[f"{k}\n(n={int(v)})" for k, v in gp.items()],
        autopct="%1.1f%%",
        pctdistance=0.72,
        startangle=135,
        colors=[pie_colors.get(k, "#94a3b8") for k in gp.index],
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
        textprops={"fontsize": 8.2, "fontweight": "bold"},
    )
    for w_idx, k in enumerate(gp.index):
        if k in ("Aware & taking action", "Acting beyond awareness"):
            autotexts[w_idx].set_color("white")
    axes[1].set_title("B. Awareness-Behavior Alignment (N = 72)", fontweight="bold", pad=10)

    fig.savefig(os.path.join(FIG_DIR, "fig4_correlation_and_gap.png"), bbox_inches="tight")
    plt.close(fig)


def generate_all_cleaning_figures(
    pdf_all: pd.DataFrame,
    pdf_clean: pd.DataFrame,
    audit_df: pd.DataFrame,
    q7q8_summary: pd.DataFrame,
    stats_out: Dict[str, Any],
) -> None:
    """Create all four data-cleaning and descriptive analysis figures."""
    os.makedirs(FIG_DIR, exist_ok=True)
    plot_fig1_cleaning_audit(pdf_all, audit_df)
    plot_fig2_likert_and_usage(stats_out["likert"], stats_out["seg"])
    plot_fig3_q7_q8_categorization(q7q8_summary)
    plot_fig4_correlation_and_gap(pdf_clean, stats_out["gap"])

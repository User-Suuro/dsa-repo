"""
config.py
=========
Shared filesystem paths, Matplotlib styling, survey encoding maps, and
qualitative regex rules for the data cleaning and analysis pipeline.
"""

import os
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_CLEANING_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.abspath(os.path.join(DATA_CLEANING_DIR, ".."))
PROJECT_DIR = os.path.abspath(os.path.join(CODE_DIR, ".."))
RAW_CSV = os.path.join(PROJECT_DIR, "dataset", "main_dataset.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
CLEANED_CSV = os.path.join(OUTPUT_DIR, "cleaned_main_dataset.csv")


def configure_plot_style() -> None:
    """Apply consistent publication-grade Matplotlib styling."""
    plt.rcParams.update({
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "font.family": "DejaVu Sans",
    })


FREQ_SCORE_MAP: Dict[str, float] = {
    "rarely": 1.0,
    "once a week": 2.0,
    "several times a week": 3.0,
    "once a day": 4.0,
    "several times a day": 5.0,
}

LIKERT_SCORE_MAP: Dict[str, float] = {
    "strongly disagree": 1.0,
    "disagree": 2.0,
    "neutral": 3.0,
    "agree": 4.0,
    "strongly agree": 5.0,
}

# Regex detecting whether Q7/Q8 addresses AI's environmental impact
ENV_TOPIC_REGEX = (
    r"water|drought|drouth|cool|scarcity|energy|electric|power|watt|grid|"
    r"carbon|emission|greenhouse|global warming|climate|co2|pollut|"
    r"e-waste|e waste|waste|hardware|mining|resource|material|contaminat|"
    r"noise|heat|temperature|loud|flora|fauna|wildlife|ecosystem|biodiversity|"
    r"habitat|forest|\bland\b|farm|pax silica|clark|deforest|aeta|"
    r"environment|nature|earth|planet|data cent|server|sustainab|ecological|green"
)

# 6 Core Q7 Environmental Concern Themes + General Environmental fallback
Q7_THEMES: Dict[str, str] = {
    "c_water": r"water|drought|drouth|cool|scarcity|fresh water",
    "c_energy": r"energy|electric|power|watt|grid",
    "c_carbon": r"carbon|emission|greenhouse|global warming|climate|co2|pollut|weather",
    "c_ecosystem": (
        r"flora|fauna|wildlife|ecosystem|biodiversity|habitat|forest|\bland\b|"
        r"farm|pax silica|clark|deforest|aeta|destroying nature|environmental destruction"
    ),
    "c_ewaste": (
        r"e-waste|e waste|electronic waste|waste|hardware|mining|"
        r"natural resource|resource depletion|contaminat"
    ),
    "c_noise_heat": r"noise|heat|temperature|loud",
}

Q7_LABELS: Dict[str, str] = {
    "c_water": "Water Use & Scarcity",
    "c_energy": "Energy & Electricity Use",
    "c_carbon": "Carbon & Climate Change",
    "c_ecosystem": "Ecosystem, Land & Pax Silica",
    "c_ewaste": "E-Waste & Resource Depletion",
    "c_noise_heat": "Heat & Noise Pollution",
    "c_general_env": "General Environmental Mention",
}

# 6 Core Q8 Suggested Solution Themes
Q8_THEMES: Dict[str, str] = {
    "s_education": (
        r"educat|school|teach|seminar|literacy|workshop|training|learn|inform|"
        r"curricul|universit|orientat|course|class|presentation|enlighten|understand|subject|knowledge"
    ),
    "s_campaign": (
        r"campaign|social media|advocacy|advocate|\bposts?\b|media|poster|tiktok|"
        r"infographic|video|platform|flyer|news|article|ads|page|promotional|"
        r"camping awareness|highlight|present the pros|reality of data|talking about|more aware"
    ),
    "s_responsible_use": (
        r"responsib|moderat|avoid|mindful|wisely|proper use|unnecessary|minimalism|"
        r"own mind|own work|own skills|own ideas|last resort|sparingly|not rely|"
        r"relying|depend|defend on it|creativity|non-digital|human resources|books|"
        r"libraries|discouraging|intentionally|limits|choice"
    ),
    "s_policy": (
        r"law|regulat|polic|government|gobyerno|legislat|block|limit|rules|"
        r"guideline|dict|ntc|heads of the institution|mandatory|capitalization"
    ),
    "s_transparency": r"transparen|disclos|report|accountab|label|compan|developer",
    "s_tech": (
        r"efficien|renewable|clean energy|green|sustainab|solar|cooling|"
        r"equipments placements|manage wastes"
    ),
}

Q8_LABELS: Dict[str, str] = {
    "s_education": "Education & Literacy Programs",
    "s_campaign": "Awareness Campaigns & Media",
    "s_responsible_use": "Responsible & Mindful AI Use",
    "s_policy": "Policy, Governance & Limits",
    "s_transparency": "Corporate Transparency & Labels",
    "s_tech": "Green & Sustainable AI Tech",
}

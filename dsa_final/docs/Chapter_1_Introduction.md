# CHAPTER I: INTRODUCTION

**Study Title:** *AI and the Environment: An Analysis of Public Awareness and Responsible AI Usage*  
**Course:** IT 119 – Data Scalability and Analytics (BSIT 3B)  
**Researchers:** Renz Dominic R. Azul, Gabriel R. de la Cruz, Godwin Reigh M. Galvez  

---

## 1.1 Background of the Study

Students and workers now open AI chatbots the same way they open a calculator. Because a response appears on a laptop or phone screen almost immediately, almost nobody pictures the server racks running behind the scenes. Those data centers house thousands of processors that consume heavy electricity, evaporate clean freshwater for cooling, and generate hardware waste. According to the International Energy Agency (IEA, 2025), global data centers drew roughly 415 terawatt-hours (TWh) of electricity in 2024, nearly 1.5% of global electricity demand, and AI workloads could push that demand to more than double by 2030.

Public understanding has not kept pace with daily adoption. Cho et al. (2025) surveyed residents in Japan and South Korea and found a wide gap: only 21.9% of Japanese respondents and 50.1% of South Korean respondents knew about AI's electricity, water, and carbon costs. Jelonek and Pawełoszek (2026) surveyed 224 respondents using a composite Environmental Awareness Index and reported that more than two-thirds backed greener AI tools once informed. In the Philippines, Dungo et al. (2025) found that while college students recognize AI's electricity demand, most still miss data-center cooling water and rare-earth mineral extraction.

That blind spot matters locally. In April 2026, the Philippines joined *Pax Silica*, a US-led supply-chain partnership for semiconductors and artificial intelligence that outlines a 4,000-acre industrial hub in the Luzon Economic Corridor (Rocamora, 2026; Villanueva, 2026). Projects at that scale bring investment and jobs, yet they also raise immediate local questions about power grids, municipal water supply, and land conversion.

This study focuses strictly on our own primary community survey from Camarines Norte, Philippines (`84` raw responses cleaned down to `72` topic-valid records after auditing and marking `12` excluded entries). We kept the study anchored on our own dataset instead of merging public surveys from abroad so our cleaning audit in Apache PySpark, our open-ended review of `Q7` and `Q8`, and our small-sample machine learning stay concrete and easy to verify.

---

## 1.2 Statement of the Problem

Everyday AI users rarely see the electricity, cooling water, or carbon attached to a prompt. Five questions guide this study:

1. Who answered the survey (`Q1`), and how often do they use AI tools (`Q2`)?
2. How aware are respondents of AI's electricity consumption (`Q3`), data-center carbon impacts (`Q4`), and overall `Awareness_Index`?
3. How responsibly do they use AI in terms of weighing environmental impact (`Q5`), skipping unnecessary prompts (`Q6`), and overall `Behavior_Index`?
4. How closely does awareness predict responsible behavior, and which supervised machine learning models hold up best on `72` records?
5. What worries (`Q7`), practical solutions (`Q8`), and unrelated or invalid entries appear across all `84` raw open-ended responses?

---

## 1.3 Objectives of the Study

Our general objective is to measure community awareness of AI's environmental impact, test how strongly awareness links to responsible AI habits, categorize every open-ended reply (including unrelated ones), and identify which machine learning models fit a small dataset (`N = 72`).

Specifically, the study aims to:
1. Profile respondents' age brackets and AI usage habits (`Light`, `Moderate`, and `Heavy` users).
2. Measure environmental awareness (`Awareness_Index`) and responsible usage (`Behavior_Index`), then evaluate their correlation and alignment gap.
3. Categorize all `84` raw open-ended answers in `Q7` and `Q8`, marking off-topic and invalid entries transparently.
4. Benchmark low-variance machine learning models on `N = 72` using Leave-One-Out Cross-Validation (LOOCV) and Repeated Stratified 5-Fold Cross-Validation.

---

## 1.4 Conceptual Framework

We follow a simple Input–Process–Output (IPO) workflow built strictly around `main_dataset.csv`:

```mermaid
flowchart LR
    subgraph INPUT["INPUT (Primary Survey Only)"]
        I1["Camarines Norte Community Survey\nmain_dataset.csv (84 Raw Responses)"]
        I2["Q1 Age & Q2 AI Usage\nQ3 Energy & Q4 Carbon Awareness\nQ5 Consider Impact & Q6 Avoid Unnecessary Use\nQ7 Concerns & Q8 Solutions (Open Text)"]
    end

    subgraph PROCESS["PROCESS (2 Python Scripts)"]
        P1["Script 1: PySpark Cleaning & Analysis\n• Remove PII & fix '24-34' -> '25-34'\n• Mark & exclude 12 unrelated/blank rows (N = 72)\n• Compute scores, correlations & Q7/Q8 categories"]
        P2["Script 2: Small-Sample ML (N = 72)\n• LOOCV & 10x5-Fold Stratified CV\n• Simple Linear Regression (Scores)\n• Gaussian NB & L2 Logistic Regression (Groups)"]
        P1 --> P2
    end

    subgraph OUTPUT["OUTPUT"]
        O1["72 Clean Records + All-84 Q7/Q8 Audit\n(Unrelated & Invalid Marked)"]
        O2["Awareness-Behavior Link (r = 0.7643)\n& Small-N ML Benchmarks"]
        O3["Local Green-AI Education &\nMindful-Use Recommendations"]
    end

    INPUT --> PROCESS --> OUTPUT
```

---

## 1.5 Significance of the Study

- **Students and Everyday AI Users**: Makes the cooling-water and electricity footprint of casual prompts concrete, pointing toward practical habits like skipping AI on tasks you can do yourself.
- **Schools and Teachers**: Directly supports curricular updates, since *Education & Literacy Programs* (`57.7%`) topped all respondent suggestions in `Q8`.
- **Local Leaders**: Captures grounded Filipino worries around municipal water shortages (`61.1%`) and land impacts tied to *Pax Silica* (`11.1%`).
- **Analytics Students**: Shows how to run clean PySpark auditing and honest cross-validated machine learning on a modest `72`-record survey without padding the sample with mismatched external files.

---

## 1.6 Scope and Delimitations

This work covers only our Google Forms community survey (`dataset/main_dataset.csv`: `84` raw responses, `72` clean records) collected in Daet and nearby towns of Camarines Norte during the Second Semester of AY 2025–2026. No external or public datasets appear in the methodology or models.

Three real boundaries apply. First, `86.1%` of our `72` clean respondents are aged 18–24, so these findings reflect a local youth and student community rather than the whole country. Second, `12` raw rows were excluded from quantitative modeling and documented openly (`7` off-topic Q7 answers about privacy, hacking, misinformation, dependence, fake reality, or sentient AI; `4` blank forms; and `1` dot-only entry), while `1` verbatim copied Q8 prompt was excluded from `Q8` counts (`n = 71` valid Q8 replies). Third, Likert ratings measure self-reported awareness and habits rather than metered device wattage.

---

## 1.7 Definition of Terms

1. **AI Usage Frequency (`Q2` / `usage_freq_n`)**: How often a person uses AI tools, scored `1` (*Rarely*) to `5` (*Several times a day*), grouped into *Light* (`1–2`), *Moderate* (`3`), and *Heavy* (`4–5`).
2. **Awareness Index**: Average of `Q3` (electricity awareness) and `Q4` (carbon/data-center awareness) on a `1.0`–`5.0` scale.
3. **Behavior Index**: Average of `Q5` (weighing environmental impact during use) and `Q6` (skipping unnecessary AI use) on a `1.0`–`5.0` scale.
4. **Awareness–Behavior Gap**: `Awareness_Index − Behavior_Index`, sorting respondents into *Aware & taking action* (`69.4%`), *Aware but not acting* (`12.5%`), *Acting beyond awareness* (`9.7%`), and *Unaware & not acting* (`8.3%`).
5. **Unrelated / Off-Topic Response**: Any `Q7` reply discussing non-environmental AI risks like hacking, privacy, misinformation, or autonomous robots (`n = 7` of `84` raw forms).
6. **Leave-One-Out Cross-Validation (LOOCV)**: Testing a model `72` separate times, each time training on `71` respondents and predicting the single left-out respondent so small-sample accuracy stays reliable.

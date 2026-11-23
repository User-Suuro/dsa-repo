# CHAPTER 3: METHODOLOGY

**Study Title:** *AI and the Environment: An Analysis of Public Awareness and Responsible AI Usage*  
**Course:** IT 119 – Data Scalability and Analytics (BSIT 3B)  
**Researchers:** Renz Dominic R. Azul, Gabriel R. de la Cruz, Godwin Reigh M. Galvez  

---

## 3.1 Study Setup

We kept our pipeline lean: one local survey file (`dataset/main_dataset.csv`) and two short Python scripts. No external or public datasets enter the workflow. Script 1 cleans the raw survey, audits every open-ended reply in Apache PySpark, and builds descriptive charts; Script 2 runs small-sample cross-validated machine learning in scikit-learn.

---

## 3.2 Survey Collection and Instrument

Over four weeks during the Second Semester of AY 2025–2026, we shared an eight-question Google Form across class group chats, student forums, and local social media pages in Daet and nearby towns of Camarines Norte using convenience and purposive sampling. Participation was voluntary and required clicking an informed consent box; all **84 respondents** agreed.

Q1 captured age bracket and Q2 captured AI usage frequency from `1` (*Rarely*) to `5` (*Several times a day*). Q3 and Q4 used 5-point Likert scales (`1 = Strongly Disagree` to `5 = Strongly Agree`) to measure awareness of AI electricity use and data-center carbon emissions, whereas Q5 and Q6 asked whether respondents weigh environmental impact while using AI and whether they avoid AI when a task can be done without it. Across the `72` clean responses, Q3 through Q6 reached strong internal reliability (**Cronbach's $\alpha = 0.91$**). Q7 and Q8 were free-text boxes asking what environmental impact worried respondents most and what actions would raise public awareness and encourage responsible use.

---

## 3.3 Two-Script Pipeline

### Script 1: Cleaning, Auditing & Q7/Q8 Categorization (`code/01_clean_and_analyze.py`)
We loaded `main_dataset.csv` (`84` raw rows) into Apache PySpark 4.2.0 with multi-line quote support because several respondents wrote multi-paragraph replies. First, we dropped timestamps and email/username columns so every submission stays completely anonymous. Second, we normalized spaces and line breaks and corrected a typo we caught in our own Google Form options, where `12` respondents had selected `'24-34'` instead of `'25-34'`.

Next, we read every `Q7` and `Q8` reply and checked it against a PySpark environmental keyword filter. Twelve raw rows (`14.3%`) failed validation and were excluded and marked openly: `7` off-topic replies talking about hacking, personal data privacy, misinformation, over-dependence, synthetic reality, or sentient AI instead of the environment; `4` blank forms; and `1` period-only (`'.'`) entry. Removing those `12` rows left **`72` clean records (`85.7%`)**. Inside those `72` records, one person gave a valid `Q7` answer but accidentally pasted our `Q8` question back into the `Q8` box; we kept them for `Q1`–`Q7` (`N = 72`) and flagged their `Q8` text as invalid (`n = 71` valid `Q8` replies).

For every clean respondent, we computed `Awareness_Index = (Q3 + Q4) / 2`, `Behavior_Index = (Q5 + Q6) / 2`, and `Gap = Awareness_Index − Behavior_Index`, grouped `Q2` into *Light* (`1–2`), *Moderate* (`3`), and *Heavy* (`4–5`) users, and tagged all `84` raw `Q7` and `Q8` answers into plain topic categories while keeping unrelated and invalid entries separated below a visual threshold.

### Script 2: Small-Sample Machine Learning (`code/02_machine_learning.py`)
A standard 75/25 holdout split fails on `72` rows because holding out 25% leaves just `18` test respondents, with only `1` or `2` cases from the minority `Low` behavior group (`n = 7` total). Instead of trusting a single lucky seed, we evaluated every model using **Leave-One-Out Cross-Validation (LOOCV)** (`72` runs training on `71` rows and testing on `1`) and **Repeated Stratified 5-Fold Cross-Validation** (`5 folds × 10 repeats = 50 folds`).

For continuous `Behavior_Index` scores, we benchmarked **Simple Linear Regression** (`Awareness_Index` only) against Ridge Regression (`alpha = 10`) and Multiple Linear Regression (`Q2, Q3, Q4`). For classification, we used `[Q2, Q3, Q4]` to predict a balanced binary split (**High Responsible Behavior** `>= 4.0`, `n = 47`, versus **Low/Moderate** `< 4.0`, `n = 25`) alongside a 3-class check (`Low = 7`, `Medium = 18`, `High = 47`). Focusing on low-variance models suited to small tables, we compared **Gaussian Naive Bayes** and **L2 Logistic Regression** against Linear SVM, a shallow Decision Tree (`max_depth = 2`), KNN (`k = 7`), and a shallow Random Forest (`max_depth = 3`).

---

## 3.4 Tools
Python 3.14.7, Apache PySpark 4.2.0, Pandas, NumPy, SciPy, Scikit-Learn 1.9.1, and Matplotlib.

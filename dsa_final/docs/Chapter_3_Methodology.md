# CHAPTER 3: METHODOLOGY

**Study Title:** *AI and the Environment: An Analysis of Public Awareness and Responsible AI Usage*  
**Course:** IT 119 - Data Scalability and Analytics (BSIT 3B)  
**Researchers:** Renz Dominic R. Azul, Gabriel R. de la Cruz, Godwin Reigh M. Galvez  

---

## 3.1 Analytical Workflow

This study uses a four-stage analytics workflow covering community survey collection (`dataset/main_dataset.csv`), data cleaning and qualitative coding in Apache PySpark, statistical correlation analysis, and supervised machine learning in scikit-learn on a 70/30 train-test split.

---

## 3.2 Survey Collection and Instrument

Over four weeks during the Second Semester of Academic Year 2025-2026, we shared an eight-question Google Form across class group chats, student forums, and local social media pages in Daet and nearby towns of Camarines Norte using convenience and purposive sampling. Participation was voluntary and required clicking an informed consent box, and all 84 respondents agreed.

Q1 captured age bracket and Q2 captured AI usage frequency from `1` (*Rarely*) to `5` (*Several times a day*). Q3 and Q4 used 5-point Likert scales (`1 = Strongly Disagree` to `5 = Strongly Agree`) to measure awareness of AI electricity use and data-center carbon emissions, whereas Q5 and Q6 asked whether respondents weigh environmental impact while using AI and whether they avoid AI when a task can be done without it. Across the `72` clean responses, Q3 through Q6 reached strong internal reliability (Cronbach's $\alpha = 0.91$). Q7 and Q8 were free-text boxes asking what environmental impact worried respondents most and what actions would raise public awareness and encourage responsible use.

---

## 3.3 Data Processing and Modeling Workflow

### Stage 1: Cleaning, Auditing & Q7/Q8 Categorization (`code/01_clean_and_analyze.py`)
We loaded `main_dataset.csv` (`84` raw rows) into Apache PySpark 4.2.0 with multi-line quote support because several respondents wrote multi-paragraph replies. First, we dropped timestamps and email/username columns so every submission stays completely anonymous. Second, we normalized spaces and line breaks and corrected a typo we caught in our own Google Form options, where `12` respondents had selected `'24-34'` instead of `'25-34'`.

Next, we analyzed every `Q7` and `Q8` reply and checked it against a PySpark environmental keyword filter. We excluded and marked twelve raw rows (`14.3%`) that failed validation, consisting of `7` off-topic replies talking about hacking, personal data privacy, misinformation, over-dependence, synthetic reality, or sentient AI instead of the environment, `4` blank forms, and `1` period-only (`'.'`) entry. Removing those `12` rows left `72` clean records (`85.7%`). Inside those `72` records, one person gave a valid `Q7` answer but accidentally pasted our `Q8` question back into the `Q8` box. We kept them for `Q1`-`Q7` (`N = 72`) and flagged their `Q8` text as invalid (`n = 71` valid `Q8` replies).

For every clean respondent, we computed `Awareness_Index = (Q3 + Q4) / 2`, `Behavior_Index = (Q5 + Q6) / 2`, and `Gap = Awareness_Index - Behavior_Index`, grouped `Q2` into *Light* (`1-2`), *Moderate* (`3`), and *Heavy* (`4-5`) users, and tagged all `84` raw `Q7` and `Q8` answers into plain topic categories while keeping unrelated and invalid entries separated below a visual threshold.

### Stage 2: Supervised Machine Learning (`code/02_machine_learning.py`)
We evaluated six standard supervised learning models in scikit-learn using a straightforward 70/30 train-test split (`50` training records and `22` held-out testing records, `random_state = 42`).

For continuous `Behavior_Index` prediction, we trained Simple Linear Regression (`Awareness_Index` only) and Multiple Linear Regression (`Q2, Q3, Q4`), measuring test Coefficient of Determination (`R²`), Root Mean Square Error (`RMSE`), and Mean Absolute Error (`MAE`). For classification, we used `[Q2, Q3, Q4]` on a stratified 70/30 split (`14` High and `8` Low/Moderate test respondents) to predict High Responsible Behavior (`>= 4.0`) versus Low/Moderate (`< 4.0`). We compared four classic supervised classifiers: Decision Tree (`maximum depth = 3`), Support Vector Machine (`Radial Basis Function kernel` with feature scaling), Gaussian Naive Bayes, and K-Nearest Neighbors (`5 nearest neighbors, k = 5`, with feature scaling), reporting test accuracy, weighted precision, recall, F1 score, and confusion matrices.

---

## 3.4 Tools
Python 3.14.7, Apache PySpark 4.2.0, Pandas, NumPy, SciPy, Scikit-Learn 1.9.1, and Matplotlib.

# CHAPTER 3: METHODOLOGY

**Study Title:** *AI and the Environment: An Analysis of Public Awareness and Responsible AI Usage*  
**Course:** IT 119 – Data Scalability and Analytics (BSIT 3B)  
**Researchers:** Renz Dominic R. Azul, Gabriel R. de la Cruz, Godwin Reigh M. Galvez  

---

## 3.1 Study Setup

This study uses a simple survey design focused strictly on our primary dataset (`dataset/main_dataset.csv`). We did not merge any external or public datasets into the analysis. Everything runs through two short Python scripts: one for cleaning and analyzing the survey in Apache PySpark, and one for small-sample machine learning in scikit-learn.

---

## 3.2 Survey Collection and Respondents

We built an eight-question Google Form (`Q1` to `Q8`) and shared it for four weeks during the Second Semester of AY 2025–2026 in Daet and nearby towns in Camarines Norte, Philippines, using convenience and purposive sampling through class group chats, student forums, and social media. Participation was voluntary, and all **84 respondents** agreed to the informed consent box.

The form had four short parts:
1. **Profile & Usage (`Q1`–`Q2`)**: Age bracket (`Q1`) and how often the respondent uses AI tools (`Q2`, scored `1 = Rarely` to `5 = Several times a day`).
2. **Environmental Awareness (`Q3`–`Q4`)**: Two 5-point Likert statements (`1 = Strongly Disagree` to `5 = Strongly Agree`) asking if they know AI uses a lot of electricity (`Q3`) and if AI data centers add to carbon emissions and other environmental impacts (`Q4`).
3. **Responsible AI Use (`Q5`–`Q6`)**: Two 5-point Likert statements on considering environmental impact when using AI (`Q5`) and avoiding unnecessary AI use when they can do the task without it (`Q6`). Internal reliability across `Q3`–`Q6` was strong (**Cronbach's $\alpha = 0.91$** on the `72` clean responses).
4. **Open-Ended Questions (`Q7`–`Q8`)**: Free-text boxes asking what environmental impact of AI worries them most (`Q7`) and what should be done to raise awareness and encourage responsible AI use (`Q8`).

---

## 3.3 Simple Two-Script Workflow

### Step 1: Data Cleaning & Q7/Q8 Grouping (`code/01_clean_and_analyze.py`)
We loaded `main_dataset.csv` (`84` raw rows) in Apache PySpark 4.2.0 and applied four simple steps:
1. **Protect Privacy**: Dropped timestamps and emails/usernames so all respondents stay completely anonymous.
2. **Fix Text & Labels**: Trimmed extra spaces, cleaned line breaks, and corrected the age form typo `'24-34'` to `'25-34'`.
3. **Mark & Remove Bad or Unrelated Rows (`84` $\to$ `72`)**: We checked every `Q7` and `Q8` answer by hand and with a PySpark keyword check. Twelve rows (`14.3%`) were removed and clearly marked:
   - **`7` Unrelated / Off-Topic `Q7` rows (`8.3%`)**: Talked about hacking, data privacy, fake news/credibility, over-dependence, synthetic reality, AI thinking like a human, or a vague system remark instead of the environment.
   - **`4` Blank forms (`4.8%`)** and **`1` dot-only `"."` entry (`1.2%`)**.
   - Inside the remaining **`72` clean records (`85.7%`)**, one person answered `Q7` properly but accidentally pasted the `Q8` question into the `Q8` box; we kept them for `Q1`–`Q7` (`N = 72`) and marked their `Q8` text as invalid (`n = 71` valid `Q8` answers).
4. **Simple Scores & Text Themes**:
   - $\text{Awareness\_Index} = (\text{Q3} + \text{Q4}) / 2$
   - $\text{Behavior\_Index} = (\text{Q5} + \text{Q6}) / 2$
   - $\text{Gap} = \text{Awareness\_Index} - \text{Behavior\_Index}$
   - Grouped `Q2` into *Light* (`1–2`), *Moderate* (`3`), and *Heavy* (`4–5`) users, and grouped all `84` raw `Q7` and `Q8` answers into plain themes while keeping unrelated and invalid entries visibly separated.

### Step 2: Small-Sample Machine Learning (`code/02_machine_learning.py`)
Because `72` records is small, a regular 75/25 split only leaves `18` test rows (and only `1`–`2` people in the small `Low` tier of `7`), which makes single-split accuracy noisy. Instead, we kept ML simple and small-data safe:
1. **Cross-Validation**: Tested every model with **Leave-One-Out Cross-Validation (LOOCV)** (testing every person one by one) and **5-Fold Cross-Validation repeated 10 times** (`50` folds total).
2. **Continuous Score Prediction**: Tested **Simple Linear Regression** (`Behavior_Index` from `Awareness_Index`) against Ridge Regression and Multiple Linear Regression (`Q2, Q3, Q4`).
3. **Behavior Group Prediction**: Used `[Q2, Q3, Q4]` to predict a balanced two-group target—**High Responsible Behavior** (`>= 4.0`, `n = 47`) vs. **Low/Moderate** (`< 4.0`, `n = 25`)—plus a 3-group check (`Low = 7`, `Medium = 18`, `High = 47`). We focused on simple, low-variance models—**Gaussian Naive Bayes** and **L2 Logistic Regression**—and compared them against Linear SVM, a shallow Decision Tree (`max_depth = 2`), KNN (`k = 7`), and a shallow Random Forest (`max_depth = 3`).

---

## 3.4 Tools Used
- Python 3.14.7, Apache PySpark 4.2.0, Pandas, NumPy, SciPy, Scikit-Learn 1.9.1, and Matplotlib.

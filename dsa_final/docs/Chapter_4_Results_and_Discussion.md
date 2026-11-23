# CHAPTER 4: RESULTS AND DISCUSSION

**Study Title:** *AI and the Environment: An Analysis of Public Awareness and Responsible AI Usage*  
**Course:** IT 119 – Data Scalability and Analytics (BSIT 3B)  
**Researchers:** Renz Dominic R. Azul, Gabriel R. de la Cruz, Godwin Reigh M. Galvez  

---

## 4.1 Cleaning Outcome (`84` Raw $\to$ `72` Clean)

Out of `84` raw survey replies (`100%` consented), **`72` (`85.7%`)** passed our environmental-topic audit and **`12` (`14.3%`)** were excluded and marked (`output/figures/fig1_data_cleaning_audit.png`). Seven respondents (`8.3%`) wrote non-environmental concerns in `Q7` (credibility of shared info, personal data privacy, hacking, full dependence on AI, AI feeling unreal, AI thinking like a human, and one vague line about "affecting the system"); four left both open-ended boxes blank (`4.8%`), and one entered only a period `"."` (`1.2%`). Inside the `72` clean records, one respondent accidentally pasted the `Q8` prompt back into the `Q8` box, leaving `71` valid `Q8` answers.

---

## 4.2 Who Answered and How They Scored (`N = 72`)

Young adults make up almost the entire sample: `86.1%` (`n = 62`) are aged 18–24, `12.5%` (`n = 9`) are 25–34, and just `1` person (`1.4%`) is 45–54. Slightly over half use AI daily or several times a day (`51.4%` Heavy users, `n = 37`), `34.7%` (`n = 25`) use it weekly (`Moderate`), and `13.9%` (`n = 10`) use it rarely or once a week (`Light`).

Both composite indices land in the High tier (`Cronbach's alpha = 0.91`): `Awareness_Index` averages **`3.98`** ($SD = 1.14$) and `Behavior_Index` averages **`3.87`** ($SD = 0.98$), giving a tiny `+0.11` average gap ($SD = 0.74$). Item-level agreement reveals a real gap in daily habits (`fig2_likert_and_usage.png`): `84.7%` agreed they avoid unnecessary AI use (`Q6`, Mean `4.11`) and `81.9%` knew data centers add emissions (`Q4`, Mean `4.04`), yet only `62.5%` agreed they think about environmental impact while actively using AI (`Q5`, Mean `3.62`), where `25.0%` (`n = 18`) picked Neutral.

Usage frequency also pulls scores downward. Light users scored highest on awareness (`4.35`) and behavior (`4.45`), Moderate weekly users dropped to `4.18` and `3.88`, and Heavy daily users came in lowest at `3.74` and `3.70`, echoing the same habit drag Rai et al. (2026) saw among university students.

---

## 4.3 Awareness–Behavior Link (`fig4_correlation_and_gap.png`)

Awareness and responsible behavior track closely together (**Pearson $r = 0.7643$, $p < 0.001$**; Spearman $\rho = 0.6356$). Usage frequency leans negative against behavior ($r = -0.1979$, $p = 0.096$), though not enough to clear significance on `72` rows. In the four-part alignment chart, **`69.4%` (`n = 50`)** are *Aware & taking action*, **`12.5%` (`n = 9`)** are *Aware but not acting*, **`9.7%` (`n = 7`)** are *Acting beyond awareness*, and **`8.3%` (`n = 6`)** are *Unaware & not acting*.

---

## 4.4 Categorizing All 84 `Q7` & `Q8` Replies (Unrelated Marked)

See `output/figures/fig3_q7_q8_categorization.png` and `output/q7_q8_categorization_all84.csv`.

| Question & Category | Status | $n$ | % Valid | % Raw (`84`) | What Respondents Wrote |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Q7: Water Use & Scarcity** | Valid | **44** | **61.1%** | 52.4% | Cooling water, municipal droughts, local clean water shortages |
| **Q7: Energy & Electricity Use** | Valid | **30** | **41.7%** | 35.7% | Server power draw and grid strain |
| **Q7: Carbon & Climate Change** | Valid | **22** | **30.6%** | 26.2% | Emissions, global warming, unusual weather patterns |
| **Q7: Ecosystem, Land & Pax Silica** | Valid | **8** | **11.1%** | 9.5% | Deforestation, 4,000-acre Pax Silica hub, farmers & Aeta land |
| **Q7: E-Waste & Resource Depletion** | Valid | **8** | **11.1%** | 9.5% | Electronic waste leaking into soil/water, rapid hardware upgrades |
| **Q7: Heat & Noise Pollution** | Valid | **5** | **6.9%** | 6.0% | Loud data centers and heat near residential areas and wildlife |
| **Q7: General Environmental Mention** | Valid | **2** | **2.8%** | 2.4% | Environmental accounting; broad environmental comment |
| **Q7: Unrelated / Off-Topic (Non-Env)** | **EXCLUDED** | **7** | — | **8.3%** | Privacy, hacking, credibility, over-dependence, fake reality, AGI |
| **Q7: Blank (`4`) / Dot `"."` (`1`)** | **EXCLUDED** | **5** | — | **6.0%** | Empty text fields or `"."` |
| **Q8: Education & Literacy Programs** | Valid | **41** | **57.7%** | 48.8% | School lessons, college courses, seminars for youth & older adults |
| **Q8: Awareness Campaigns & Media** | Valid | **39** | **54.9%** | 46.4% | Social media, TikTok campaigns, infographics, news, town meetings |
| **Q8: Responsible & Mindful AI Use** | Valid | **35** | **49.3%** | 41.7% | Digital minimalism, protecting books/libraries, own creativity |
| **Q8: Policy, Governance & Limits** | Valid | **10** | **14.1%** | 11.9% | DICT/NTC programs, institutional rules, per-device limits |
| **Q8: Corporate Transparency & Labels** | Valid | **9** | **12.7%** | 10.7% | Energy/water usage labels and company disclosure |
| **Q8: Green & Sustainable AI Tech** | Valid | **4** | **5.6%** | 4.8% | Non-freshwater cooling and greener AI engineering |
| **Q8: Excluded w/ Off-Topic `Q7`** | **EXCLUDED** | **7** | — | **8.3%** | Attached to off-topic `Q7` replies |
| **Q8: Blank (`4`) / Dot `"."` (`1`)** | **EXCLUDED** | **5** | — | **6.0%** | Empty text fields or `"."` |
| **Q8: Copied Question Text** | **INVALID** | **1** | — | **1.2%** | Pasted the `Q8` prompt verbatim |

Water scarcity beat electricity by nearly 20 percentage points, grounded in real local worries about Philippine drought and drinking water rather than abstract global statistics. Respondents in `Q8` clearly favored education, social media outreach, and personal restraint ("digital minimalism," using books and libraries) over government bans.

---

## 4.5 Small-Sample Machine Learning Results (`N = 72`)

### 1. Predicting Exact Behavior Scores (`fig5_ml_regression_loocv.png`)
Simple Linear Regression ($\text{Behavior} = 0.6539 \times \text{Awareness} + 1.2661$) beat both Ridge Regression (`LOOCV R² = 0.5339`, `RMSE = 0.6610`) and Multiple Linear Regression (`LOOCV R² = 0.5247`, `RMSE = 0.6675`), delivering **LOOCV $R^2 = 0.5483$**, **RMSE = `0.6507`**, and **MAE = `0.5035`** (in-sample $R^2 = 0.5842$). Because `Q3` and `Q4` overlap tightly ($r = 0.88$), splitting them apart only added small-sample noise.

### 2. Predicting Behavior Groups (`fig6_ml_small_sample_classification.png`)
| Classifier (`Q2, Q3, Q4`) | Binary 10×5 CV | Binary LOOCV | Binary AUC | 3-Class 10×5 CV |
| :--- | :---: | :---: | :---: | :---: |
| **Gaussian Naive Bayes (Recommended)** | 77.7% (±9.3) | 77.8% (F1 0.761) | **0.7719 (#1)** | **73.8% (#1)** |
| **Logistic Regression L2 (Recommended)** | 78.0% (±8.7) | 76.4% (F1 0.743) | 0.6860 | 72.6% |
| **Random Forest (`max_depth=3`)** | **79.8% (±8.5)** | **79.2% (F1 0.773)** | 0.6953 | 72.5% |
| **Linear SVM (`C=1.0`)** | 77.7% (±8.3) | 77.8% (F1 0.749) | 0.7523 | 71.3% |
| **KNN (`k=7`)** | 77.5% (±8.4) | 77.8% (F1 0.749) | 0.5915 | 73.0% |
| **Decision Tree (`max_depth=2`)** | 76.6% (±8.1) | 76.4% (F1 0.748) | 0.5834 | 71.9% |

**Gaussian Naive Bayes** posted the top probability ranking (`ROC-AUC = 0.7719`), caught `44` of `47` High-behavior users in LOOCV, and finished #1 on 3-class CV (`73.8%`). **L2 Logistic Regression** hit `79.1%` in 5-fold CV while giving plain odds ratios (`Q3` energy awareness `OR = 2.06`, `Q4` carbon awareness `OR = 2.07`, `Q2` usage `OR = 0.93`). Even though shallow **Random Forest (`max_depth=3`)** nudged raw binary LOOCV accuracy up by `1.4%` (`79.2%`), its weaker AUC (`0.6953`) and 3-class dip (`70.8%` LOOCV) leave Gaussian Naive Bayes, L2 Logistic Regression, and Simple Linear Regression as the cleanest, steadiest models for `72` rows.

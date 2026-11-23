# CHAPTER 4: RESULTS AND DISCUSSION

**Study Title:** *AI and the Environment: An Analysis of Public Awareness and Responsible AI Usage*  
**Course:** IT 119 – Data Scalability and Analytics (BSIT 3B)  
**Researchers:** Renz Dominic R. Azul, Gabriel R. de la Cruz, Godwin Reigh M. Galvez  

---

## 4.1 Cleaning Audit (`84` Raw $\to$ `72` Clean Records)

Out of `84` raw survey replies (`100%` consented), **`72` (`85.7%`)** were clean and on-topic, while **`12` (`14.3%`)** were removed and marked (`output/figures/fig1_data_cleaning_audit.png`):
- **`7` Unrelated / Off-Topic `Q7` (`8.3%`)**: Credibility, data privacy, hacking, over-dependence, fake reality, AI thinking like a human, and a vague system remark.
- **`4` Blank forms (`4.8%`)** and **`1` dot-only `"."` entry (`1.2%`)**.
- Within the `72` clean rows, `1` reply copied the `Q8` question verbatim and was marked invalid for `Q8` coding (`n = 71` valid `Q8` replies).

---

## 4.2 Respondents, Usage, Awareness, and Behavior (`N = 72`)

- **Age**: `86.1%` (`n = 62`) are 18–24, `12.5%` (`n = 9`) are 25–34, and `1.4%` (`n = 1`) is 45–54.
- **Usage (`Q2`)**: `51.4%` (`n = 37`) use AI daily or several times a day (*Heavy*), `34.7%` (`n = 25`) use it several times a week (*Moderate*), and `13.9%` (`n = 10`) use it rarely or once a week (*Light*).
- **Overall Scores (`Cronbach's alpha = 0.91`)**:
  - `Awareness_Index`: **Mean = `3.98`** ($SD = 1.14$, High) — `Q3` Energy `3.92` (`75.0%` agree); `Q4` Carbon `4.04` (`81.9%` agree).
  - `Behavior_Index`: **Mean = `3.87`** ($SD = 0.98$, High) — `Q5` Consider Impact `3.62` (`62.5%` agree, **`25.0%` neutral**); `Q6` Avoid Unnecessary Use `4.11` (`84.7%` agree).
  - `Gap`: **`+0.11`** ($SD = 0.74$, closely balanced).
- **Usage Trend (`fig2_likert_and_usage.png`)**: Light users scored highest on awareness (`4.35`) and behavior (`4.45`), Moderate users scored `4.18` and `3.88`, and Heavy daily users scored lowest (`3.74` and `3.70`), matching Rai et al. (2026).

---

## 4.3 Awareness–Behavior Link (`fig4_correlation_and_gap.png`)

- Awareness and behavior are strongly linked (**Pearson $r = 0.7643$, $p < 0.001$**; Spearman $\rho = 0.6356$).
- Usage frequency showed only a weak negative trend with behavior ($r = -0.1979$, $p = 0.096$, not significant at `N = 72`).
- **Alignment Groups**:
  - *Aware & taking action*: **50 (`69.4%`)**
  - *Aware but not acting*: **9 (`12.5%`)**
  - *Acting beyond awareness*: **7 (`9.7%`)**
  - *Unaware & not acting*: **6 (`8.3%`)**

---

## 4.4 Open-Ended `Q7` & `Q8` Categories (All 84 Raw Rows; Unrelated Marked)

See `output/figures/fig3_q7_q8_categorization.png` and `output/q7_q8_categorization_all84.csv`.

| Question & Category | Status | $n$ | % Valid | % Raw (`84`) | Key Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Q7: Water Use & Scarcity** | Valid | **44** | **61.1%** | 52.4% | Cooling water, municipal drought, drinking water shortages |
| **Q7: Energy & Electricity Use** | Valid | **30** | **41.7%** | 35.7% | Server power draw and electricity strain |
| **Q7: Carbon & Climate Change** | Valid | **22** | **30.6%** | 26.2% | Emissions, global warming, weather shifts |
| **Q7: Ecosystem, Land & Pax Silica** | Valid | **8** | **11.1%** | 9.5% | Deforestation, Pax Silica 4,000-acre hub, farmers & Aeta land |
| **Q7: E-Waste & Resource Depletion** | Valid | **8** | **11.1%** | 9.5% | Electronic waste and hardware turnover |
| **Q7: Heat & Noise Pollution** | Valid | **5** | **6.9%** | 6.0% | Loud data centers and heat near homes/animals |
| **Q7: General Environmental Mention** | Valid | **2** | **2.8%** | 2.4% | Environmental accounting; broad environmental note |
| **Q7: Unrelated / Off-Topic (Non-Env)** | **EXCLUDED** | **7** | — | **8.3%** | Privacy, hacking, credibility, dependence, fake reality, AGI |
| **Q7: Blank (`4`) / Dot `"."` (`1`)** | **EXCLUDED** | **5** | — | **6.0%** | Empty answers or `"."` |
| **Q8: Education & Literacy Programs** | Valid | **41** | **57.7%** | 48.8% | School classes, courses, seminars for youth & older adults |
| **Q8: Awareness Campaigns & Media** | Valid | **39** | **54.9%** | 46.4% | Social media, TikTok videos, infographics, town meetings |
| **Q8: Responsible & Mindful AI Use** | Valid | **35** | **49.3%** | 41.7% | Digital minimalism, using books/libraries, own creativity |
| **Q8: Policy, Governance & Limits** | Valid | **10** | **14.1%** | 11.9% | DICT/NTC programs, institutional rules, device limits |
| **Q8: Corporate Transparency & Labels** | Valid | **9** | **12.7%** | 10.7% | Energy/water labels and company reporting |
| **Q8: Green & Sustainable AI Tech** | Valid | **4** | **5.6%** | 4.8% | Non-freshwater cooling and greener AI design |
| **Q8: Excluded w/ Off-Topic `Q7`** | **EXCLUDED** | **7** | — | **8.3%** | Tied to non-environmental `Q7` replies |
| **Q8: Blank (`4`) / Dot `"."` (`1`)** | **EXCLUDED** | **5** | — | **6.0%** | Empty answers or `"."` |
| **Q8: Copied Question Text** | **INVALID** | **1** | — | **1.2%** | Pasted the `Q8` prompt verbatim |

---

## 4.5 Small-Sample Machine Learning Results (`N = 72`)

### 1. Predicting Exact Behavior Scores (`fig5_ml_regression_loocv.png`)
- **Simple Linear Regression (Best)**: $\text{Behavior} = 0.6539 \times \text{Awareness} + 1.2661$, **LOOCV $R^2 = 0.5483$**, **RMSE = `0.6507`**, **MAE = `0.5035`** (full $R^2 = 0.5842$).
- **Ridge Regression (`a = 10`)**: LOOCV $R^2 = 0.5339$, RMSE = `0.6610`.
- **Multiple Linear Regression**: LOOCV $R^2 = 0.5247$, RMSE = `0.6675`.
- Keeping a single line from `Awareness_Index` predicted unseen rows better than splitting `Q2`, `Q3`, and `Q4` apart.

### 2. Predicting Behavior Groups (`fig6_ml_small_sample_classification.png`)
| Classifier (`Q2, Q3, Q4`) | Binary 10×5 CV | Binary LOOCV | Binary AUC | 3-Class 10×5 CV |
| :--- | :---: | :---: | :---: | :---: |
| **Gaussian Naive Bayes (Recommended)** | 77.7% (±9.3) | 77.8% (F1 0.761) | **0.7719 (#1)** | **73.8% (#1)** |
| **Logistic Regression L2 (Recommended)** | 78.0% (±8.7) | 76.4% (F1 0.743) | 0.6860 | 72.6% |
| **Random Forest (`max_depth=3`)** | **79.8% (±8.5)** | **79.2% (F1 0.773)** | 0.6953 | 72.5% |
| **Linear SVM (`C=1.0`)** | 77.7% (±8.3) | 77.8% (F1 0.749) | 0.7523 | 71.3% |
| **KNN (`k=7`)** | 77.5% (±8.4) | 77.8% (F1 0.749) | 0.5915 | 73.0% |
| **Decision Tree (`max_depth=2`)** | 76.6% (±8.1) | 76.4% (F1 0.748) | 0.5834 | 71.9% |

- **Why these models fit `N = 72` best**: **Gaussian Naive Bayes** achieved the strongest overall separation (`ROC-AUC = 0.7719`), correctly caught `44` of `47` High-behavior users in LOOCV, and led 3-class cross-validation (`73.8%`). **L2 Logistic Regression** hit `79.1%` in 5-fold CV and gives easy-to-read odds ratios (`Q3` energy awareness `OR = 2.06`, `Q4` carbon awareness `OR = 2.07`, `Q2` usage `OR = 0.93`). Together with **Simple Linear Regression**, they stay simple, transparent, and steady on small data.

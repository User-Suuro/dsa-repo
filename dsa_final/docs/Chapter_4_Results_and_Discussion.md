# CHAPTER 4: RESULTS AND DISCUSSION

**Study Title:** *AI and the Environment: An Analysis of Public Awareness and Responsible AI Usage*  
**Course:** IT 119 - Data Scalability and Analytics (BSIT 3B)  
**Researchers:** Renz Dominic R. Azul, Gabriel R. de la Cruz, Godwin Reigh M. Galvez  

---

## 4.1 Cleaning Outcome (`84` Raw $\to$ `72` Clean)

Out of `84` raw survey replies (`100%` consented), `72` (`85.7%`) passed our environmental-topic audit and we excluded and marked `12` (`14.3%`) (`output/figures/fig1_data_cleaning_audit.png`). Seven respondents (`8.3%`) wrote non-environmental concerns in `Q7` (credibility of shared info, personal data privacy, hacking, full dependence on AI, AI feeling unreal, AI thinking like a human, and one vague line about "affecting the system"). Four left both open-ended boxes blank (`4.8%`), and one entered only a period `"."` (`1.2%`). Inside the `72` clean records, one respondent accidentally pasted the `Q8` prompt back into the `Q8` box, leaving `71` valid `Q8` answers.

Because removing the `7` off-topic respondents (who scored lower at Awareness Mean `2.29` and Behavior Mean `2.93`, while the `5` blank or period entries averaged `4.20` on both) could introduce upward selection bias, we ran a sensitivity check across all `84` raw responses. On all `84` rows, mean Awareness (`3.85`, $SD = 1.25$) and mean Behavior (`3.81`, $SD = 1.05$) remained in the High band with nearly identical awareness-behavior correlation ($r = 0.7766$, $p < 0.001$), while the negative usage-behavior correlation became slightly stronger ($r = -0.2380$, $p = 0.029$).

---

## 4.2 Who Answered and How They Scored (`N = 72`)

Young adults make up almost the entire sample: `86.1%` (`n = 62`) are aged 18-24, `12.5%` (`n = 9`) are 25-34, and just `1` person (`1.4%`) is 45-54. Slightly over half use AI daily or several times a day (`51.4%` Heavy users, `n = 37`), `34.7%` (`n = 25`) use it weekly (`Moderate`), and `13.9%` (`n = 10`) use it rarely or once a week (`Light`).

Both composite indices land in the High tier (`Cronbach's alpha = 0.91`): `Awareness_Index` averages `3.98` ($SD = 1.14$) and `Behavior_Index` averages `3.87` ($SD = 0.98$), giving a tiny `+0.11` average gap ($SD = 0.74$). Item-level agreement shows a gap in daily habits (`fig2_likert_and_usage.png`): `84.7%` agreed they avoid unnecessary AI use (`Q6`, Mean `4.11`) and `81.9%` knew data centers add emissions (`Q4`, Mean `4.04`), yet only `62.5%` agreed they think about environmental impact while actively using AI (`Q5`, Mean `3.62`), where `25.0%` (`n = 18`) picked Neutral.

Group averages also showed an exploratory downward trend as usage rose. Light users (`n = 10`) averaged `4.35` on awareness and `4.45` on behavior, Moderate weekly users averaged `4.18` and `3.88`, and Heavy daily users averaged `3.74` and `3.70`, echoing the pattern Rai et al. (2026) observed among university students. However, these group differences were not statistically significant on `N = 72` (one-way ANOVA `p = 0.098` for behavior and `p = 0.182` for awareness) and should be interpreted cautiously given the small Light-user group (`n = 10`).

---

## 4.3 Awareness-Behavior Link (`fig4_correlation_and_gap.png`)

Awareness and responsible behavior track closely together (Pearson $r = 0.7643$, $p < 0.001$, Spearman $\rho = 0.6356$). Usage frequency leans negative against behavior ($r = -0.1979$, $p = 0.096$), though not enough to clear significance on `72` rows. In the four-part alignment chart, `69.4%` (`n = 50`) are *Aware & taking action*, `12.5%` (`n = 9`) are *Aware but not acting*, `9.7%` (`n = 7`) are *Acting beyond awareness*, and `8.3%` (`n = 6`) are *Unaware & not acting*.

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
| **Q7: General Environmental Mention** | Valid | **2** | **2.8%** | 2.4% | Environmental accounting and broad environmental comment |
| **Q7: Unrelated / Off-Topic (Non-Env)** | **EXCLUDED** | **7** | - | **8.3%** | Privacy, hacking, credibility, over-dependence, fake reality, AGI |
| **Q7: Blank (`4`) / Dot `"."` (`1`)** | **EXCLUDED** | **5** | - | **6.0%** | Empty text fields or `"."` |
| **Q8: Education & Literacy Programs** | Valid | **41** | **57.7%** | 48.8% | School lessons, college courses, seminars for youth & older adults |
| **Q8: Awareness Campaigns & Media** | Valid | **39** | **54.9%** | 46.4% | Social media, TikTok campaigns, infographics, news, town meetings |
| **Q8: Responsible & Mindful AI Use** | Valid | **35** | **49.3%** | 41.7% | Digital minimalism, protecting books/libraries, own creativity |
| **Q8: Policy, Governance & Limits** | Valid | **10** | **14.1%** | 11.9% | DICT/NTC programs, institutional rules, per-device limits |
| **Q8: Corporate Transparency & Labels** | Valid | **9** | **12.7%** | 10.7% | Energy/water usage labels and company disclosure |
| **Q8: Green & Sustainable AI Tech** | Valid | **4** | **5.6%** | 4.8% | Non-freshwater cooling and greener AI engineering |
| **Q8: Excluded w/ Off-Topic `Q7`** | **EXCLUDED** | **7** | - | **8.3%** | Attached to off-topic `Q7` replies |
| **Q8: Blank (`4`) / Dot `"."` (`1`)** | **EXCLUDED** | **5** | - | **6.0%** | Empty text fields or `"."` |
| **Q8: Copied Question Text** | **INVALID** | **1** | - | **1.2%** | Pasted the `Q8` prompt back into the box |

Water scarcity beat electricity by nearly 20 percentage points, grounded in local worries about Philippine drought and drinking water rather than abstract global statistics. Respondents in `Q8` clearly favored education, social media outreach, and personal restraint ("digital minimalism," using books and libraries) over government bans.

---

## 4.5 Supervised Machine Learning Results (`N = 72`, 70/30 Split)

### 1. Predicting Exact Behavior Scores (`fig5_ml_regression.png`)
On the `22` held-out test records (`50` training / `22` testing), Simple Linear Regression ($\text{Behavior} = 0.4824 \times \text{Awareness} + 2.0327$) reached Test Coefficient of Determination ($R^2$) = `0.6391`, Root Mean Square Error (`RMSE = 0.7577`), and Mean Absolute Error (`MAE = 0.5710`). Multiple Linear Regression (`Q2, Q3, Q4`) scored almost identically at Test $R^2 = 0.6409$, Root Mean Square Error (`RMSE = 0.7558`), and Mean Absolute Error (`MAE = 0.5728`), with positive weights for `Q3` energy awareness (`+0.2371`) and `Q4` carbon awareness (`+0.2341`) and a slight negative weight for `Q2` usage frequency (`-0.0485`). Both models explain roughly `64%` of the variance in unseen behavior scores with just over half a Likert point of average error, capturing predictive association on cross-sectional self-report data rather than proven causation.

### 2. Predicting Behavior Groups (`fig6_ml_classification.png`)
| Classifier (`Q2, Q3, Q4`) | Test Accuracy (`n = 22`) | Weighted Precision | Weighted Recall | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree (`maximum depth = 3`)** | **77.3%** | **0.778** | **0.773** | **0.758** |
| **Support Vector Machine (SVM)** | 72.7% | 0.732 | 0.727 | 0.699 |
| **Gaussian Naive Bayes** | 72.7% | 0.732 | 0.727 | 0.699 |
| **K-Nearest Neighbors (`5 nearest neighbors, k = 5`)** | 72.7% | 0.732 | 0.727 | 0.699 |

Across the `22` stratified test respondents (`14` High, `8` Low/Moderate), all four classifiers performed comparably: Decision Tree (`maximum depth = 3`) achieved `77.3%` accuracy (`Weighted F1 = 0.758`), classifying `17` of `22` test cases correctly (`13` of `14` High and `4` of `8` Low/Moderate), while Support Vector Machine (SVM), Gaussian Naive Bayes, and K-Nearest Neighbors (`5 nearest neighbors, k = 5`) each achieved `72.7%` accuracy (`Weighted F1 = 0.699`), catching `16` of `22` test cases (`13` of `14` High and `3` of `8` Low/Moderate). Because that `4.6` percentage-point difference reflects just one test respondent, the small `n = 22` test set does not prove a statistically meaningful accuracy win for Decision Tree, whose main practical strength here is transparent rule interpretability.

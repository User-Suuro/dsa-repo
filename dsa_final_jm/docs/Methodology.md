# III. METHODOLOGY

**Study Title:** *Assessing College Students' Awareness and Preparedness Toward Flood Risks: A Big Data Scalability and Machine Learning Approach*  
**Course & Project:** IT 119 — Data Scalability and Analytics  
**Researchers:** John Mark I. Pardo, Byron Boral, Aerold Steven Reyes (Group 11 – Skywalker)  
**Academic Affiliation:** College of Information and Communications Technology, Camarines Norte, Philippines  

---

## A. Research Design

This study adopts a **quantitative, descriptive, correlational, and predictive machine learning research design**, complemented by **exploratory natural language processing (NLP) text mining** of qualitative student narratives. The architecture is engineered to evaluate empirical relationships between cognitive hazard awareness and tangible disaster preparedness behaviors among tertiary-level students.

The research design is operationalized across four complementary dimensions:
1. **Descriptive Dimension**: Quantitatively profiles respondents' sociodemographic attributes, residential living arrangements (e.g., off-campus boarding houses, dormitories, familial residences), geographical flood exposure frequencies, levels of flood risk awareness across six sub-domains (`aw_*`), and levels of personal emergency preparedness across six action areas (`prep_*`).
2. **Correlational and Diagnostic Dimension**: Formulates bivariate and multivariate associations between cognitive knowledge and proactive behavior, evaluating the empirical magnitude of the **Awareness–Preparedness Gap** ($\Delta = \text{Awareness} - \text{Preparedness}$) and categorizing respondents into diagnostic behavioral segments (*Aligned*, *Aware but Unprepared*, *Prepared beyond Awareness*).
3. **Supervised Machine Learning Dimension**: Employs supervised predictive algorithms—Simple Linear Regression (SLR), Multiple Linear Regression (MLR), Decision Tree Classifier (CART), $K$-Nearest Neighbors (KNN), Gaussian Naive Bayes (GNB), and Support Vector Machine (SVM)—to model continuous preparedness scores and classify students into discrete readiness tiers (*Low*, *Moderate*, *High*).
4. **Exploratory Clustering and NLP Text-Mining Dimension**: Leverages unsupervised clustering ($K$-Means with Elbow/Silhouette optimization and non-parametric Mean-Shift mode-seeking) to uncover latent behavioral personas, paired with PySpark regular-expression (`rlike`) keyword theme extraction to categorize open-ended narratives detailing personal logistical bottlenecks (`c_*`) and institutional support needs (`s_*`).

---

## B. Respondents and Sampling

### 1. Primary Collegiate Target Population
The primary empirical core of this investigation comprises **$N = 128$ undergraduate college students** enrolled across academic degree programs—principally the Bachelor of Science in Information Technology (BSIT) and Bachelor of Science in Information Systems (BSIS)—within higher education institutions in **Camarines Norte, Bicol Region, Philippines**. 

Camarines Norte serves as a critical geographic focal point due to its direct exposure to the Pacific typhoon belt, northeast monsoonal precipitation (*Habagat*), shear-line torrents, and low-lying coastal/riverine flood vulnerabilities along the Daet-Talisay river basins.

### 2. Multi-Source Big Data Integration
While localized field surveys offer acute contextual accuracy, reliance on a localized sample of $128$ records poses constraints regarding statistical variance and generalizability across diverse disaster typologies. In compliance with the enterprise competencies of **IT 119 (Data Scalability and Analytics)**, the primary collegiate dataset was systematically merged with **three topically-aligned, peer-reviewed open-access disaster research datasets**:
- **Public Study 2: Segovia & Tungol (2026)** (*Centro Escolar University*, Mendeley Data DOI: `10.17632/462nvdfvjj.1`): An empirical survey investigating anticipatory mobility, psychological distress, and flood exposure among Philippine young adults ($N_{\text{raw}} = 541$). Following rigorous attention-filtering and psychometric validation, **$370$ clean topic-valid records** were retained.
- **Public Study 3: NOAA / IPPRA FL24 (2024)** (*University of Oklahoma*, Harvard Dataverse DOI: `10.7910/DVN/I9751R`): A standardized benchmark evaluating flood perceptions, warning dissemination, and emergency kit readiness ($N = 1,200$).
- **Public Study 4: Ravago et al. (2020)** (*Data in Brief*, Mendeley Data DOI: `10.17632/9y2yhw45zt.1`): A municipal-level disaster risk reduction and management (DRRM) dataset examining early-warning protocols across Philippine local government units ($N = 193$).

This integration produced an initial corpus of **$2,062$ raw observations**, which was cleansed and filtered into a final unified master dataset of **$N = 1,891$ clean, topic-valid records**.

---

## C. Sampling Method

A **non-probability purposive and convenience sampling strategy** was employed for primary field data gathering:
- **Purposive Sampling**: Ensured that all respondents satisfied mandatory eligibility criteria:
  1. Currently enrolled tertiary-level college student.
  2. Residing within Camarines Norte during periods of seasonal storm exposure and monsoonal flooding.
  3. Enrolled in degree programs subject to suspension and logistical disruption during hydrometeorological emergencies.
- **Convenience Sampling**: Enabled rapid and accessible survey dissemination across collegiate student bodies, departmental academic chat channels, and campus peer networks.

Demographic distributions were cleansed and categorized in Stage 2 into canonical structures:
- **Academic Programs**: Standardized into `BS Information Technology`, `BS Information Systems`, and allied disciplines.
- **Municipal Residence**: Normalized to discrete local government jurisdictions in Camarines Norte (Daet, Talisay, Mercedes, Labo, Basud, Vinzons, Jose Panganiban, Paracale, San Vicente, San Lorenzo Ruiz).
- **Living Arrangements**: Partitioned into `Living with family/parents`, `Boarding house/dormitory/rented room`, and `Living with relatives`.
- **Chronological Age**: Grouped into standardized analytical cohorts: *18–20*, *21–24*, *25–34*, *35–49*, and *50+*.

---

## D. Data Gathering and Ethics

### 1. Data Collection Procedure
Primary data gathering was administered through a structured, mobile-responsive **Google Forms** digital survey. The instrument was deployed through a dual-channel strategy:
1. **Digital Dissemination**: Circulated across verified university course group chats, departmental social media groups, and official institutional announcements.
2. **In-Person Intercepts (QR Scan)**: Field deployment using printed quick-response (QR) codes displayed on collegiate bulletin boards and scanned directly by students via mobile devices during campus assembly sessions.

Continuous response logging and timestamp verification prevented duplicate submissions. At the close of the survey window, raw CSV records were archived into the localized project staging environment (`dsa_final_jm/dataset/`).

### 2. Ethical Safeguards and Data Privacy
The research strictly adhered to the principles of the **Philippine Data Privacy Act of 2012 (Republic Act No. 10173)** and institutional academic ethics:
- **Voluntary Informed Consent**: An explicit disclosure agreement was integrated into the opening page of the survey. Participants were briefed on the study’s educational scope, guaranteed total anonymity, and granted unconditioned rights to terminate participation at any point.
- **PII Scrubbing and De-Identification**: All personally identifiable information (PII)—including respondent full names, email addresses, student identification numbers, and contact numbers—was permanently stripped during data cleansing. Each observation was assigned a cryptographically unique alphanumeric tracking key (`respondent_id`, e.g., `MAIN_000001` or `SEGOVIA_0001`).
- **Academic Non-Commercial Ring-Fencing**: All datasets, features, and model artifacts were sequestered exclusively for IT 119 academic course evaluation and will not be commercialized or distributed to third-party entities.

---

## E. Data Processing Pipeline and Measures

### 1. Enterprise 5-Stage Apache PySpark Pipeline
To handle large-scale multi-source data ingestion, cleansing, transformation, and reduction, an automated five-stage distributed **Apache PySpark (v4.3.0)** pipeline was engineered:

```
[Raw Data Sources: Main Study (128) + Segovia (541) + NOAA (1,200) + Ravago (193) = 2,062 Raw Rows]
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Ingestion & Canonicalization (01_data_integration.py)                                 │
│ • Maps diverse column naming into 26-column unified schema                                      │
│ • Enforces explicit StructType metadata and logs initial attrition logs                        │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Data Cleansing & Quality Screening (02_data_cleansing.py)                             │
│ • Filters audit flag is_valid_record == 1; screens out 171 invalid records from Segovia (2026) │
│ • Normalizes string whitespaces and clamps Likert scales strictly to [1.0, 5.0]                │
│ • Imputes missing items via PySpark Imputer(strategy="median") -> N = 1,891 clean records      │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Feature Engineering & Qualitative Theme Coding (03_data_transformation.py)            │
│ • Computes composite construct scores (awareness_score, preparedness_score, gap)                │
│ • Calculates sub-dimension indices (warning, local hazard, health, supplies, planning)          │
│ • Derives behavioral classifications (gap_typology, risk_tier)                                  │
│ • Codes open text via PySpark rlike regex: 5 challenge themes (c_*) & 5 support themes (s_*)   │
│ • Standardizes features via PySpark ML StandardScaler (*_z) -> 44 columns                      │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 4: Topic-Aligned Data Reduction (04_data_reduction.py)                                   │
│ • Performs zero-variance feature verification                                                  │
│ • Applies PySpark PCA(k=3) on 12 z-scores (capturing 64.50% cumulative variance)                │
│ • Executes 50% stratified sampling (N = 976) across preparedness levels                         │
│ • Outputs single master analytical dataset: integrated_master_dataset.csv (1,891 x 63 cols)    │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 5: Statistical Analytics & Visualizations (05_data_analysis_and_visualization.py)        │
│ • Computes 8 empirical statistical tables (descriptives, Likert, correlations, gap types)      │
│ • Generates 6 publication-grade figures (fig1 through fig6) and key_insights.txt               │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
┌──────────────────────────────────────┐     ┌───────────────────────────────────────────────┐
│ Supervised Machine Learning Suite    │     │ Unsupervised Machine Learning Suite           │
│ • SLR & MLR Continuous Regressions   │     │ • K-Means Clustering (k=4 Behavioral Personas)│
│ • Decision Tree, KNN, NB, SVM        │     │ • Mean-Shift Density Mode Discovery           │
└──────────────────────────────────────┘     └───────────────────────────────────────────────┘
```

- **Stage 1 (Integration)**: Maps varied external schemas to canonical variables, exporting `stage1_integrated_dataset.csv` ($2,062 \times 26$) and loss audits.
- **Stage 2 (Cleansing)**: Enforces quality screening, filtering out $171$ inattentive entries from Segovia (2026), clamping Likert bounds to $[1.0, 5.0]$, and applying median imputation to produce `stage2_cleansed_dataset.csv` ($1,891 \times 26$).
- **Stage 3 (Transformation)**: Derives composite metrics, sub-indices, behavioral segmentations, qualitative regex indicators, and unit-variance $z$-scores, exporting `stage3_transformed_dataset.csv` ($1,891 \times 44$).
- **Stage 4 (Reduction)**: Validates variance, computes 3-component PCA ($64.50\%$ cumulative variance), generates a $50\%$ stratified sample ($n = 976$), and serializes the master tabular output: `integrated_master_dataset.csv` ($1,891 \times 63$).
- **Stage 5 (Analytics)**: Calculates 8 empirical CSV tables, exports `key_insights.txt`, and generates publication figures.

---

### 2. Variables and Measures

**TABLE I: SUMMARY OF RESEARCH VARIABLES, OPERATIONAL DEFINITIONS, AND MEASURES**

| Variable Category | Primary Variables / Field Names | Data Type | Operational Definition and Measurement Scale |
| :--- | :--- | :--- | :--- |
| **Identifiers** | `respondent_id`, `source_study`, `study_role` | Identifier / Categorical | Programmatic key (`MAIN_000001`, `SEGOVIA_0001`) and source tag (Main Study PH, Public S2, Public S3, Public S4). |
| **Demographics & Profile** | `age`, `age_group`, `sex`, `year_level`, `course_discipline` | Continuous / Categorical | Chronological age, age brackets (*18–20*, *21–24*, *25–34*, *35–49*, *50+*), biological sex, academic level, and collegiate course. |
| **Residential Exposure** | `living_arrangement`, `municipality_residence`, `flood_history`, `flood_history_n` | Categorical / Ordinal / Continuous | Living arrangement (*Family*, *Boarding/Dorm*, *Relatives*), residence municipality, and historical flood frequency ($0.0 = \text{Never}$ to $3.0 = \text{Frequently}$). |
| **Flood Risk Awareness (6 Items)** | `aw_warning_signals`, `aw_monitor_updates`, `aw_residence_risk`, `aw_emergency_contacts`, `aw_evac_routes`, `aw_health_risks` | Ordinal | Measured on a 5-point Likert scale ($1.0 = \text{Strongly Disagree}$ to $5.0 = \text{Strongly Agree}$) covering warning alerts, weather bulletins, neighborhood hazards, MDRRMO hotlines, evacuation routes, and waterborne disease risks. |
| **Personal Preparedness (6 Items)** | `prep_go_bag`, `prep_food_water`, `prep_secure_docs`, `prep_evac_plan`, `prep_follow_protocols`, `prep_breaker_cutoff` | Ordinal | Measured on a 5-point Likert scale ($1.0 = \text{Strongly Disagree}$ to $5.0 = \text{Strongly Agree}$) evaluating 72-hour Go-Bags, 3-day food/water rations, document waterproofing, family evacuation plans, municipal protocol compliance, and electrical breaker shutoff. |
| **Composite Scores & Indices** | `awareness_score`, `preparedness_score`, `awareness_preparedness_gap` | Continuous | Unweighted mean construct scores across 6 awareness items and 6 preparedness items $[1.0, 5.0]$; and their arithmetic gap: $\text{Awareness} - \text{Preparedness}$. |
| **Sub-Dimension Indices** | `aw_warning_index`, `aw_local_hazard_index`, `aw_safety_health_index`, `prep_supplies_index`, `prep_planning_index` | Continuous | Domain indices calculated as 2-item and 3-item means $[1.0, 5.0]$ isolating warnings, local hazards, health, emergency supplies, and planning protocols. |
| **Behavioral Classifications** | `awareness_level`, `preparedness_level`, `gap_typology`, `risk_tier` | Categorical | Threshold levels (*Low* $< 2.50$, *Moderate* $2.50–3.75$, *High* $> 3.75$); Gap Typology (*Aligned* $|\Delta| \le 0.50$, *Aware but Unprepared* $\Delta > 0.50$, *Prepared beyond Awareness* $\Delta < -0.50$); and Risk Tiers (*High*, *Moderate*, *Low*). |
| **Qualitative Themes** | `c_*` (5 Challenge Themes), `s_*` (5 Support Themes) | Binary | Binary indicators ($1 = \text{Present}, 0 = \text{Absent}$) extracted via PySpark `rlike`: Challenges (supplies, financial cost, evacuation access, power/comms, housing) and Support (Go-Bags, SMS alerts, shelter, seminars, financial aid). |
| **Dimensionality Reduction** | `pca_1`, `pca_2`, `pca_3` | Continuous | Top 3 principal component orthogonal projections calculated on 12 standardized Likert $z$-scores, accounting for $64.50\%$ cumulative variance. |

---

## F. Data Analysis and Modeling

Data processed through the distributed PySpark pipeline were evaluated using descriptive and inferential statistical treatments alongside an enterprise machine learning modeling framework.

### 1. Statistical Treatment

#### a. Descriptive Statistics and Verbal Interpretation
Continuous indices (`awareness_score`, `preparedness_score`, `awareness_preparedness_gap`) were evaluated using the mean ($M$), standard deviation ($SD$), median, minimum, and maximum. Item-level Likert responses were assessed via frequency and percentage agreement distributions. Mean scores were interpreted using a standard 5-interval Likert continuum with equal widths of $0.80$:

**TABLE II: LIKERT CONTINUUM SCALE AND VERBAL INTERPRETATION GUIDE**

| Mean Range | Likert Anchor | Verbal Interpretation: Awareness | Verbal Interpretation: Preparedness |
| :---: | :---: | :---: | :---: |
| **1.00 – 1.80** | Strongly Disagree | Very Low Awareness | Very Low Preparedness |
| **1.81 – 2.60** | Disagree | Low Awareness | Low Preparedness |
| **2.61 – 3.40** | Neutral / Undecided | Moderate Awareness | Moderate Preparedness |
| **3.41 – 4.20** | Agree | High Awareness | High Preparedness |
| **4.21 – 5.00** | Strongly Agree | Very High Awareness | Very High Preparedness |

#### b. Pearson Product-Moment Correlation Analysis
Bivariate linear associations between awareness items and preparedness behaviors were measured using Pearson’s $r$:

$$r = \frac{\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{n} (X_i - \bar{X})^2 \sum_{i=1}^{n} (Y_i - \bar{Y})^2}}$$

Coefficients were benchmarked using established standards: $\pm 0.00–0.19$ (very weak), $\pm 0.20–0.39$ (weak), $\pm 0.40–0.59$ (moderate), $\pm 0.60–0.79$ (strong), and $\pm 0.80–1.00$ (very strong).

---

### 2. Machine Learning Modeling Framework

The modeling architecture incorporates both **supervised predictive algorithms** and **unsupervised clustering algorithms** trained on the unified master dataset ($N = 1,891$). Supervised algorithms were evaluated using a stratified **$75/25$ train-test split** ($1,418$ training instances / $473$ testing instances, `random_state=42`).

```
                    ┌────────────────────────────────────────────────────────┐
                    │     Cleaned Master Analytical Dataset (N = 1,891)      │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     ▼                                                     ▼
┌─────────────────────────────────────────┐   ┌───────────────────────────────────────────┐
│        SUPERVISED PREDICTIVE ML         │   │         UNSUPERVISED CLUSTERING ML        │
│                                         │   │                                           │
│ Continuous Prediction (Regression):     │   │ Centroid Partitioning:                    │
│ 1. Simple Linear Regression (SLR)       │   │ 1. K-Means Clustering (k = 4 Personas)    │
│ 2. Multiple Linear Regression (MLR)     │   │    • Elbow (WSSSE) & Silhouette diagnostic│
│                                         │   │    • Latent 2D PCA cluster projection     │
│ Categorical Prediction (Classification):│   │                                           │
│ 3. Decision Tree Classifier (CART)      │   │ Density Mode Discovery:                   │
│ 4. K-Nearest Neighbors (KNN)            │   │ 2. Mean-Shift Clustering (h = 1.8944)     │
│ 5. Gaussian Naive Bayes (GNB)           │   │    • Non-parametric density modes         │
│ 6. Support Vector Machine (SVM - RBF)   │   │    • Kernel density peak verification     │
└─────────────────────────────────────────┘   └───────────────────────────────────────────┘
```

#### Detailed Breakdown of Models Used and Reasons for Implementation:

#### 1. Simple Linear Regression (SLR)
- **Model Specification**:
  $$\text{preparedness\_score} = \beta_0 + \beta_1 (\text{awareness\_score}) + \epsilon$$
  - Fitted Parameters: Intercept $\beta_0 = 0.4592$, Slope $\beta_1 = 0.7117$.
  - Evaluation Metrics: $R^2 = 0.4498$, $\text{RMSE} = 0.5710$, $\text{MAE} = 0.4756$.
- **Why It Is Used**:
  - Serves as the primary parametric baseline to test the foundational research hypothesis: *Does general flood awareness directly translate into personal preparedness?*
  - Quantifies the direct bivariate explanatory power ($R^2 = 44.98\%$), mathematically proving that cognitive awareness accounts for a substantial proportion of preparedness behavior, while isolating the residual unexplained variance ($\approx 55\%$) attributable to external structural barriers.

#### 2. Multiple Linear Regression (MLR)
- **Model Specification**:
  $$\text{preparedness\_score} = \beta_0 + \beta_1 X_{\text{warning}} + \beta_2 X_{\text{hazard}} + \beta_3 X_{\text{health}} + \beta_4 X_{\text{history}} + \beta_5 C_{\text{supplies}} + \beta_6 C_{\text{cost}} + \epsilon$$
  - Model Fit: $R^2 = 0.4820$, $\text{RMSE} = 0.5541$, $\text{MAE} = 0.4621$.
  - Coefficients: Intercept $= +0.5031$; Warning alerts $= +0.1430$; Residence hazard $= +0.1098$; Health awareness $= +0.4395$; Flood history $= +0.0227$; Supply constraints $= +0.3691$; Financial constraints $= +0.4620$.
- **Why It Is Used**:
  - Disaster preparedness is multidimensional; single-variable models cannot capture the concurrent effects of early warnings, localized hazard proximity, waterborne disease knowledge, past flood trauma, and financial limitations.
  - Determines which specific knowledge domains exert the strongest empirical influence on actionable readiness, revealing that health risk awareness ($\beta = +0.4395$) and resource bottlenecks are stronger behavioral drivers than passive warning monitoring.

#### 3. Decision Tree Classifier (CART)
- **Model Specification**:
  - Criterion: Gini Impurity; $\text{max\_depth} = 3$; Target: `preparedness_level` (*Low*, *Moderate*, *High*).
  - Accuracy: **$71.04\%$**; Weighted $F_1\text{-score} = 0.71$.
  - Feature Importance Hierarchy: `awareness_score` ($64.87\%$), `aw_safety_health_index` ($21.99\%$), `flood_history_n` ($12.47\%$), `aw_warning_index` ($0.66\%$), `aw_local_hazard_index` ($0.00\%$).
- **Why It Is Used**:
  - Unlike "black-box" models, decision trees provide explicit, transparent, and human-interpretable **if-then decision rules**.
  - Identifies the critical tipping-point thresholds (e.g., whether a student with an awareness score $> 3.52$ transitions into high preparedness), allowing school administrators and disaster planners to establish concrete cut-off scores for emergency intervention.

#### 4. $K$-Nearest Neighbors (KNN) Classifier
- **Model Specification**:
  - Distance Metric: Standardized Euclidean distance; Neighborhood size: evaluated across $k \in [1, 15]$.
  - Accuracy: **$66.17\%$** at $k = 5$; optimal performance of **$67.65\%$** achieved at $k = 13$.
- **Why It Is Used**:
  - Operates as a non-parametric, instance-based classifier that makes no prior assumptions regarding underlying Gaussian probability distributions.
  - Effectively captures localized geometric clusters of students who exhibit similar combinations of flood exposure and resource availability, classifying student readiness based on peer similarity.

#### 5. Gaussian Naive Bayes (GNB)
- **Model Specification**:
  - Likelihood Distribution: Continuous Gaussian; zero-variance smoothing applied.
  - Accuracy: **$65.96\%$**; Weighted $F_1\text{-score} = 0.65$.
- **Why It Is Used**:
  - Provides a rigorous probabilistic baseline grounded in Bayes’ Theorem under the assumption of conditional feature independence.
  - Computes posterior probabilities of preparedness category membership, serving as an efficient diagnostic benchmark to test whether feature correlations degrade classification reliability under strict independence constraints.

#### 6. Support Vector Machine (SVM)
- **Model Specification**:
  - Kernel: Radial Basis Function (RBF); Regularization parameter: $C = 1.0$; Features standardized to zero mean and unit variance.
  - Accuracy: **$68.08\%$**; Weighted $F_1\text{-score} = 0.68$.
- **Why It Is Used**:
  - Human disaster behaviors and psychological gaps are non-linearly separable in Euclidean space. The RBF kernel projects features into an infinite-dimensional Hilbert space, constructing optimal maximum-margin separating hyperplanes.
  - Delivers robust classification generalization by minimizing structural risk, avoiding overfitting when mapping subtle boundaries between students who are genuinely prepared versus those who are merely aware.

#### 7. $K$-Means Clustering
- **Model Specification**:
  - Partitioning Algorithm: Lloyd's algorithm on standardized feature vectors (`aw_warning_index`, `aw_local_hazard_index`, `aw_safety_health_index`, `prep_supplies_index`, `prep_planning_index`, `flood_history_n`).
  - Optimization Diagnostics: Evaluated across $k \in [2, 6]$; mathematical elbow confirmed at $k = 4$ ($\text{WSSSE} = 4,480.39$); Silhouette Coefficient $= 0.3101$.
  - Discovered Personas:
    1. *Aware but Unprepared* ($n = 564, 29.83\%$): High warning awareness ($3.64$) but severely deficient emergency supplies ($3.05$) and evacuation planning ($2.10$).
    2. *Low-Exposure Learners* ($n = 526, 27.82\%$): Lowest flood exposure ($1.06$), possessing moderate cognitive awareness but limited practical preparation.
    3. *Resilient Ready Guardians* ($n = 452, 23.90\%$): Highest awareness ($4.34$) and highest tangible preparedness ($4.15$).
    4. *Vulnerable Novices* ($n = 349, 18.46\%$): Critical deficits across awareness ($2.41$) and supplies ($2.29$).
- **Why It Is Used**:
  - Identifies natural, unprompted behavioral archetypes (student disaster personas) without human labeling bias.
  - Provides collegiate administrators with empirical segmentation data to replace generic "one-size-fits-all" orientations with targeted campus interventions (e.g., subsidizing emergency Go-Bags specifically for the *Aware but Unprepared* cohort).

#### 8. Mean-Shift Clustering
- **Model Specification**:
  - Kernel: Flat/Gaussian kernel; Bandwidth parameter: automatically estimated using pairwise distance quantiles ($h = 1.8944$, $\text{quantile} = 0.15$).
  - Results: Automatically identified **4 natural density modes**; Silhouette Coefficient $= 0.2356$.
  - Mode Distribution: Mode 0 ($n = 1,343, 71.02\%$, moderate/vulnerable baseline); Mode 1 ($n = 536, 28.34\%$, resilient/proactive mode); Modes 2 & 3 ($n = 12, 0.63\%$, distinct behavioral outliers).
- **Why It Is Used**:
  - Validates findings from $K$-Means through a completely non-parametric, mode-seeking methodology that **does not require the researchers to pre-specify the number of clusters ($k$)**.
  - Confirms the true topological density structure of the population, demonstrating that student disaster preparedness naturally clusters into two dominant modal densities (a large vulnerable/moderate mass and a distinct proactive cluster).

---

## G. Tools, Libraries, and Computational Environment

The end-to-end data processing, feature engineering, statistical modeling, and machine learning pipeline was implemented within an isolated, fully reproducible Python virtual environment (`.venv`):

- **Host Operating System**: Microsoft Windows 11 Enterprise (64-bit, Build 26100).
- **Programming Language**: Python 3.14.7 (64-bit).
- **Big Data Engine**: **Apache Spark 4.3.0 (`pyspark==4.3.0`)**, operating in local multi-threaded standalone mode (`local[*]`). Custom runtime configurations dynamically set `SPARK_HOME` to resolve parent path spaces on Windows platforms.
- **Java Runtime**: OpenJDK 17 LTS (Eclipse Adoptium 64-bit), required for distributed Spark JVM execution and memory management.
- **Machine Learning & Statistical Libraries**:
  - `scikit-learn==1.9.1`: Execution of SLR, MLR, Decision Tree, KNN, Naive Bayes, SVM, K-Means, Mean-Shift, and evaluation metrics.
  - `scipy==1.18.1`: Statistical distributions and correlation algorithms.
- **Data Engineering & Manipulation**:
  - `pandas==3.1.1`: Tabular transformations, multi-source canonical mapping, and CSV serialization.
  - `numpy==2.5.4`: Vectorized array manipulations, matrix transformations, and numerical operations.
- **Data Visualization Toolchain**:
  - `matplotlib==3.11.2`: Headless publication figure rendering (`matplotlib.use("Agg")`), diverging bar charts, and dual-axis diagnostic plots.
  - `seaborn==0.13.2`: Correlation heatmaps, density curves, and pair plots.
- **Document & PDF Analysis**: `pypdf==7.0.1` for survey codebook verification.
- **Integrated Development Environment (IDE)**: Visual Studio Code configured with Windows PowerShell 5.1 / PowerShell 7, executing scripts via absolute virtual environment binary paths (`.\dsa_final_jm\.venv\Scripts\python.exe`).


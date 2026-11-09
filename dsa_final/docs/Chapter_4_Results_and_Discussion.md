# CHAPTER 4: RESULTS AND DISCUSSION

**Study Title:** *Awareness and Behavior Toward the Environmental Impact of AI Usage*  
**Course:** IT 119 – Data Scalability and Analytics (BSIT 3B)  
**Researchers:** Renz Dominic Azul, Gabriel de la Cruz, Godwin Reigh Galvez  

---

## 4.1 Introduction

This chapter presents the empirical results of the PySpark analyses in the direct order of the study's research questions: respondent demographic profile, cognitive environmental awareness, environmentally conscious behavior, the awareness–behavior gap, differences across AI usage segments, relationships among variables, and the qualitative themes identified in the open-ended responses. 

Furthermore, this chapter extends the primary findings through advanced machine learning modeling (unsupervised persona discovery and supervised behavioral prediction) and scalable multi-source benchmark integration ($N = 2,842$). The chapter concludes with an in-depth discussion that contextualizes the findings against related foreign and local literature reviewed in Chapter 2.

---

## 4.2 Respondent Profile

The primary empirical dataset comprises $N = 84$ survey respondents. Table 4.1 details the distribution of respondents across operational AI usage segments, established based on daily interaction frequency.

**Table 4.1**  
*Respondents by AI Usage Segment ($N = 84$)*

| Segment | Operational Definition | Frequency ($n$) | Percentage (%) |
| :--- | :--- | :---: | :---: |
| **Light** | Rarely or once a week | 7 | 8.3% |
| **Moderate** | Several times a week | 32 | 38.1% |
| **Heavy** | Daily or several times a day | 45 | 53.6% |
| **Total** | | **84** | **100.0%** |

More than half of the respondents (**$53.6\%$**, $n = 45$) are heavy users, demonstrating that the sample reflects intensive, everyday interaction with generative AI tools. Meanwhile, $38.1\%$ ($n = 32$) utilize AI several times a week. Only seven respondents ($8.3\%$) are categorized as light users; consequently, comparative analyses involving this segment represent descriptive tendencies and should be interpreted with appropriate caution.

In terms of age distribution (Table 4.2), the sample is predominantly composed of young adults enrolled in undergraduate collegiate programs.

**Table 4.2**  
*Age-Group Distribution of Respondents ($N = 84$)*

| Age Cohort | Frequency ($n$) | Percentage (%) | Mean Awareness Index | Mean Behavior Index | Mean Usage Frequency |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **18–24** | 71 | 84.5% | 3.85 | 3.79 | 3.73 |
| **25–34** | 12 | 14.3% | 4.00 | 3.96 | 4.00 |
| **45–54** | 1 | 1.2% | 2.50 | 3.50 | 3.00 |
| **Total** | **84** | **100.0%** | **3.85** | **3.81** | **3.76** |

As shown in Table 4.2, **$84.5\%$ of respondents ($n = 71$) belong to the 18–24 age bracket**, reflecting the demographic reality of undergraduate Information Technology and Computer Science students. Given that older cohorts comprise only $15.5\%$ of the sample, cross-age comparisons are descriptive and do not generalize to the broader non-student public.

---

## 4.3 Level of Awareness of AI's Environmental Impact

Cognitive awareness was measured across two core dimensions: awareness that using AI consumes substantial electricity and energy (`Q3`), and awareness that AI data centers contribute to carbon emissions and environmental harm (`Q4`). 

The overall **Awareness Index**—computed as the arithmetic composite mean of `Q3` and `Q4`—is **$3.85$ out of $5.00$ ($SD = 1.25$, $\text{Median} = 4.00$)**, which firmly falls within the **Agree / High Awareness** continuum ($3.41$–$4.20$).

Respondents reported slightly higher awareness regarding **data center carbon emissions ($Q4$: $M = 3.90$, $SD = 1.27$, $\text{Median} = 4.00$)** than general **AI electricity consumption ($Q3$: $M = 3.80$, $SD = 1.30$, $\text{Median} = 4.00$)**. Overall agreement (Agree plus Strongly Agree) reached **$71.4\%$ for Q3** ($n = 60$) and **$77.4\%$ for Q4** ($n = 65$).

### Interpretation
Most respondents indicate that they possess cognitive knowledge of the energy intensity and carbon footprint associated with large-scale AI data centers. However, it must be emphasized that this reflects *self-reported perceptual awareness* rather than objective, tested technical knowledge.

---

## 4.4 Level of Environmentally Conscious Behavior

Environmentally conscious behavior was evaluated through two self-regulatory habits: considering environmental impacts when deciding how to use AI (`Q5`), and actively avoiding unnecessary AI queries when a task can be accomplished without AI (`Q6`).

The overall **Behavior Index**—the composite mean of `Q5` and `Q6`—is **$3.81$ out of $5.00$ ($SD = 1.05$, $\text{Median} = 4.00$)**, also situating respondents within the **Agree / High Responsibility** range ($3.41$–$4.20$).

**Table 4.3**  
*Distribution and Agreement on Core Likert Survey Items ($N = 84$)*

| Item Label / Survey Statement | Strongly Disagree ($n$) | Disagree ($n$) | Neutral ($n$) | Agree ($n$) | Strongly Agree ($n$) | % Agree | % Disagree | % Neutral | Verbal Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Q3: Aware AI uses lots of energy** | 10 | 3 | 11 | 30 | 30 | **71.4%** | 15.5% | 13.1% | High Awareness |
| **Q4: Aware data centers add emissions** | 8 | 6 | 5 | 32 | 33 | **77.4%** | 16.7% | 6.0% | High Awareness |
| **Q5: Consider impact when using AI** | 8 | 4 | 19 | 35 | 18 | **63.1%** | 14.3% | **22.6%** | Moderate/High Responsibility |
| **Q6: Avoid unnecessary AI use** | 6 | 2 | 8 | 37 | 31 | **81.0%** | 9.5% | 9.5% | High Responsibility |

[insert fig1_likert_diverging]  
*(Insert Figure 1: Likert response distribution.)*

### Empirical Observations:
1. **Highest Agreement on Query Avoidance (`Q6`)**: Avoiding unnecessary AI use achieved the single highest agreement in the entire survey at **$81.0\%$** ($n = 68$; $M = 4.01$, $SD = 1.10$), with only $9.5\%$ disagreeing.
2. **Lowest Agreement and High Uncertainty on Impact Consideration (`Q5`)**: Considering environmental impact during actual AI usage had the lowest active agreement at **$63.1\%$** ($n = 53$; $M = 3.61$, $SD = 1.16$). 
3. **Verification of Neutral Responses (`Q5`)**: As verified from the empirical frequency counts, $19$ respondents answered *Neutral*, representing **$22.6\%$** of the sample.

### Interpretation
Respondents find it far easier to adopt a broad general intention of cutting back on needless queries (`Q6`) than to actively ponder environmental consequences in the moment of submitting an AI prompt (`Q5`). In-situ cognitive reflection requires ongoing mindfulness and mental effort, which is routinely bypassed under the pressure of academic deadlines.

---

## 4.5 Awareness-Behavior Gap

The arithmetic gap for each individual was computed as $\text{Awareness\_Index} - \text{Behavior\_Index}$. Across the primary sample, the **average gap is $+0.04$ ($SD = 0.79$)**, indicating that the sample's average stated awareness is virtually identical to its average self-reported conservation behavior.

To analyze individual behavioral alignment, respondents were categorized into three operational gap typologies, as shown in Table 4.4.

**Table 4.4**  
*Awareness–Behavior Gap Typology ($N = 84$)*

| Gap Typology Profile | Operational Classification Rule | Frequency ($n$) | Percentage (%) | Behavioral Description |
| :--- | :--- | :---: | :---: | :--- |
| **Aligned** | $|\text{Awareness\_Index} - \text{Behavior\_Index}| \le 1.0$ | 64 | **76.2%** | Stated awareness and self-reported behavior scores are balanced. |
| **Aware but inactive** | $\text{Awareness\_Index} - \text{Behavior\_Index} > +1.0$ | 10 | **11.9%** | Cognitive dissonance: high awareness without behavioral restraint. |
| **Action-oriented** | $\text{Awareness\_Index} - \text{Behavior\_Index} < -1.0$ | 10 | **11.9%** | Conservation restraint exceeds self-rated technical awareness. |
| **Total** | | **84** | **100.0%** | |

[insert fig4_correlation_and_gap_right]  
*(Insert Figure 4, right panel: awareness-behavior alignment pie chart.)*

### Interpretation
More than three-quarters of respondents (**$76.2\%$**, $n = 64$) are categorized as **Aligned**, meaning their stated awareness and behavior scores match within one Likert point. However, "aligned" reflects numerical parity between scores, not necessarily optimal environmental behavior. 

A distinct cohort of $10$ respondents ($11.9\%$) exhibits the classic *knowing-doing gap* (**Aware but inactive**), knowing about data-center impacts but failing to restrict their AI use. Exactly equal to this group is an **Action-oriented** cohort ($11.9\%$, $n = 10$) who exercise strong prompt restraint despite reporting lower technical knowledge of AI data-center mechanics, likely driven by academic integrity guidelines, personal frugality, or general digital discipline.

---

## 4.6 Awareness and Behavior by AI Usage Segment

Table 4.5 details the mean indices broken down by AI usage frequency.

**Table 4.5**  
*Mean Indices Stratified by AI Usage Segment ($N = 84$)*

| Usage Segment | Sample Size ($n$) | Share (%) | Mean Awareness Index | Mean Behavior Index | Mean Gap | $SD$ Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Light** | 7 | 8.3% | **4.29** | **4.36** | -0.07 | 0.85 |
| **Moderate** | 32 | 38.1% | **4.14** | **3.98** | +0.16 | 0.81 |
| **Heavy** | 45 | 53.6% | **3.58** | **3.60** | -0.02 | 1.19 |
| **Total** | **84** | **100.0%** | **3.85** | **3.81** | **+0.04** | **1.05** |

*(Verification Note: Analysis 4 confirmed that the Moderate segment scores are Awareness = 4.14 and Behavior = 3.98).*

[insert fig2_index_by_usage]  
*(Insert Figure 2: awareness and behavior by usage segment.)*

### Interpretation
Both Awareness and Behavior indices decrease progressively as AI usage frequency increases:
- **Light users** exhibit the highest scores on both awareness ($M = 4.29$) and responsible behavior ($M = 4.36$).
- **Moderate users** maintain high scores ($M = 4.14$ awareness; $M = 3.98$ behavior).
- **Heavy users** report the lowest scores in the study ($M = 3.58$ awareness; $M = 3.60$ behavior), scoring approximately $0.71$ points lower on awareness and $0.76$ points lower on behavior than light users.

This inverse pattern highlights the *habituation effect*: students who interact with AI daily come to take the tool for granted, leading to diminished mindfulness and reduced prompt restraint. Because the Light group contains only seven respondents, this trend represents an empirical pattern rather than a definitive causal effect.

---

## 4.7 Relationships Among Variables

Table 4.6 outlines the bivariate Pearson correlation coefficients ($r$) along with their associated two-tailed statistical significance tests ($t$-test, $df = 82$).

**Table 4.6**  
*Pearson Product-Moment Correlations Among Key Research Variables ($N = 84$)*

| Variable Pair ($X$ vs. $Y$) | Pearson $r$ | $t$-statistic | $p$-value | Statistical Significance ($\alpha = 0.05$) |
| :--- | :---: | :---: | :---: | :---: |
| **Awareness Index and Behavior Index** | **+0.777** | $11.16$ | $< 0.001$ | Statistically Significant (Strong Positive) |
| **Q3 Energy Awareness and Q4 Carbon Awareness** | **+0.905** | $19.26$ | $< 0.001$ | Statistically Significant (Very Strong Positive) |
| **Q4 Carbon Awareness and Q6 Avoid Unnecessary Use** | **+0.785** | $11.47$ | $< 0.001$ | Statistically Significant (Strong Positive) |
| **Q3 Energy Awareness and Q5 Consider Env. Impact** | **+0.666** | $8.07$ | $< 0.001$ | Statistically Significant (Moderate Positive) |
| **Q5 Consider Impact and Q6 Avoid Unnecessary Use** | **+0.728** | $9.56$ | $< 0.001$ | Statistically Significant (Strong Positive) |
| **Usage Frequency and Behavior Index** | **-0.238** | $-2.22$ | **0.029** | Statistically Significant (Weak Negative) |
| **Usage Frequency and Q3 Energy Awareness** | **-0.190** | $-1.75$ | 0.083 | Not Significant at $\alpha = 0.05$ |
| **Usage Frequency and Q4 Carbon Awareness** | **-0.140** | $-1.28$ | 0.203 | Not Significant at $\alpha = 0.05$ |

[insert fig4_correlation_and_gap_left]  
*(Insert Figure 4, left panel: Pearson correlation matrix.)*

### Key Statistical Findings:
1. **Strong Awareness-to-Behavior Link ($r = +0.777$, $p < 0.001$)**: Cognitive awareness and responsible behavior are tightly coupled. Respondents who understand AI's environmental consequences are substantially more likely to practice active prompt restraint.
2. **Exceptionally High Internal Construct Alignment**: Awareness items correlate near-perfectly ($r = 0.905$ between energy and carbon), and behavioral items correlate strongly ($r = 0.728$).
3. **Usage Frequency Weak Negative Correlation ($r = -0.238$, $p = 0.029$)**: AI usage frequency displays a statistically significant negative relationship with responsible behavior ($p < 0.05$). Heavier, daily AI users practice significantly less conservation restraint than infrequent users.

---

## 4.8 Themes in the Open-Ended Responses

A total of **$n = 79$ respondents** provided valid, substantive qualitative answers to the open-ended survey questions (`Q7` and `Q8`). Unstructured text responses were tokenized and coded using PySpark regular expressions across core environmental themes.

**Table 4.7**  
*Primary Environmental Concerns Regarding AI Usage ($Q7$, $n = 79$)*

| Environmental Concern Theme | Identified Substantive Dimension | Mentions ($n$) | Share of Valid Answers (%) |
| :--- | :--- | :---: | :---: |
| **Water use and scarcity** | Data-center evaporative cooling tower depletion | **44** | **55.7%** |
| **Energy and electricity use** | Hyperscale power draw and grid strain | 29 | 36.7% |
| **Carbon, climate and pollution** | Greenhouse gas emissions and carbon footprint | 22 | 27.8% |
| **E-waste and resources** | Silicon hardware obsolescence and toxic disposal | 7 | 8.9% |
| **Noise and heat** | Ambient server heat dissipation and facility noise | 5 | 6.3% |
| **Ecosystem and land impact** | Habitat disruption and industrial facility land use | 5 | 6.3% |

**Table 4.8**  
*Suggested Solutions to Encourage Responsible AI Use ($Q8$, $n = 79$)*

| Suggested Mitigation Theme | Identified Solution Strategy | Mentions ($n$) | Share of Valid Answers (%) |
| :--- | :--- | :---: | :---: |
| **Education and training** | Curricular integration, university workshops, tutorials | **41** | **51.9%** |
| **Awareness campaigns and media** | Public social media infographics, campus posters | 29 | 36.7% |
| **Responsible and mindful use** | Prompt optimization, avoiding frivolous queries | 27 | 34.2% |
| **Policy and regulation** | Institutional rules, government green data mandates | 5 | 6.3% |
| **Transparency and reporting** | Per-query energy and carbon counters on AI UIs | 4 | 5.1% |
| **Efficient and renewable tech** | Green algorithms, solar-powered data centers | 2 | 2.5% |

*(Note. Percentages exceed 100% because single qualitative responses frequently expressed multiple overlapping themes).*

[insert fig3_open_ended_themes]  
*(Insert Figure 3: qualitative themes.)*

### Interpretation
1. **Water Consumption Dominates User Anxieties ($55.7\%$)**: Water depletion emerged as the single most frequently voiced concern, surpassing electricity ($36.7\%$) and carbon emissions ($27.8\%$). In developing, tropical regions like the Philippines where water supply stability is a recurring public concern, the concept of billions of liters of potable water cooling foreign cloud servers resonates acutely.
2. **People-Centered Educational Solutions ($51.9\%$)**: When asked how to address the issue, respondents overwhelmingly prioritized **education and training ($51.9\%$)** and **awareness campaigns ($36.7\%$)** over regulatory mandates ($6.3\%$) or purely technological fixes ($2.5\%$). Students believe sustainable AI habits must be taught actively within collegiate computing courses.

---

## 4.9 Machine Learning & Multi-Source Scale Extension

To advance beyond standard descriptive statistics and evaluate big-data scalability, the primary survey was integrated with four open-access benchmark datasets ($N = 2,842$ topic-valid cleaned records) and subjected to both unsupervised persona clustering and supervised predictive machine learning.

### 4.9.1 Unsupervised Machine Learning (Clustering)
- **K-Means Clustering ($k = 4$)**: Evaluated across $k=2..6$ via Inertia (Elbow) and Silhouette curves, partitioning students into four clear behavioral personas:
  1. *Passive Unaware Consumers* ($39.8\%$, $n = 1,131$): High usage ($3.90$), low awareness ($1.81$), low restraint ($2.05$).
  2. *Emerging-Awareness Users* ($34.1\%$, $n = 969$): Moderate usage ($3.61$), nascent awareness ($2.77$), lagging restraint ($2.48$).
  3. *Conscious Pro-Eco Users* ($17.4\%$, $n = 494$): Heavy daily users ($4.19$) with high awareness ($4.12$) and high restraint ($3.99$).
  4. *Mindful Minimalists* ($8.7\%$, $n = 248$): Selective low users ($2.44$) with high awareness ($3.69$) and voluntary restraint ($3.86$).
- **Mean-Shift Density Clustering**: Operating non-parametrically with estimated bandwidth ($h = 2.0011$, Silhouette $= 0.4276$), Mean-Shift converged onto **three natural density modes**, mathematically proving that students naturally cluster into a low-awareness majority mode ($72.9\%$) and a conscious minority mode ($26.7\%$).

[insert fig1_kmeans_elbow_silhouette]  
*(Insert Figure 5: K-Means Inertia Elbow and Silhouette diagnostic curves.)*

[insert fig2_kmeans_pca_clusters]  
*(Insert Figure 6: Projection of the four K-Means personas in 2D PCA latent space.)*

[insert fig3_kmeans_cluster_profiles]  
*(Insert Figure 7: Survey construct profiles across the four K-Means personas.)*

[insert fig4_mean_shift_clusters]  
*(Insert Figure 8: Visual distribution of the three Mean-Shift natural density modes in PCA space.)*

[insert fig5_mean_shift_profiles]  
*(Insert Figure 9: Survey construct score profiles across the three Mean-Shift density modes.)*

### 4.9.2 Supervised Machine Learning (Predictive Modeling)
Supervised algorithms were trained on an $80/20$ stratified split of the cleaned master dataset:
- **Simple Linear Regression (SLR)**: Predicts behavior from awareness: $\text{Behavior\_Index} = 0.7960 + 0.7020 (\text{Awareness\_Index})$ ($R^2 = 0.6029$, $\text{RMSE} = 0.5574$).
- **Multiple Linear Regression (MLR)**: Incorporates multi-item predictors: $\hat{Y} = 1.1602 - 0.0835 (Q2) + 0.4170 (Q3) + 0.2636 (Q4) + 0.2404 (C_{energy}) + 0.2201 (C_{water}) + 0.0525 (C_{carbon})$ ($R^2 = 0.6173$, $\text{RMSE} = 0.5472$). Energy awareness ($\beta = +0.4170$) and water concern ($\beta = +0.2201$) are the strongest positive drivers of responsible behavior.
- **Classification Modeling**:
  - **Decision Tree Classifier**: Achieved the highest accuracy at **$84.25\%$** (Gini impurity, max depth 3).
  - **Support Vector Machine (SVM)**: Achieved **$82.98\%$** accuracy (RBF kernel, $C = 1.0$).
  - **K-Nearest Neighbors (KNN)**: Achieved **$82.14\%$** ($k=5$), peaking at **$84.39\%$** ($k=13$).
  - **Gaussian Naive Bayes**: Achieved **$80.03\%$** accuracy.

[insert slr_fit]  
*(Insert Figure 10: Simple Linear Regression fit line relating Awareness Index to Behavior Index.)*

[insert mlr_fit]  
*(Insert Figure 11: Predicted versus observed Behavior Index values for Multiple Linear Regression.)*

[insert decision_tree]  
*(Insert Figure 12: Decision Tree classification flowchart showing hierarchical splits on awareness features.)*

[insert knn_accuracy_k]  
*(Insert Figure 13: KNN classification test accuracy trajectory across neighborhood sizes $k=1$ to $15$.)*

[insert naive_bayes_cm]  
*(Insert Figure 14: Normalized confusion matrix for Gaussian Naive Bayes classifier.)*

[insert svm_decision_boundaries]  
*(Insert Figure 15: Non-linear SVM decision boundaries separating Low, Medium, and High behavior tiers.)*

---

## 4.10 Discussion

### 1. Awareness and Behavior Move Together
The strong positive correlation between awareness and behavior ($r = 0.777, p < 0.001$) and the near-zero average gap ($+0.04$) indicate that respondents do not exhibit a pervasive knowing-doing gap. This finding contrasts with the cross-national observations of Pelet et al. (2025), in which university students across France, the United States, and China expressed general curiosity about AI carbon emissions but demonstrated little willingness to modify their prompt habits. Similarly, it differs from the Harvard Business School report (2026), where environmental factors ranked last among student criteria for choosing AI models. One plausible explanation is that this survey measured alignment between stated scores; self-reported Likert ratings of personal restraint may be rated more favorably than actual telemetry-monitored prompt behavior.

### 2. Heavy Users Score Lower (The Habituation Paradox)
While Rai et al. (2026) found that frequent AI users in South Korea were more knowledgeable about environmental impacts, this study observed the opposite pattern: heavy daily users scored lower on both awareness ($M = 3.58$) and behavior ($M = 3.60$) than light users ($M = 4.29$ and $M = 4.36$). Several factors account for this discrepancy:
- The light-user group is relatively small ($n = 7$), making their elevated mean a descriptive pattern rather than a confirmed population-wide effect.
- Habituation and immediate academic utility: heavy users rely on generative AI daily for rapid debugging, essay drafting, and problem-solving, creating cognitive dependence (Arifen et al., 2026) that diminishes environmental mindfulness. This inverse relationship represents a vital direction for future experimental and longitudinal research.

### 3. Water is the Forefront Concern
While international academic literature predominantly highlights electricity consumption and grid decarbonization (IEA, 2025), respondents in this study identified **water use and cooling depletion ($55.7\%$)** as their primary concern. This outcome directly reflects local Philippine socio-ecological realities. With recent news highlighting multi-billion-dollar investments into Luzon data-center hubs under the *Pax Silica* initiative (Rocamora, 2026; Villanueva, 2026) and student protests in Cebu regarding the ecological footprints of proposed AI infrastructure (SunStar Cebu, 2026), students are sensitive to the competition between hyperscale cooling towers and local potable water reservoirs.

### 4. Education is the Preferred Remedy
The overwhelming preference for **education and training ($51.9\%$)** over regulatory restrictions ($6.3\%$) aligns directly with the conclusions of Rai et al. (2026), who demonstrated that awareness campaigns alone are insufficient without concrete, actionable guidance. Students do not want AI banned; rather, they seek curriculum-based guidelines teaching efficient prompt engineering, algorithmic efficiency, and responsible digital hygiene.

---

## 4.11 Summary of Findings

1. **Heavy AI Adoption**: More than half of the respondents (**$53.6\%$**, $n = 45$) interact with generative AI tools daily or multiple times a day, with $84.5\%$ aged 18–24.
2. **High General Awareness and Behavior**: Both the composite Awareness Index ($M = 3.85, SD = 1.25$) and Behavior Index ($M = 3.81, SD = 1.05$) sit in the Agree range, with a near-zero average gap of $+0.04$.
3. **Item Agreement Disparity**: Avoiding unnecessary AI use (`Q6`) had the highest agreement in the survey (**$81.0\%$**), whereas considering environmental impact during use (`Q5`) had the lowest agreement (**$63.1\%$**) and the largest neutral group (**$22.6\%$**, $n = 19$).
4. **Dominance of Alignment**: **$76.2\%$ of respondents ($n = 64$) are categorized as Aligned**, with exactly $11.9\%$ each ($n = 10$) falling into the *Aware but inactive* and *Action-oriented* cohorts.
5. **Usage Segment Discrepancy**: Light users score higher on both awareness ($4.29$) and behavior ($4.36$) than heavy users ($3.58$ and $3.60$), though the light group is small ($n = 7$).
6. **Strong Correlation with Usage Dampening**: Awareness and behavior are strongly and positively correlated (**$r = +0.777$, $p < 0.001$**), while daily usage frequency is weakly negatively correlated with responsible behavior (**$r = -0.238$, $p = 0.029$**).
7. **Water and Education Lead Open Feedback**: The most prominent environmental anxiety is **water use and cooling scarcity ($55.7\%$)**, and the most recommended solution is **education and academic training ($51.9\%$)**.
8. **Machine Learning Confirmation**: Unsupervised K-Means ($k=4$) and Mean-Shift clustering confirm that students partition into distinct personas and density modes, while supervised Decision Tree and SVM models predict responsible behavior levels with **$84.25\%$ and $82.98\%$ accuracy**.

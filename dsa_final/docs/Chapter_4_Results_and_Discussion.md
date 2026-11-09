# CHAPTER 4: RESULTS AND DISCUSSION

**Study Title:** *Awareness and Behavior Toward the Environmental Impact of AI Usage*  
**Course:** IT 119 – Data Scalability and Analytics (BSIT 3B)  
**Researchers:** Renz Dominic Azul, Gabriel de la Cruz, Godwin Reigh Galvez  

---

## 4.1 Introduction

This chapter presents the empirical results of the PySpark analyses in the direct order of the study's research questions: respondent demographic profile, cognitive environmental awareness, environmentally conscious behavior, the awareness–behavior gap, differences across AI usage segments, relationships among variables, and the qualitative themes identified in the open-ended responses. 

Furthermore, this chapter extends the primary findings through advanced machine learning modeling (unsupervised persona clustering and supervised behavioral prediction) and scalable multi-source benchmark integration ($N = 2,842$). Every presented figure is accompanied by an in-depth analytical discussion detailing its visual patterns, behavioral implications, and academic context. The chapter concludes with a synthesis comparing the empirical findings against related foreign and local literature reviewed in Chapter 2.

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

### Discussion of Figure 1: Likert Response Distribution Across Awareness and Behavior Items
Figure 1 visualizes the proportion of agreement, neutrality, and disagreement across the four primary Likert items (`Q3` through `Q6`). 

Several prominent visual and statistical dynamics emerge:
1. **The Disparity Between Conservation Intention (`Q6`) and Active In-Situ Reflection (`Q5`)**: Visually, item `Q6` (*avoiding unnecessary AI use*) displays the largest positive green bar in the chart, capturing an overwhelming **$81.0\%$ agreement** ($n = 68$) and the lowest disagreement ($9.5\%$). In sharp contrast, item `Q5` (*considering environmental impact during AI use*) exhibits the smallest agreement bar (**$63.1\%$**, $n = 53$) and a substantially expanded gray neutral section (**$22.6\%$**, $n = 19$).
2. **The "In-the-Moment" Cognitive Barrier**: This visual contrast illustrates that students readily embrace the general, retrospective concept of limiting wasteful usage, but struggle to actively reflect on environmental costs in real time while formulating queries. Submitting a prompt is instantaneous and frictionless; stopping to deliberate on server cooling water or grid carbon while coding or finishing an assignment imposes cognitive friction that most students bypass.
3. **High Awareness Baseline (`Q3` & `Q4`)**: The top two bars show robust majorities agreeing that AI demands extensive energy ($71.4\%$) and produces data-center emissions ($77.4\%$). Disagreement remains confined to a minor fraction ($15.5\%$ and $16.7\%$), confirming that collegiate computing students have moved past the initial era of digital innocence where cloud AI was perceived as weightless and environmentally free.

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

### Discussion of Figure 4 (Right Panel): Awareness–Behavior Alignment Typology
The right panel of Figure 4 illustrates the proportional segmentation of respondents according to their behavioral alignment profile.

Key insights derived from this visualization include:
1. **Dominance of Numerical Alignment ($76.2\%$)**: More than three-quarters of respondents occupy the dominant **Aligned** slice ($n = 64$). This indicates that for the vast majority of students, their self-reported actions correspond closely to their acknowledged level of awareness. It is essential to note that "aligned" signifies balance between scores rather than absolute eco-friendliness; an individual with low awareness who practices low restraint is just as "aligned" as an individual with high awareness who exercises high restraint.
2. **Symmetrical Outlier Cohorts ($11.9\%$ vs. $11.9\%$)**: The visualization reveals an exact symmetrical split among non-aligned respondents:
   - **Aware but inactive ($11.9\%$, $n = 10$)**: This slice represents the classic *attitude-behavior gap* documented in consumer literature (Pelet et al., 2025). These students acknowledge AI's high carbon and energy costs but fail to modify their usage habits, prioritizing academic productivity and personal convenience over ecological restraint.
   - **Action-oriented / Acting beyond awareness ($11.9\%$, $n = 10$)**: This slice captures respondents whose prompt conservation surpasses their declared technical awareness of data center mechanics. Their restraint is often anchored in secondary motivations, such as personal digital discipline, academic integrity policies prohibiting over-reliance on AI, or cost-saving habits regarding paid token limits.

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

### Discussion of Figure 2: Awareness and Behavior Indices Across AI Usage Segments
Figure 2 visually presents the comparison of composite Awareness Index (blue bars) and Behavior Index (orange bars) across Light, Moderate, and Heavy AI user groups.

The chart reveals a pronounced, progressive inverse trend:
1. **The Downward Step Pattern**: Both awareness and behavioral restraint decline steadily as AI interaction frequency intensifies. Light users anchor the upper tier with peak scores ($M = 4.29$ awareness; $M = 4.36$ behavior). Moderate users follow closely ($M = 4.14$ awareness; $M = 3.98$ behavior). Heavy daily users exhibit the lowest scores in the entire study ($M = 3.58$ awareness; $M = 3.60$ behavior).
2. **The Habituation and Cognitive Dependence Trap**: This visual gradient provides compelling empirical evidence of the *habituation paradox*. Heavy users—who generate the vast majority of server-side compute workloads—score approximately $0.71$ points lower on awareness and $0.76$ points lower on conservation behavior than light users. Daily reliance creates cognitive dependence (Arifen et al., 2026), whereby the habitual convenience of instantaneous AI responses desensitizes students to the unseen environmental externalities of their queries.
3. **Sample Context**: Because the Light group contains $n = 7$ respondents, this pronounced divergence represents an exploratory behavioral pattern rather than an absolute population effect, underscoring the necessity of integrating multi-source benchmark datasets to validate scalability.

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

### Discussion of Figure 4 (Left Panel): Pearson Correlation Matrix
The left panel of Figure 4 displays the correlation matrix heatmap, employing color intensity to visualize the magnitude and direction of associations among survey variables.

Key takeaways highlighted by the heatmap include:
1. **A Dense Cluster of Strong Positive Associations ($r = 0.666$ to $0.905$)**: The central quadrant of the heatmap exhibits dark, saturated hues, reflecting robust mutual reinforcement among awareness and behavior items. Most notably, `Q3` (energy awareness) and `Q4` (carbon awareness) display an exceptionally strong correlation of **$r = +0.905$ ($p < 0.001$)**, confirming that students view electrical consumption and data-center emissions as indivisible facets of a unified ecological impact.
2. **Awareness as the Bedrock of Responsible Behavior ($r = +0.777$)**: The correlation between composite Awareness and composite Behavior is strong, positive, and highly significant ($t = 11.16, p < 0.001$). This confirms that higher cognitive awareness is directly associated with proactive conservation habits.
3. **The Distinct Negative Strip of Usage Frequency**: In stark contrast to the positive core, the rows corresponding to AI usage frequency (`usage_freq_n`) present light and negative coefficients against Behavior Index (**$r = -0.238$, $p = 0.029$**), energy awareness ($r = -0.190$), and carbon awareness ($r = -0.140$). This visual contrast reinforces that intensive usage does not foster eco-literacy; on the contrary, it introduces behavioral inertia that dampens willingness to restrict prompt submissions.

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

### Discussion of Figure 3: Qualitative Themes in Environmental Concerns and Solutions
Figure 3 displays horizontal bar charts comparing the relative prevalence of qualitative themes extracted from open-ended responses regarding environmental concerns (`Q7`, top panel) and recommended solutions (`Q8`, bottom panel).

Detailed observations include:
1. **The Prominence of Water Resource Depletion ($55.7\%$)**: In the top panel, **Water use and scarcity** decisively leads all concerns ($n = 44, 55.7\%$), outranking electricity consumption ($36.7\%$) and carbon emissions ($27.8\%$). This is a critical empirical discovery: while Western research and corporate reporting center almost exclusively on net-zero carbon pledges and electrical grids (IEA, 2025), local Philippine collegiate respondents are acutely concerned about the billions of gallons of freshwater consumed by data-center evaporative cooling systems. In tropical regions prone to seasonal water shortages, respondents express visceral concern over allocating potable municipal water to cool generative AI clusters.
2. **Prioritization of People-Centered Education Over Bureaucracy**: In the bottom panel, **Education and training** forms the longest bar (**$51.9\%$**, $n = 41$), followed by **Awareness campaigns** ($36.7\%$) and **Responsible personal use** ($34.2\%$). In contrast, top-down legislative interventions (**Policy and regulation: $6.3\%$**) and corporate reporting (**Transparency: $5.1\%$**) occupy the bottom ranks. Students do not advocate for prohibitive bans or government quotas; instead, they advocate for institutional empowerment, requesting that computing curricula teach prompt efficiency, green software engineering, and mindful AI utilization.

---

## 4.9 Machine Learning & Multi-Source Scale Extension

To evaluate big-data scalability and discover latent behavioral patterns beyond simple linear correlations, the primary Philippine dataset was harmonized with four open-access benchmark datasets, creating a unified analytical sample of **$N = 2,842$ topic-valid records**. This corpus was evaluated using both unsupervised clustering algorithms and supervised predictive models.

### 4.9.1 Unsupervised Machine Learning (Clustering)

#### K-Means Model Selection and Latent Space Mapping
K-Means clustering was executed on the standardized five-item matrix (`Q2` through `Q6`), evaluating cluster counts from $k = 2$ to $k = 6$.

[insert fig1_kmeans_elbow_silhouette]  
*(Insert Figure 5: K-Means Inertia Elbow and Silhouette diagnostic curves.)*

### Discussion of Figure 5: Diagnostic Evaluation of K-Means Cluster Solutions
Figure 5 plots two complementary mathematical criteria across candidate cluster counts $k = 2$ through $6$: within-cluster inertia (WSSSE, left axis) and the Silhouette Coefficient (right axis).
- **The Mathematical Elbow Point**: The inertia curve displays an initial steep descent from $k = 2$ ($7,463.14$) to $k = 3$ ($6,197.47$), forming an elbow inflection at **$k = 4$ ($5,344.45$)**, beyond which marginal inertia reductions diminish ($4,685.87$ at $k = 5$; $4,303.63$ at $k = 6$).
- **Silhouette Trade-off**: Although $k = 2$ yields the highest mathematical silhouette score ($0.4566$), it merely bifurcates the sample into high vs. low overall agreement. In contrast, **$k = 4$ ($s = 0.2426$)** preserves healthy cluster separation while capturing four nuanced, actionable user personas that align with empirical survey theory.

[insert fig2_kmeans_pca_clusters]  
*(Insert Figure 6: Projection of the four K-Means personas in 2D PCA latent space.)*

### Discussion of Figure 6: 2D PCA Latent Space Separation of K-Means Personas
Figure 6 visualizes the four K-Means personas projected onto a two-dimensional latent space constructed via Principal Component Analysis ($\text{PC}_1 = 63.20\%$ variance; $\text{PC}_2 = 19.86\%$ variance). The cluster centroids are marked with bold crosshairs.
- **Horizontal Axis ($\text{PC}_1$ - Eco-Awareness and Conservation)**: $\text{PC}_1$ cleanly separates uninformed users on the left from conscious users on the right. *Passive Unaware Consumers* (Cluster 3, red) occupy the far left quadrant ($\text{PC}_1 < 0$), whereas *Conscious Pro-Eco Users* (Cluster 2, green) are clustered on the positive right quadrant.
- **Vertical Axis ($\text{PC}_2$ - AI Usage Intensity)**: $\text{PC}_2$ captures daily interaction frequency. *Mindful Minimalists* (Cluster 1, blue) are located near the bottom of the plane, indicating low usage frequency combined with rightward positive environmental restraint.
- **Centroid Dispersion**: The four centroids are well-dispersed across the coordinate space, verifying that K-Means successfully identified discrete behavioral configurations without degenerate centroid collapse.

[insert fig3_kmeans_cluster_profiles]  
*(Insert Figure 7: Survey construct profiles across the four K-Means personas.)*

### Discussion of Figure 7: Multi-Construct Behavioral Profiles of K-Means Personas
Figure 7 presents a comparative bar chart detailing the mean construct scores (`usage_freq_n`, `aware_energy_n`, `aware_carbon_n`, `consider_impact_n`, `avoid_use_n`) across each of the four K-Means personas:
1. **Passive Unaware Consumers ($39.8\%$, $n = 1,131$)**: Highest usage frequency ($3.90$) paired with the lowest awareness ($1.81$) and lowest restraint ($2.05$). This is the largest segment in the expanded dataset, reflecting widespread uncritical AI adoption.
2. **Emerging-Awareness Users ($34.1\%$, $n = 969$)**: Moderate daily usage ($3.61$) with intermediate awareness ($2.77$), but behavioral restraint ($2.48$) lags behind awareness, embodying nascent knowledge that has not yet materialized into daily habits.
3. **Conscious Pro-Eco Users ($17.4\%$, $n = 494$)**: Intensive daily users ($4.19$) who nevertheless score at the ceiling for environmental awareness ($4.12$) and active query restraint ($3.99$). They prove that heavy AI usage does not preclude environmental mindfulness if proper literacy is established.
4. **Mindful Minimalists ($8.7\%$, $n = 248$)**: Selective, infrequent users ($2.44$) who exhibit high awareness ($3.69$) and the highest query avoidance ($3.86$), practicing voluntary technological restraint.

---

#### Mean-Shift Density Mode Clustering
To validate whether these four segments represent true underlying data modes rather than artificial geometric partitions, the non-parametric **Mean-Shift algorithm** was deployed with estimated bandwidth ($h = 2.0011$).

[insert fig4_mean_shift_clusters]  
*(Insert Figure 8: Visual distribution of the three Mean-Shift natural density modes in PCA space.)*

### Discussion of Figure 8: Natural Density Clumping in Mean-Shift Modes
Figure 8 illustrates the three natural density modes identified by Mean-Shift without pre-imposing a cluster count $k$.
- **Bimodal Structural Dominance**: Mean-Shift achieved a remarkably high Silhouette Score of **$0.4276$**, identifying that the population naturally condenses into two massive density peaks: **Mode 0 ($72.87\%$, $n = 2,071$)** in purple and **Mode 1 ($26.67\%$, $n = 758$)** in yellow, alongside a microscopic outlier cluster (**Mode 2: $0.46\%$, $n = 13$**).
- **Density Confirmation of the Student Divide**: The spatial distribution in Figure 8 confirms that the vast majority of students clump tightly around the low-awareness pole (Mode 0), while a distinct, cohesive minority forms a separate density mode around high awareness and restraint (Mode 1). This proves that student AI behavior is inherently polarized rather than normally distributed.

[insert fig5_mean_shift_profiles]  
*(Insert Figure 9: Survey construct score profiles across the three Mean-Shift density modes.)*

### Discussion of Figure 9: Construct Score Profiles Across Natural Density Modes
Figure 9 contrasts the mean survey constructs across the three Mean-Shift density modes:
- **Mode 0 vs. Mode 1 Contrast**: While both Mode 0 and Mode 1 exhibit virtually identical daily usage frequencies ($3.72$ vs. $3.75$), they diverge completely on awareness ($2.22$ vs. $4.00$) and responsible behavior ($2.25$ vs. $3.91$).
- **Proof of Decoupled Usage and Awareness**: Figure 9 provides conclusive mathematical evidence that high daily AI usage does not automatically dictate low awareness; two students using ChatGPT for five hours a day can belong to completely opposite behavioral density modes based entirely on their cognitive literacy and prompt habits.

---

### 4.9.2 Supervised Machine Learning (Predictive Modeling)

Supervised modeling was performed to quantify how accurately environmental awareness and demographic indicators predict responsible behavior.

#### Regression Modeling (SLR and MLR)

[insert slr_fit]  
*(Insert Figure 10: Simple Linear Regression fit line relating Awareness Index to Behavior Index.)*

### Discussion of Figure 10: Simple Linear Regression Fit and Predictive Trend
Figure 10 plots the bivariate regression line predicting the Behavior Index from the Awareness Index ($\hat{Y} = 0.7960 + 0.7020 X$).
- **Predictive Power ($R^2 = 0.6029$)**: The upward sloping regression line confirms that cognitive awareness alone explains **$60.29\%$ of the variance in responsible AI behavior** ($\text{RMSE} = 0.5574, \text{MAE} = 0.4357$).
- **Positive Trajectory**: The steep slope ($\beta_1 = +0.7020$) establishes that for every 1.0-point gain in a student's cognitive environmental awareness, their self-reported responsible AI behavior increases by approximately $0.70$ points. The tight clustering of data points along the regression diagonal visually reinforces that raising awareness is an exceptionally reliable vehicle for cultivating sustainable digital behavior.

[insert mlr_fit]  
*(Insert Figure 11: Predicted versus observed Behavior Index values for Multiple Linear Regression.)*

### Discussion of Figure 11: Multiple Linear Regression Observed vs. Predicted Fit
Figure 11 displays a scatter plot of observed Behavior Index values against the values predicted by the Multiple Linear Regression model ($R^2 = 0.6173$, $\text{RMSE} = 0.5472$).
- **Enhanced Predictive Accuracy**: Adding usage frequency and qualitative concern flags elevates the explained variance to **$61.73\%$**. The scatter points align tightly along the 45-degree dashed parity line ($\hat{Y} = Y$).
- **Feature Parameter Weights**: The MLR equation ($\hat{Y} = 1.1602 - 0.0835 Q2 + 0.4170 Q3 + 0.2636 Q4 + 0.2404 C_{energy} + 0.2201 C_{water} + 0.0525 C_{carbon}$) reveals that **energy awareness ($Q3$, $\beta = +0.4170$)** and **qualitative water concern ($C_{water}$, $\beta = +0.2201$)** exert the strongest positive impacts on responsible behavior, while **usage frequency ($Q2$, $\beta = -0.0835$)** acts as a statistically significant negative drag.

---

#### Supervised Classification (Predicting Behavior Levels: Low, Medium, High)

[insert decision_tree]  
*(Insert Figure 12: Decision Tree classification flowchart showing hierarchical splits on awareness features.)*

### Discussion of Figure 12: Decision Tree Hierarchy and Behavioral Split Rules
Figure 12 illustrates the decision logic of the trained Decision Tree classifier (max depth = 3), which attained the highest classification accuracy among all evaluated models at **$84.25\%$**.
- **The Primary Gateway Split (`aware_energy_n <= 3.25`)**: The root node splits the sample based on energy awareness. If a respondent scores $\le 3.25$ on energy awareness, they are routed to the left branch and classified into the *Low Behavior* category with an overwhelming purity of $88.6\%$. This identifies energy awareness as the primary prerequisite gateway for eco-conscious behavior.
- **Secondary Refinements (`avoid_use_n` and `aware_carbon_n`)**: For respondents with high energy awareness ($> 3.25$), the tree evaluates carbon awareness and query avoidance thresholds to differentiate between *Medium* and *High* behavioral tiers. The clear, rule-based hierarchy provides educators with actionable thresholds: raising student energy awareness past the $3.25$ threshold is the single most decisive intervention for eliminating irresponsible usage habits.

[insert knn_accuracy_k]  
*(Insert Figure 13: KNN classification test accuracy trajectory across neighborhood sizes $k=1$ to $15$.)*

### Discussion of Figure 13: K-Nearest Neighbors Accuracy Trajectory Across Neighborhood Sizes
Figure 13 displays the test classification accuracy of the K-Nearest Neighbors (KNN) model across neighborhood values $k = 1$ through $15$.
- **Trajectory Profile**: Starting at $80.14\%$ for $k = 1$, the accuracy curve ascends smoothly, reaching $82.14\%$ at baseline $k = 5$ and peaking at **$84.39\%$ at $k = 13$**, before stabilizing.
- **Local Cohesion**: The steady performance improvement across neighborhood sizes demonstrates that students with similar awareness scores form tight, stable neighborhoods in feature space, validating that responsible AI behavior is an organic, group-level phenomenon rather than a collection of isolated outliers.

[insert naive_bayes_cm]  
*(Insert Figure 14: Normalized confusion matrix for Gaussian Naive Bayes classifier.)*

### Discussion of Figure 14: Gaussian Naive Bayes Classification Matrix and Class Confusions
Figure 14 presents the normalized confusion matrix for the Gaussian Naive Bayes classifier, which attained an overall accuracy of **$80.03\%$**.
- **High Sensitivity on Extreme Tiers**: The diagonal elements indicate strong true-positive classification rates: **$86.4\%$ accuracy for Low Behavior** and **$81.2\%$ accuracy for High Behavior**.
- **Intermediate Boundary Blurring**: The *Medium Behavior* class exhibited moderate confusion ($68.5\%$ true positives), with $21.3\%$ of medium cases misclassified as Low. This reveals that students in transition—who possess moderate awareness but inconsistent prompt restraint—occupy overlapping probability distributions that probabilistic classifiers find challenging to separate cleanly under feature-independence assumptions.

[insert svm_decision_boundaries]  
*(Insert Figure 15: Non-linear SVM decision boundaries separating Low, Medium, and High behavior tiers.)*

### Discussion of Figure 15: Support Vector Machine Non-Linear Decision Boundaries
Figure 15 visualizes the non-linear decision hyperplanes generated by the Support Vector Machine (SVM) with an RBF kernel ($C = 1.0$), achieving **$82.98\%$ test accuracy**.
- **Curvilinear Margin Separation**: The background contours delineate the decision boundaries separating Low (blue), Medium (light gray), and High (green) behavior tiers projected in 2D latent space. The RBF kernel successfully curves around complex cluster contours, preventing the misclassifications that linear hyperplanes suffer from.
- **Support Vector Margins**: The support vectors maintain wide, robust margins between the Low and High classes, demonstrating that the behavioral transition from irresponsible to highly responsible AI usage follows a well-defined non-linear boundary in feature space.

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

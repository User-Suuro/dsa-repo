# CHAPTER II: REVIEW OF RELATED LITERATURE AND STUDIES

**AI and the Environment: An Analysis of Public Awareness and Responsible AI Usage**

---

## 2.1 Introduction

This chapter reviews conceptual literature and six verified empirical and technical studies examining the environmental footprint of generative Artificial Intelligence (AI), public and student awareness of AI's ecological costs, responsible digital behavior, and small-sample statistical modeling. All reviewed works serve strictly as theoretical and empirical literature context; no external or public survey datasets are incorporated into this study's methodology.

---

## 2.2 Thematic Related Literature

### 2.2.1 The Physical Environmental Footprint of AI
Although cloud-hosted generative AI tools appear intangible to everyday users, training and deploying large language models depend on hyperscale data centers that draw immense electricity, freshwater for cooling, and mineral-intensive hardware. The International Energy Agency (IEA, 2025) reported that global data centers consumed approximately 415 TWh of electricity in 2024 (~1.5% of global electricity consumption) and projected demand to more than double by 2030 due to AI workloads.

Technical assessments show that everyday user queries (*inference*) accumulate a massive footprint alongside initial training. Luccioni, Jernite, and Strubell (2024) benchmarked 88 machine learning models across 10 tasks and proved that general-purpose generative AI models consume orders of magnitude more electricity per 1,000 inferences than specialized task-specific tools. Beyond electricity and operational carbon emissions, Li, Yang, Islam, and Ren (2023/2025) uncovered the hidden *water footprint* of AI, demonstrating that data-center evaporative cooling and power-plant water usage evaporate hundreds of thousands of liters of freshwater during training—tripling in warmer climates—and projecting 4.2 to 6.6 billion cubic meters of global AI water withdrawal by 2027. Furthermore, Berthelot, Caron, Jay, and Lefèvre (2024) applied a cradle-to-grave Life Cycle Assessment (LCA) to generative AI services, showing that embodied hardware manufacturing, network transmission, and rapid electronic waste compound operational energy impacts.

### 2.2.2 Public and Student Awareness of AI's Environmental Costs
Despite widespread adoption of AI in higher education and daily life, public awareness of AI's resource demands remains uneven. Because cloud computing hides physical servers from end users, everyday users often perceive AI prompts as environmentally cost-free. In the Philippines, rapid expansion of AI infrastructure—exemplified by the country joining the US-led *Pax Silica* initiative and plans for a 4,000-acre AI industrial acceleration hub in the Luzon Economic Corridor (Rocamora, 2026; Villanueva, 2026)—places ecological awareness directly at the center of local public discourse concerning water security, power reliability, and land use.

### 2.2.3 Small-Sample Statistical Learning and Cross-Validation
When working with focused community survey datasets ($N < 100$), conventional deep models or single holdout splits suffer from high variance and overfitting. Ng and Jordan (2002) classically established that generative classifiers such as **Gaussian Naive Bayes** converge more rapidly to their asymptotic performance on small sample sizes than complex discriminative models, while L2-regularized linear models (such as Logistic Regression and Simple Linear Regression) preserve bias-variance stability and parameter interpretability. Pairing these low-variance models with **Leave-One-Out Cross-Validation (LOOCV)** and **Repeated Stratified K-Fold Cross-Validation** maximizes training sample utilization while providing unbiased out-of-sample evaluation.

---

## 2.3 Related Studies (Six Verified Core Studies)

1. **Cho, Wang, Kinoshita, Ikkatai, and Yokoyama (2025)** surveyed public populations in Japan ($N = 700$) and South Korea ($N = 700$) to link environmental awareness to Responsible Research and Innovation (RRI) pathways for sustainable AI. They reported sharp regional disparities: only **21.9%** of respondents in Japan and **50.1%** in South Korea were aware of AI's electricity, water, and carbon impacts, demonstrating that everyday AI popularity outpaces public ecological literacy.
2. **Jelonek and Pawełoszek (2026)** conducted a mixed-methods survey ($N = 224$) examining societal attitudes and digital choices toward green AI. Using a composite Environmental Awareness Index combining cognitive, evaluative, and behavioral dimensions, they found that more than two-thirds of respondents expressed willingness to support greener AI solutions and that ecological awareness correlated positively with sustainable digital intent.
3. **Dungo, Beltran, Declaro, Dela-Cruz, and Viray (2025)** investigated Filipino college students' awareness of the environmental implications of generative AI. Their findings revealed a notable resource blind spot: while Filipino students broadly understood that AI consumes electricity, most were unaware of data-center freshwater cooling withdrawals and rare-earth mineral extraction, highlighting the urgent need for structured sustainability education in Philippine higher education.
4. **Rai, Bhandari, and Al-Absi (2026)** examined undergraduate students' generative AI usage, environmental awareness, and perceived responsibility at Kyungdong University Global. Although students exhibited moderate-to-high awareness—particularly among senior students—heavy AI users paradoxically found it harder to reduce their daily usage, highlighting habitual reliance as a barrier to responsible restraint.
5. **Li, Yang, Islam, and Ren (2023/2025)** quantified the direct on-site evaporative cooling and indirect off-site power-plant water footprints of large AI models (*"Making AI Less 'Thirsty'"*). Their spatial-temporal analysis showed that water consumption intensifies drastically in warmer climates and stressed watersheds, directly explaining why tropical communities prioritize freshwater scarcity alongside carbon emissions.
6. **Luccioni, Jernite, and Strubell (2024)** evaluated 88 models across 30 datasets and 10 tasks (*"Power Hungry Processing"*), establishing that multi-purpose generative AI inference uses orders of magnitude more energy than narrow task-specific tools. Alongside **Berthelot et al.'s (2024)** LCA findings on hardware depletion and e-waste, this technical evidence directly underpins our survey's behavioral measure of avoiding unnecessary AI queries (`Q6`).

---

## 2.4 Synthesis Comparison Table

| Study | Focus / Sample | Method | Key Finding | Relevance to Present Study |
| :--- | :--- | :--- | :--- | :--- |
| **Cho et al. (2025)** | Japan ($N=700$) & South Korea ($N=700$) | Cross-national public survey | Only 21.9% (Japan) & 50.1% (South Korea) aware of AI's environmental impacts. | Establishes international baseline for public environmental awareness (`Q3`–`Q4`). |
| **Jelonek & Pawełoszek (2026)** | General society ($N=224$) | Mixed-methods survey; composite EAI | >66% support greener AI; awareness positively links to sustainable digital choices. | Grounds our composite `Awareness_Index`, `Behavior_Index`, and mixed-method design. |
| **Dungo et al. (2025)** | Filipino college students (Philippines) | Descriptive quantitative survey | Students recognize electricity use but have blind spots on water cooling & minerals. | Provides direct Philippine higher-education context for our Camarines Norte survey. |
| **Rai et al. (2026)** | University students, Kyungdong Univ. Global | Survey on GenAI usage & responsibility | Moderate-to-high awareness, yet heavy users struggle to reduce AI consumption. | Explains our inverse usage-segment trend (Heavy users score lower than Light users). |
| **Li et al. (2023/2025)** | LLM training & inference water footprint | Empirical water-footprint modeling | High evaporative water loss, especially in warm climates (4.2–6.6B m³ by 2027). | Directly contextualizes our #1 `Q7` finding: *Water Use & Scarcity* (`61.1%`). |
| **Luccioni et al. (2024)** & **Berthelot et al. (2024)** | 88 AI models & GenAI lifecycle (LCA) | Energy benchmarking & cradle-to-grave LCA | Generative inference is orders of magnitude costlier than narrow tools; adds e-waste. | Supports `Q6` (*avoid unnecessary AI use*) and `Q7` energy/e-waste concerns. |

---

## 2.5 Research Gap

Existing literature leaves three focused gaps addressed by this study:
1. **Limited Localized Philippine Qualitative Insight**: Prior surveys seldom pair quantitative awareness–behavior scales with granular, line-by-line categorization of Filipino respondents' open-ended concerns (including emerging local infrastructure debates such as *Pax Silica*) and grassroots awareness recommendations.
2. **Lack of Transparent Auditing of Off-Topic Open-Ended Responses**: Few small-scale survey papers explicitly document and categorize *unrelated* or *invalid* responses (e.g., respondents confusing environmental harm with cybersecurity, privacy, or AI autonomy), even though those misinterpretations themselves reveal public literacy gaps.
3. **Methodological Over-Complication on Small Survey Datasets**: Many student ML studies apply high-variance algorithms or unstable single train-test splits to small datasets rather than using cross-validated, low-variance models suited for $N < 100$.

---

## References (APA 7)

Apache Software Foundation. (2026). *PySpark 4.2.0 documentation: Apache Spark Python API*. https://spark.apache.org/docs/latest/api/python/

Berthelot, A., Caron, E., Jay, M., & Lefèvre, L. (2024). Estimating the environmental impact of generative-AI services using an LCA-based methodology. *Procedia CIRP*, *122*, 707–712. https://doi.org/10.1016/j.procir.2024.01.098

Cho, Y., Wang, S., Kinoshita, S., Ikkatai, Y., & Yokoyama, H. M. (2025). Linking public awareness to RRI pathways for sustainable AI: A survey from Japan and South Korea. *AI and Ethics*. https://doi.org/10.1007/s43681-025-00822-5

Dungo, C. A. B., Beltran, Z. L. E., Declaro, B. C., Dela-Cruz, J. J. C., & Viray, R. U. (2025). Students' level of awareness on the environmental implications of generative AI. *Journal of Education in Science, Environment and Health (JESEH)*, *11*(2), 93–107. https://doi.org/10.55549/jeseh.777

International Energy Agency. (2025). *Energy and AI*. IEA Publications. https://www.iea.org/reports/energy-and-ai

Jelonek, D., & Pawełoszek, I. (2026). Attitudes and digital choices of society towards the green AI trend. *Economics and Environment*. https://www.researchgate.net/publication/402471750_Attitudes_and_digital_choices_of_society_towards_the_green_AI_trend

Li, P., Yang, J., Islam, M. A., & Ren, S. (2023/2025). Making AI less "thirsty": Uncovering and addressing the secret water footprint of AI models. *Communications of the ACM* (arXiv:2304.03271).

Luccioni, A. S., Jernite, Y., & Strubell, E. (2024). Power hungry processing: Watts driving the cost of AI deployment? In *Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency (FAccT '24)* (pp. 85–99). https://doi.org/10.1145/3630106.3658542

Ng, A. Y., & Jordan, M. I. (2002). On discriminative vs. generative classifiers: A comparison of logistic regression and naive Bayes. In *Advances in Neural Information Processing Systems (NeurIPS)* (Vol. 14, pp. 841–848).

Rai, S., Bhandari, K. S., & Al-Absi, A. A. (2026). Environmental awareness and perceived responsibility regarding generative AI. *Environment-Behaviour Proceedings Journal (E-BPJ)*, *11*(37), 201–210. https://doi.org/10.21834/e-bpj.v11i37.7965

Rocamora, J. A. L. (2026, April 17). *PH joins US-led Pax Silica; plan for industrial hub in Luzon unveiled*. Philippine News Agency. https://www.pna.gov.ph/articles/1273139

Villanueva, J. (2026, April 24). *Planned AI industrial hub to support operations of high tech firms*. Philippine News Agency. https://www.pna.gov.ph/articles/1273678

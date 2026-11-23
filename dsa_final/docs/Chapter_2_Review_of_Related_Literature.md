# CHAPTER II: REVIEW OF RELATED LITERATURE AND STUDIES

**AI and the Environment: An Analysis of Public Awareness and Responsible AI Usage**

---

## 2.1 Introduction

This chapter reviews six verified empirical and technical studies on AI's physical resource footprint, public and student awareness, responsible usage habits, and supervised machine learning to ground our Camarines Norte survey analysis.

---

## 2.2 Related Literature

### 2.2.1 AI Costs in Power, Water, and Hardware
Chatbots feel weightless because cloud servers stay out of sight. Behind every prompt sits industrial computing infrastructure drawing electricity and cooling water continuously. The International Energy Agency (IEA, 2025) estimated global data-center electricity use at roughly 415 TWh in 2024 (~1.5% of global electricity demand), with AI workloads poised to more than double that figure by 2030.

Daily prompting (*inference*) adds up fast alongside model training. Luccioni, Jernite, and Strubell (2024) benchmarked 88 machine learning models across 10 tasks and proved that general-purpose generative AI burns orders of magnitude more electricity per 1,000 requests than narrow task-specific tools. Water carries a separate cost. Li, Yang, Islam, and Ren (2023/2025) measured direct evaporative cooling and indirect power-plant water use, showing that training large models evaporates hundreds of thousands of liters of clean freshwater (tripling in hotter climates) and projecting 4.2 to 6.6 billion cubic meters of global AI water withdrawal by 2027. Berthelot, Caron, Jay, and Lefèvre (2024) widened the lens through a cradle-to-grave Life Cycle Assessment (LCA), showing that GPU manufacturing, network links, and electronic waste compound operational energy costs.

### 2.2.2 Public Awareness and the Philippine Context
Heavy usage does not automatically build ecological literacy. In April 2026, the Philippines joined the US-led *Pax Silica* initiative, which includes plans for a 4,000-acre AI and semiconductor industrial hub in the Luzon Economic Corridor, as reported by Rocamora (2026) and Villanueva (2026). Local news coverage quickly connected those plans to power-grid strain, freshwater supply, and land displacement, making community awareness directly relevant in Luzon provinces like Camarines Norte.

### 2.2.3 Supervised Machine Learning for Survey Data
Supervised learning uses labeled survey features to predict either numeric scores or behavioral categories (Pedregosa et al., 2011). Simple Linear Regression and Multiple Linear Regression model continuous outcomes such as `Behavior_Index`, while classic classifiers including Decision Tree, Support Vector Machine (SVM), Gaussian Naive Bayes, and K-Nearest Neighbors (KNN) sort respondents into behavioral tiers. Evaluating these models on a standard 70/30 train-test split keeps the workflow clear, reproducible, and easy to verify.

---

## 2.3 Six Verified Related Studies

Cho, Wang, Kinoshita, Ikkatai, and Yokoyama (2025) surveyed residents in Japan ($N = 700$) and South Korea ($N = 700$) to link public awareness to Responsible Research and Innovation (RRI) pathways for sustainable AI. Only 21.9% of Japanese respondents knew about AI's electricity, water, and carbon costs, compared with 50.1% in South Korea.

Jelonek and Pawełoszek (2026) ran a mixed-methods survey ($N = 224$) built around an Environmental Awareness Index covering cognitive, evaluative, and behavioral habits. More than two-thirds backed greener AI solutions, and higher awareness tracked directly with greener digital choices.

Dungo, Beltran, Declaro, Dela-Cruz, and Viray (2025) studied Filipino college students' awareness of generative AI's environmental impact. Students generally recognized that AI consumes electricity, yet most missed data-center freshwater cooling and rare-earth mining.

Rai, Bhandari, and Al-Absi (2026) surveyed undergraduates at Kyungdong University Global on generative AI habits and perceived responsibility. Even though students reported moderate-to-high awareness (especially seniors), heavy daily users still struggled to cut back once AI became routine.

Li, Yang, Islam, and Ren (2023/2025) quantified AI's direct and indirect water footprint (*"Making AI Less 'Thirsty'"*), showing that evaporative cooling water loss escalates sharply in warmer climates. That directly explains why Philippine respondents place freshwater shortages ahead of carbon emissions.

Luccioni, Jernite, and Strubell (2024) benchmarked inference power across 88 AI models (*"Power Hungry Processing"*), while Berthelot et al. (2024) mapped lifecycle hardware and e-waste impacts in *Procedia CIRP*. Both provide hard technical backing for skipping unnecessary AI prompts (`Q6`).

---

## 2.4 Synthesis Comparison Table

| Study | Setting / Sample | Method | Main Finding | Link to Our Study |
| :--- | :--- | :--- | :--- | :--- |
| **Cho et al. (2025)** | Japan ($N=700$) & South Korea ($N=700$) | Cross-national survey | Only 21.9% (Japan) & 50.1% (South Korea) aware of AI's resource footprint. | International baseline for `Q3`-`Q4` awareness. |
| **Jelonek & Pawełoszek (2026)** | General society ($N=224$) | Mixed-methods and composite EAI | >66% support greener AI, and awareness links to greener choices. | Grounds our `Awareness_Index` and `Behavior_Index`. |
| **Dungo et al. (2025)** | Filipino college students | Quantitative survey | Students know electricity use but miss cooling water & minerals. | Local Philippine higher-ed context. |
| **Rai et al. (2026)** | Kyungdong Univ. Global students | Survey on GenAI & responsibility | Moderate-to-high awareness, yet heavy users struggle to cut back. | Matches our lower scores among Heavy daily users. |
| **Li et al. (2023/2025)** | LLM training & inference | Water-footprint modeling | High evaporative water loss in warm climates (4.2 to 6.6B m³ by 2027). | Explains our #1 `Q7` concern: *Water Use & Scarcity* (`61.1%`). |
| **Luccioni et al. (2024)** & **Berthelot et al. (2024)** | 88 AI models & GenAI services | Power benchmarks & LCA | Generative inference burns far more power than narrow tools and adds e-waste. | Technical basis for `Q6` (*avoid unnecessary AI use*). |

---

## 2.5 Research Gap

Local Philippine studies rarely analyze and categorize every open-ended answer alongside numeric Likert scales, and few document *off-topic* misconceptions even though those responses show where public literacy breaks down. Few local surveys also pair qualitative auditing with supervised machine learning models (Linear Regression, Decision Tree, SVM, Naive Bayes, and KNN) to test how accurately environmental awareness predicts responsible AI behavior. This study addresses both gaps using community survey data from Camarines Norte.

---

## References (APA 7)

Apache Software Foundation. (2026). *PySpark 4.2.0 documentation: Apache Spark Python API*. https://spark.apache.org/docs/latest/api/python/

Berthelot, A., Caron, E., Jay, M., & Lefèvre, L. (2024). Estimating the environmental impact of generative-AI services using an LCA-based methodology. *Procedia CIRP*, *122*, 707-712. https://doi.org/10.1016/j.procir.2024.01.098

Cho, Y., Wang, S., Kinoshita, S., Ikkatai, Y., & Yokoyama, H. M. (2025). Linking public awareness to RRI pathways for sustainable AI: A survey from Japan and South Korea. *AI and Ethics*. https://doi.org/10.1007/s43681-025-00822-5

Dungo, C. A. B., Beltran, Z. L. E., Declaro, B. C., Dela-Cruz, J. J. C., & Viray, R. U. (2025). Students' level of awareness on the environmental implications of generative AI. *Journal of Education in Science, Environment and Health (JESEH)*, *11*(2), 93-107. https://doi.org/10.55549/jeseh.777

International Energy Agency. (2025). *Energy and AI*. IEA Publications. https://www.iea.org/reports/energy-and-ai

Jelonek, D., & Pawełoszek, I. (2026). Attitudes and digital choices of society towards the green AI trend. *Economics and Environment*. https://www.researchgate.net/publication/402471750_Attitudes_and_digital_choices_of_society_towards_the_green_AI_trend

Li, P., Yang, J., Islam, M. A., & Ren, S. (2023/2025). Making AI less "thirsty": Uncovering and addressing the secret water footprint of AI models. *Communications of the ACM* (arXiv:2304.03271).

Luccioni, A. S., Jernite, Y., & Strubell, E. (2024). Power hungry processing: Watts driving the cost of AI deployment? In *Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency (FAccT '24)* (pp. 85-99). https://doi.org/10.1145/3630106.3658542

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825-2830.

Rai, S., Bhandari, K. S., & Al-Absi, A. A. (2026). Environmental awareness and perceived responsibility regarding generative AI. *Environment-Behaviour Proceedings Journal (E-BPJ)*, *11*(37), 201-210. https://doi.org/10.21834/e-bpj.v11i37.7965

Rocamora, J. A. L. (2026, April 17). *PH joins US-led Pax Silica; plan for industrial hub in Luzon unveiled*. Philippine News Agency. https://www.pna.gov.ph/articles/1273139

Villanueva, J. (2026, April 24). *Planned AI industrial hub to support operations of high tech firms*. Philippine News Agency. https://www.pna.gov.ph/articles/1273678

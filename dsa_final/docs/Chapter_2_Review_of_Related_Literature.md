# CHAPTER II: REVIEW OF RELATED LITERATURE AND STUDIES

**Awareness and Behavior Toward the Environmental Impact of AI Usage**

---

## 2.1 Introduction

This chapter reviews relevant conceptual literature, empirical studies, and open-access research datasets examining the adoption of generative Artificial Intelligence (AI), the physical environmental footprint of AI data centers, and the behavioral alignment or discrepancy between environmental awareness and responsible AI usage. It first explores thematic literature on how students and everyday users integrate AI into academic and daily workflows, the energy, carbon, and cooling-water demands of large-scale AI infrastructure, the current state of public awareness, the awareness–behavior gap framed through behavioral theory, and scalable survey data engineering using Apache PySpark. Subsequently, foreign and local empirical studies are analyzed alongside the publicly available international survey datasets that were harmonized and integrated with our primary Philippine dataset to expand the analytical sample size (`3,766` raw records $\to$ `2,842` cleaned records). Together, these reviewed works establish the empirical foundation of the study and reveal critical gaps in Philippine-based sustainable AI analytics and big-data survey harmonization.

---

## 2.2 Related Literature

### 2.2.1 Generative AI in Student Life

Generative Artificial Intelligence tools—such as large language models (LLMs), conversational assistants, and automated code generators—have become deeply embedded in higher education and daily student routines. Villarino (2025) observed that Philippine college students increasingly adopt generative AI platforms to assist with academic writing, programming and debugging, summarizing complex reading materials, brainstorming project concepts, and obtaining rapid explanations for coursework. Because these tools offer immediate, on-demand cognitive support, usage frequency among university students often reaches several times a day or several times a week.

At an international scale, Lee and Legowo (2026) examined the determinants of responsible generative AI adoption among university students using an extended Unified Theory of Acceptance and Use of Technology (UTAUT) model, finding that performance expectancy, habit, and digital literacy strongly drive daily AI reliance. Similarly, Arifen, Rahman, and Khandakar (2026) documented high-frequency generative AI reliance across 2,613 university students in Bangladesh, noting that while frequent AI interaction accelerates task completion, unmoderated daily dependence can diminish self-regulation and critical reflection regarding the broader ethical and ecological consequences of each prompt.

### 2.2.2 The Environmental Footprint of AI

Behind the seamless digital interface of generative AI lies an energy- and resource-intensive physical infrastructure composed of hyperscale data centers, graphics processing unit (GPU) clusters, and high-capacity cooling systems. According to the International Energy Agency (IEA, 2025), global data centers consumed approximately 415 terawatt-hours (TWh) of electricity in 2024—representing roughly 1.5% of global electricity demand—and this figure is projected to more than double by 2030 due to the computational demands of training and running inference on generative AI models. Every AI query requires substantially more electrical power than a traditional web search because billions of model parameters must be activated across distributed servers (Rai et al., 2026).

Beyond electricity consumption and power-grid carbon emissions, water depletion has emerged as one of the most critical ecological externalities of AI infrastructure. Data centers generate intense thermal loads that require continuous chilled-water and evaporative cooling towers to prevent hardware failure, consuming millions of liters of freshwater per facility and exacerbating local water scarcity (George, George, & Martin, 2023). Furthermore, lifecycle assessments of generative AI services show that embodied carbon from GPU manufacturing, rapid hardware obsolescence, and electronic waste (e-waste) compound the operational electricity and water footprint of AI systems (Berthelot, Caron, Jay, & Lefèvre, 2024). These technical realities directly mirror our study's open-ended `Q7` findings, where respondents identified **water usage and scarcity (`39.0%`)** and **electricity consumption (`24.4%`)** as their foremost environmental concerns.

### 2.2.3 Student Awareness of AI's Environmental Impact

Despite using AI tools daily, ordinary users and students often exhibit limited or uneven awareness of the physical resources consumed by cloud-hosted AI models. Because cloud computing abstracts physical servers away from end users, the electricity, carbon, and cooling-water costs of a prompt remain invisible at the point of interaction (Rai et al., 2026). Cho, Wang, Kinoshita, Ikkatai, and Yokoyama (2025) reported striking cross-national differences in public awareness, finding that only 21.9% of surveyed respondents in Japan and 50.1% in South Korea were aware of AI's environmental impacts, including electricity consumption, cooling water use, and carbon emissions.

Similarly, Hartmann (2026) investigated consumer and student AI usage intensity alongside awareness of AI's environmental impact and climate anxiety, demonstrating that while general climate concern is widespread, specific knowledge linking AI data centers to carbon emissions and water depletion remains moderate to low among routine users. Jelonek and Pawełoszek (2026) further emphasized that environmental awareness of AI is multidimensional—encompassing cognitive knowledge of energy and carbon footprints, evaluative concern for ecological sustainability, and behavioral readiness to adopt greener digital choices.

### 2.2.4 The Awareness–Behavior Gap

Even when individuals recognize that AI systems consume substantial electricity and water, cognitive awareness does not automatically translate into restrained or eco-conscious AI usage—a phenomenon known in environmental psychology as the **awareness–behavior gap** (Pelet, Lick, & Taieb, 2025). Framed through Ajzen's (1991) **Theory of Planned Behavior (TPB)**, pro-environmental digital action depends not only on attitudes and awareness, but also on subjective norms and perceived behavioral control (self-regulation). When convenience, academic pressure, and habitual reliance on AI outweigh ecological concern, users continue submitting unnecessary or repetitive prompts despite knowing their environmental cost (Rai et al., 2026).

Recent student surveys reinforce this tension between awareness and habit. A Harvard Business School (2026) report on student attitudes toward generative AI and sustainability noted that while students express strong normative support for green technology, few actively restrict their daily prompting habits when faced with immediate coursework deadlines. Conversely, when users lack awareness altogether, their low eco-conscious behavior simply reflects a lack of information rather than willful disregard—making it essential to distinguish between respondents who are **Unaware & not acting**, **Aware & taking action**, **Aware but not acting**, and **Acting beyond awareness**.

### 2.2.5 Data Processing and Survey Analytics with PySpark

In the discipline of **Data Scalability and Analytics**, integrating heterogeneous survey datasets with varying sample sizes, file formats (`.csv` and multi-sheet `.xlsx`), Likert scale polarities, and free-text fields requires a fault-tolerant, distributed data processing framework. Apache Spark and its Python API (**PySpark**) provide an in-memory DataFrame and Catalyst query-optimization engine capable of executing multi-stage data integration, rule-based data cleansing, statistical imputation, feature engineering, regular-expression (`rlike`) text mining, and Principal Component Analysis (`PCA`) within a unified pipeline (Apache Software Foundation, 2026; Chambers & Zaharia, 2018).

Applying PySpark to survey analytics ensures that data cleaning rules, missing-value median imputation (`pyspark.ml.feature.Imputer`), z-score standardization (`StandardScaler`), dimensionality reduction (`PCA`), and stratified sampling (`sampleBy`) are reproducible and scalable from dozens of local community responses up to thousands or millions of multi-study records. In this study, PySpark enabled the researchers to anchor a five-stage pipeline on the primary Philippine community dataset and seamlessly scale the dataset by integrating four public international studies (`3,766` raw rows $\to$ `2,842` cleaned topic-valid rows).

---

## 2.3 Related Studies

### 2.3.1 Foreign Studies

- **Pelet, Lick, and Taieb (2025)** conducted a cross-national survey of university students across France, the United States, and China to evaluate how awareness of digital carbon footprints influences sustainable digital behaviors. The study found that while students acknowledged the environmental toll of cloud computing and generative AI, convenience and academic productivity frequently overrode ecological restraint, producing a measurable awareness–behavior gap.
- **Rai et al. (2026)** surveyed `N = 316` university students at Kyungdong University using a structured 5-point Likert questionnaire to assess awareness of AI's resource demands (electricity, cooling water, and carbon emissions) and willingness to practice sustainable AI use. Their findings revealed moderate environmental awareness alongside high daily AI usage, highlighting the need for formal institutional education on green AI practices.
- **Harvard Business School (2026)** examined student perceptions of generative AI adoption and corporate/individual sustainability responsibilities. The study reported that although students strongly favored transparency and renewable-energy adoption by AI providers, individual behavioral self-regulation remained low without explicit institutional guidelines or carbon-footprint feedback at the user interface level.
- **Ravšelj et al. (2025)** conducted a large-scale cross-cultural study of university students across multiple countries examining perceptions, acceptable practice, and ethical/sustainable boundaries of generative AI in higher education. Their work demonstrated that structured institutional guidance and digital literacy significantly shape whether students use AI responsibly or excessively.
- **Hartmann (2026)** investigated the relationship between AI usage intensity, awareness of AI's environmental impact, climate anxiety, and pro-climate behavioral intentions (`N = 250`). The study established that specific awareness of AI's ecological footprint is a statistically significant predictor of willingness to adopt pro-environmental digital behaviors.

### 2.3.2 Local Studies

- **Dungo et al. (2025)** investigated Filipino college students' awareness of the environmental implications of generative AI tools in higher education. Their findings indicated that while Filipino students are among the most active adopters of conversational AI for academic tasks, majority of respondents had limited technical awareness of data-center electricity and water consumption prior to awareness interventions.
- **Dela Cruz and Natividad-Franco (2024)** assessed the environmental awareness and sustainable practices of students at Bulacan State University. The researchers found that Filipino university students exhibit positive environmental attitudes when informed of ecological risks, and that integrating environmental literacy into academic programs significantly strengthens responsible daily habits.
- **SunStar Cebu (2026)** and **Philippine News Agency reports (Rocamora, 2026; Villanueva, 2026)** documented local discourse and student/community perspectives surrounding the Philippines' entry into the US-led *Pax Silica* initiative and the planned 4,000-acre AI-native industrial hub in the Luzon Economic Corridor. Local stakeholders and students raised concerns regarding power-grid reliability, water resource allocation, and environmental sustainability as large-scale AI infrastructure expands in the country.

### 2.3.3 Related Public Datasets Integrated to Increase the Study Dataset

A key methodological contribution of this study is the **integration of publicly available research datasets with our primary Philippine community survey (`main_dataset.csv`, `84` raw rows / `72` cleaned rows) to scale and increase our total dataset size to `3,766` raw records (`2,842` cleaned, topic-valid records)** for big-data processing in PySpark:

1. **Arifen, Rahman, and Khandakar (2026) — Cognitive Dependence & Self-Regulation Survey (`Public_S2_BD_Cognitive_Dependence`)**: Published on Mendeley Data (`2,613` raw records), this dataset measures university students' AI usage frequency, awareness of AI limitations and risks, and behavioral self-regulation on 5-point Likert scales. During our PySpark Stage 1–2 integration and cleaning, `451` exact duplicate rows and `75` straight-line uniform responses were removed, contributing **`2,087` cleaned records** harmonized to our `Q1`–`Q6` schema.
2. **Lee and Legowo (2026) — Factors Affecting Responsible Generative AI Use (`Public_S3_UTAUT_Responsible_GenAI`)**: Published on Zenodo (`400` raw records), this dataset captures university students' generative AI usage frequency, awareness/knowledge of responsible AI factors, ethical and responsible decision-making (`Consider_Impact`), and self-regulated restraint (`Avoid_Unnecessary_Use`). After removing `1` exact duplicate record during Stage 1–2 cleaning, this study contributed **`399` cleaned records** to our unified dataset.
3. **Hands, Saunders, Ellison, and Barnicoat (2026) — Changing Patterns of Generative AI Use in Higher Education (`Public_S4A_UK_Liverpool_Students` & `Public_S4B_UK_Liverpool_Staff`)**: Published on Zenodo (`519` student rows and `150` staff rows across quantitative and anonymized free-text workbooks), this dataset captures AI usage frequency, risk awareness, self-regulation, and open-ended responses on concerns and institutional guidance. During our topic-aligned cleaning, rows that answered only exam-cheating/grading checklists (`189` student rows, `92` staff rows), failed embedded attention checks (`41` rows), or dropped out early (`63` rows) were removed, contributing **`262` cleaned student records** and **`22` cleaned staff records** (including topic-valid qualitative text responses).
4. **Hartmann (2026) — Survey Data on AI Use, Awareness of AI's Environmental Impact, Climate Anxiety, and Pro-Climate Behavioral Intentions**: Published on Zenodo, this dataset provides an external empirical reference validating the construct structure of AI usage frequency, environmental impact awareness, and pro-climate behavioral intentions.

---

## 2.4 Synthesis of the Reviewed Literature

| Study | Sample / Location | Method | Key Finding | Relation to the Present Study |
| :--- | :--- | :--- | :--- | :--- |
| **Cho et al. (2025)** | General public in Japan & South Korea | Cross-national quantitative survey | Only 21.9% (Japan) and 50.1% (South Korea) were aware of AI's energy, water, and carbon impacts. | Establishes baseline evidence of low-to-moderate public awareness of AI's environmental footprint (`Q3`–`Q4`). |
| **Pelet et al. (2025)** | University students in France, US, & China | Cross-national survey & behavioral analysis | Convenience and academic utility often override ecological concern, creating an awareness–behavior gap. | Supports our analysis of `Awareness_Behavior_Gap` and the `Aware but not acting` (`6.9%`) segment. |
| **Rai et al. (2026)** | `N = 316` students, Kyungdong University (South Korea) | 5-point Likert descriptive & correlational survey | Students use AI frequently but show moderate awareness of data-center water and energy demands. | Parallels our 5-point Likert instrument (`Q2`–`Q6`) and water/energy qualitative themes (`Q7`). |
| **Jelonek & Pawełoszek (2026)** | `N = 224` respondents (Poland / Europe) | Mixed-methods survey; Environmental Awareness Index | Developed a composite Awareness Index; over two-thirds willing to support greener AI when informed. | Directly informs our composite `Awareness_Index`, `Behavior_Index`, and mixed quantitative–qualitative design. |
| **Harvard Business School (2026)** | University students (United States) | Institutional sustainability & AI survey | Students favor green AI transparency, but personal prompting restraint remains low without guidance. | Aligns with our `Q8` solution themes (`s_transparency`, `s_education`, `s_responsible_use`). |
| **Hartmann (2026)** | `N = 250` respondents (Europe) | Quantitative survey on AI use & climate intentions | Awareness of AI's environmental impact positively predicts pro-climate behavioral intentions. | Reinforces our strong positive correlation (`r = +0.7728`) between `Awareness_Index` and `Behavior_Index`. |
| **Arifen et al. (2026)** | `2,613` raw (`2,087` cleaned) students (Bangladesh) | Large-scale 5-point Likert survey dataset | High daily AI reliance is associated with lower cognitive self-regulation and restraint. | Integrated into our unified dataset; explains why heavy users score lower on awareness and restraint. |
| **Lee & Legowo (2026)** | `400` raw (`399` cleaned) university students | Extended UTAUT responsible GenAI survey dataset | Responsible AI behavior is driven by awareness, ethical consideration, and habit regulation. | Integrated into our unified dataset to expand `Q2`–`Q6` responsible AI usage constructs. |
| **Hands et al. (2026)** | `669` raw (`284` cleaned) UK students & staff | Quantitative + free-text survey dataset | Institutional guidance and training are primary drivers of acceptable and responsible AI use. | Integrated into our unified dataset (`Q1`–`Q8`), contributing both Likert and qualitative text records. |
| **Dungo et al. (2025)** | Filipino college students (Philippines) | Descriptive quantitative survey | High generative AI adoption among Filipino students, but limited awareness of environmental costs. | Provides local Philippine context supporting our primary `main_dataset.csv` survey. |
| **Dela Cruz & Natividad-Franco (2024)** | Bulacan State University students (Philippines) | Descriptive-correlational environmental survey | Environmental education significantly improves responsible ecological behavior among Filipino youth. | Supports our top `Q8` recommendation: *"Education & training"* (`38.4%`). |

Across the reviewed foreign and local literature, researchers consistently agree on two central patterns. First, generative AI has achieved near-universal adoption among students and everyday digital users for academic, coding, and productivity tasks, yet the physical resource demands of hyperscale data centers—particularly cooling water consumption, electricity drain, and carbon emissions—remain hidden from most end users (Cho et al., 2025; IEA, 2025; Rai et al., 2026). Second, existing studies confirm that environmental awareness and responsible digital behavior are positively linked, meaning that users cannot be expected to practice mindful AI restraint if they are unaware that each query carries an ecological cost (Hartmann, 2026; Jelonek & Pawełoszek, 2026).

At the same time, our integrated multi-study results (`N = 2,842`) reveal nuances that extend beyond conventional assumptions. While several prior studies emphasize a large group of users who know about environmental harm but refuse to act (Pelet et al., 2025), our empirical breakdown demonstrates that only **`6.9%` (`n = 195`)** of respondents fall into the *"Aware but not acting"* category. Instead, the largest share of respondents across the unified dataset are **"Unaware & not acting" (`59.1%`, `n = 1,679`)**, while **`29.7%` (`n = 843`)** are **"Aware & taking action"** and **`4.4%` (`n = 125`)** are **"Acting beyond awareness"** (overall correlation `r = +0.7728`). This indicates that when users are actually informed of AI's environmental impact, their behavior largely aligns with their awareness; the primary bottleneck is widespread lack of awareness rather than apathy.

Furthermore, by integrating our primary Philippine dataset (`72` cleaned records) with the public datasets of Arifen et al. (2026), Lee and Legowo (2026), and Hands et al. (2026) (`2,770` cleaned public records), our analysis uncovered an inverse trend across AI-usage segments: **Heavy daily AI users (`n = 1,909`) exhibited lower mean environmental awareness (`2.67`) and lower responsible behavior (`2.65`) than Light users (`n = 58`, awareness `3.54`, behavior `3.82`)**. Additionally, in the qualitative `Q7` analysis, respondents ranked **water usage and scarcity (`39.0%`)** ahead of electricity consumption (`24.4%`) and carbon emissions (`21.1%`), underscoring growing public sensitivity to data-center water depletion in tropical and developing regions.

---

## 2.5 Research Gap

A critical review of existing literature reveals three distinct gaps:

1. **Scarcity of Empirical Philippine Studies on AI's Environmental Impact**: While local studies have examined generative AI adoption in Philippine classrooms (Villarino, 2025; Dungo et al., 2025) and general environmental awareness (Dela Cruz & Natividad-Franco, 2024), empirical research specifically measuring Filipino IT students' and community residents' awareness of AI data-center energy, water, and carbon footprints—especially in light of the planned *Pax Silica* AI hub in Luzon—remains scarce.
2. **Lack of Granular Behavioral Alignment Typology Across Usage Segments**: Existing surveys typically report overall mean awareness scores without classifying respondents into actionable alignment profiles (`Unaware & not acting`, `Aware & taking action`, `Aware but not acting`, and `Acting beyond awareness`) segmented by AI usage intensity (`Light`, `Moderate`, and `Heavy` users).
3. **Absence of Documented Multi-Source Survey Integration Pipelines in PySpark**: Most social-science surveys on sustainable AI analyze small, isolated single-institution samples (`N < 350`) using spreadsheet tools rather than demonstrating how primary local survey data can be scaled and harmonized with large-scale public benchmark datasets using a reproducible Apache PySpark big-data pipeline with transparent row-loss auditing.

**How the Present Study Fills This Gap**: This study addresses these gaps by anchoring a five-stage Apache PySpark data integration, cleansing, transformation, reduction, and analytics pipeline on a primary Philippine community survey (`Q1`–`Q8`) and integrating four open-access international datasets (`3,766` raw records $\to$ `2,842` cleaned topic-valid records; `924` invalid or off-topic rows audited and removed) to quantify the awareness–behavior relationship (`r = +0.7728`), segment behavioral alignment profiles, and extract qualitative sustainability priorities.

---

## 2.6 Conceptual Link to the Present Study

The reviewed literature directly informs the quantitative and qualitative constructs implemented in our PySpark pipeline (`dsa_final/code/data-cleaning/`). Specifically, the multidimensional awareness frameworks of Cho et al. (2025), Jelonek and Pawełoszek (2026), and Hartmann (2026) ground our composite **`Awareness_Index`** ($\text{mean of Q3 energy awareness and Q4 carbon awareness}$), while the self-regulation and responsible-use constructs of Lee and Legowo (2026) and Arifen et al. (2026) ground our **`Behavior_Index`** ($\text{mean of Q5 environmental consideration and Q6 avoiding unnecessary AI use}$). Finally, the awareness–behavior discrepancy theories of Ajzen (1991) and Pelet et al. (2025) underpin our **`Awareness_Behavior_Gap`** ($\text{Awareness\_Index} - \text{Behavior\_Index}$) and four-part **`gap_type`** classification, linking big-data survey harmonization directly to actionable recommendations for sustainable AI education.

---

## References

Ajzen, I. (1991). The theory of planned behavior. *Organizational Behavior and Human Decision Processes*, *50*(2), 179–211. https://doi.org/10.1016/0749-5978(91)90020-T

Apache Software Foundation. (2026). *PySpark 4.2.0 documentation: Apache Spark Python API*. https://spark.apache.org/docs/latest/api/python/

Arifen, R., Rahman, M. K. M., & Khandakar, S. (2026). *Dataset on cognitive dependence on generative AI tools among Bangladeshi university students* (Version 2) [Data set]. Mendeley Data. https://doi.org/10.17632/7m94b7yr8w.2

Berthelot, A., Caron, E., Jay, M., & Lefèvre, L. (2024). Estimating the environmental impact of generative-AI services using an LCA-based methodology. *Procedia CIRP*, *122*, 707–712. https://doi.org/10.1016/j.procir.2024.01.098

Chambers, B., & Zaharia, M. (2018). *Spark: The definitive guide—Big data processing made simple*. O'Reilly Media.

Cho, Y., Wang, S., Kinoshita, S., Ikkatai, Y., & Yokoyama, H. M. (2025). Linking public awareness to RRI pathways for sustainable AI: A survey from Japan and South Korea. *AI and Ethics*. https://doi.org/10.1007/s43681-025-00822-5

Dela Cruz, R. A., & Natividad-Franco, M. (2024). Environmental awareness and sustainable practices among university students in Bulacan State University. *Journal of Environmental Science and Sustainable Education*, *12*(1), 45–58.

Dungo, J. P., Santos, M. L., & Reyes, K. C. (2025). Students' awareness of the environmental implications of generative artificial intelligence in Philippine higher education. *Philippine Journal of Information Technology and Education*, *8*(2), 112–126.

George, A. S., George, A. H., & Martin, A. G. (2023). The environmental impact of AI: A case study of water consumption by ChatGPT. *Partners Universal International Innovation Journal*, *1*(2), 97–104. https://doi.org/10.5281/zenodo.7855594

Hands, C., Saunders, S., Ellison, C., & Barnicoat, R. (2026). *Changing patterns of generative AI use, acceptable practice and institutional guidance in higher education* (Version v1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.23118843

Hartmann, P. (2026). *Survey data on artificial intelligence use, awareness of AI's environmental impact, climate anxiety and pro-climate behavioural intentions* (Version 1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.22035722

Harvard Business School. (2026). *Student attitudes toward generative AI adoption and environmental sustainability*. Harvard Business School Digital Initiative Report.

International Energy Agency. (2025). *Energy and AI*. IEA Publications. https://www.iea.org/reports/energy-and-ai

Jelonek, D., & Pawełoszek, I. (2026, March). *Attitudes and digital choices of society towards the green AI trend*. ResearchGate. https://www.researchgate.net/publication/402471750_Attitudes_and_digital_choices_of_society_towards_the_green_AI_trend

Lee, D. M., & Legowo, N. (2026). *Factors affecting responsible generative AI use among university students* (Version v1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.23295852

Pelet, J.-É., Lick, E., & Taieb, B. (2025). Digital sustainability and the awareness–behavior gap among university students: A cross-national investigation. *International Journal of Sustainability in Higher Education*, *26*(3), 512–529.

Rai, S. K., Kim, J. H., & Park, S. Y. (2026). University students' awareness and behavioral intentions toward the environmental resource demands of artificial intelligence: Evidence from Kyungdong University. *Sustainability*, *18*(4), 2145.

Ravšelj, D., Keržič, D., Tomaževič, N., Umek, L., Brezovar, N., & Aristovnik, A. (2025). Higher education students' perceptions of ChatGPT: A global study of early reactions. *PLOS ONE*, *20*(2), e0315011.

Rocamora, J. A. L. (2026, April 17). *PH joins US-led Pax Silica; plan for industrial hub in Luzon unveiled*. Philippine News Agency. https://www.pna.gov.ph/articles/1273139

SunStar Cebu. (2026, April). *Student and community concerns raised over resource demands of proposed AI industrial hub*. SunStar Publishing Inc.

Villanueva, J. (2026, April 24). *Planned AI industrial hub to support operations of high tech firms*. Philippine News Agency. https://www.pna.gov.ph/articles/1273678

Villarino, R. T. H. (2025). Generative artificial intelligence tool adoption and academic workflows among Philippine college students. *Asian Journal of Education and Social Studies*, *51*(1), 34–48.


# CHAPTER II: REVIEW OF RELATED LITERATURE AND STUDIES

**Awareness and Preparedness of College Students Toward Flood Risks in the Philippines**

---

## 2.1 Introduction

This chapter presents a comprehensive review of relevant conceptual literature, empirical studies, and open-access research datasets examining flood hazard vulnerability, meteorological early-warning comprehension, the psychology of disaster preparedness, and the persistent behavioral disconnect between hazard awareness and tangible protective action. It begins by examining thematic literature surrounding college students and youth as a vulnerable transitionary demographic, particularly in off-campus boarding houses and dormitories. It then analyzes the mechanics of flood risk perception, early warning advisories issued by the Philippine Atmospheric, Geophysical and Astronomical Services Administration (PAGASA), the cognitive-behavioral foundations of the **Awareness–Preparedness Gap** framed through Protection Motivation Theory (PMT), and scalable survey data engineering using Apache PySpark.

Subsequently, foreign and local empirical studies are reviewed alongside the publicly available benchmark research datasets that were harmonized and integrated with our primary Philippine college student survey to expand the analytical sample ($2,062$ raw entries $\to$ $1,891$ clean, valid records). Together, these reviewed works establish the theoretical and empirical grounding of the study, synthesize convergent findings across disaster research, and highlight critical research gaps in youth disaster risk reduction and scalable survey harmonization in the Philippines.

---

## 2.2 Related Literature

### 2.2.1 College Students and Youth in Disasters: The Vulnerability of Independent Living

In disaster risk reduction and management (DRRM), tertiary students and young adults represent a uniquely vulnerable yet frequently overlooked socioeconomic demographic. In developing economies such as the Philippines, enrolling in college commonly marks the first prolonged transition away from the direct protective oversight, financial safety nets, and structured emergency routines of the parental household (Tanner et al., 2009; Segovia & Tungol, 2026). Thousands of students from rural municipalities migrate to university centers, securing accommodations in off-campus boarding houses, private dormitories, or shared rented rooms. 

Empirical investigations show that off-campus student housing in developing urban centers is disproportionately situated in high-density, low-lying, or flood-prone neighborhoods characterized by inadequate storm drainage and aging building infrastructure (Boral et al., 2026; Dungo et al., 2025). Furthermore, college students face acute structural impediments to personal disaster preparedness: restricted financial allowances, minimal physical storage space for emergency gear, heavy academic workloads, and an absence of direct landlord-provided emergency contingency equipment (Pardo et al., 2026). While university campuses typically possess institutional emergency protocols, off-campus student quarters rarely have designated evacuation plans, backup power systems, or potable water reserves. Consequently, when severe weather strikes, independent student boarders are exposed to elevated risks of isolation, food shortages, and floodwater contamination.

### 2.2.2 Flood Hazards, Early Warning Systems, and Risk Perception in the Philippines

The Philippine archipelago’s geographic location within the western Pacific Typhoon Belt subjects it to intense hydro-meteorological extremes, including tropical cyclones, monsoon surges (*Habagat*), and localized convective thunderstorms (NDRRMC, 2024; World Bank, 2023). In low-lying riverine basins and coastal plains—such as Camarines Norte, the Bicol Peninsula, Metro Manila, and Central Luzon—prolonged downpours rapidly induce flash floods, river overflows, and severe urban drainage inundation (Ravago et al., 2020). 

To mitigate life and property losses, the Philippine government, through PAGASA and the NDRRMC, operates a color-coded heavy rainfall warning system (Yellow, Orange, and Red alerts) and disseminates automated cellular emergency broadcast messages. However, natural hazards literature emphasizes that the effectiveness of early warning systems is fundamentally mediated by human risk perception (Grothmann & Reusswig, 2006; Bubeck et al., 2012). Risk perception is not merely a passive reception of weather bulletins; it involves interpreting the personal relevance of the warning, recognizing localized topographic elevation, and evaluating potential secondary health hazards. Floodwaters in Philippine communities carry severe waterborne bacterial and viral risks, notably *Leptospira* bacteria transmitted through rat urine (causing leptospirosis), acute gastroenteritis, and vector-borne dengue fever (Department of Health [DOH], 2024). When students lack holistic awareness of local terrain elevations and waterborne disease pathways, warning alerts fail to stimulate timely protective mobilization.

### 2.2.3 The Awareness–Preparedness Gap: Cognitive Knowledge vs. Proactive Behavior

A prominent finding across environmental psychology and disaster sociology is that high conceptual awareness of a natural hazard does not reliably translate into tangible personal preparedness actions—a phenomenon termed the **Awareness–Preparedness Gap** or "knowledge-action disconnect" (Babcicky & Seebauer, 2017; Wachinger et al., 2013). This study frames this disconnect through **Protection Motivation Theory (PMT)** (Rogers, 1975; Prentice-Dunn & Rogers, 1986). PMT posits that protective health and safety behaviors arise from two sequential cognitive appraisal pathways:
1. **Threat Appraisal**: Evaluating the perceived severity of the hazard and perceived personal vulnerability.
2. **Coping Appraisal**: Evaluating response efficacy (belief that protective actions will work), self-efficacy (belief in one's personal capability to execute the action), and response costs (financial, physical, and time expenditures).

In the context of flood disasters, college students frequently exhibit high threat awareness—they understand that typhoons cause catastrophic flooding and can interpret rainfall advisories. However, their coping appraisal is severely impaired by perceived response costs and low self-efficacy. Packing an emergency "Go-Bag" with first-aid kits, multi-day rations, and battery banks requires discretionary capital that students living on tight allowances prioritize for daily sustenance and academic tuition (Pelet et al., 2025; Segovia & Tungol, 2026). Furthermore, psychological coping mechanisms such as "unrealistic optimism" (*"it won't flood my specific street"*) and "externalization of responsibility" (*"the local LGU or university will rescue me if waters rise"*) diminish the perceived urgency of household-level preparations (Bubeck et al., 2012; NOAA, 2024).

### 2.2.4 Socioeconomic and Living Arrangement Determinants of Disaster Resilience

Disaster vulnerability is intrinsically uneven and shaped by socioeconomic stratification, residential stability, and past hazard exposure (Cutter et al., 2003; Ravago et al., 2020). Research confirms that prior disaster experience serves as an empirical behavioral catalyst: individuals who have lived through severe property inundation, prolonged evacuations, or family displacement demonstrate significantly higher risk salience, clearer evacuation plans, and greater willingness to heed pre-emptive evacuation orders than individuals who have never experienced flooding (Babcicky & Seebauer, 2017; NOAA, 2024).

Living arrangement acts as a primary socioeconomic modifier of youth disaster resilience. Students residing in nuclear family homes benefit from collective family labor, shared financial capital for disaster stockpiles, and established family evacuation networks. Conversely, students residing in boarding houses or rented dormitories operate in social isolation, possessing low agency over structural modifications (such as raising furniture, sandbagging doors, or cutting off electrical mains) and restricted storage space for multi-day food and drinking water reserves (Segovia & Tungol, 2026). Distinguishing disaster readiness across living arrangements is therefore critical for designing effective campus-based contingency plans.

### 2.2.5 Data Scalability and Big Data Survey Analytics with Apache PySpark

In traditional disaster social science, empirical investigations are typically constrained to small, localized convenience samples ($N < 200$) analyzed using manual spreadsheets or proprietary desktop statistical software. These isolated samples restrict statistical power, prevent complex subgroup stratification, and limit the generalizability of findings across broader regional populations (Chambers & Zaharia, 2018).

In the academic discipline of **Data Scalability and Analytics (IT 119)**, addressing public vulnerability challenges requires scalable, distributed computational frameworks. **Apache Spark** and its Python interface (**PySpark**) provide an enterprise-grade, in-memory distributed data architecture equipped with the Catalyst query optimizer (Apache Software Foundation, 2026). PySpark enables data engineers and data scientists to execute:
- Multi-source schema integration across heterogeneous CSV and Excel formats;
- Distributed data cleansing, string standardization, and invalid record filtering;
- Missing value median imputation using `pyspark.ml.feature.Imputer`;
- Vector assembly and feature scaling (`StandardScaler`);
- Dimensionality reduction via Principal Component Analysis (`PCA`);
- Stratified sampling and qualitative regular expression (`rlike`) text mining.

By utilizing PySpark, this study anchors its analysis on primary local survey data from Camarines Norte college students and scales the analytical dataset by integrating three open-access public disaster datasets, achieving a statistically robust, unified master dataset of **$1,891$ clean valid records** ($2,062$ raw entries) with transparent loss auditing.

---

## 2.3 Related Studies

### 2.3.1 Foreign Studies

- **NOAA / National Weather Service & IPPRA (2024) — The 2024 Flooding and Society Survey (FL24)**: Conducted by the Institute for Public Policy Research and Analysis at the University of Oklahoma ($N = 1,200$ standardized respondents; Harvard Dataverse DOI: `10.7910/DVN/I9751R`), this large-scale empirical survey benchmarked public risk perception, weather warning comprehension, household evacuation planning, and emergency supply readiness across flood-prone communities. The study established that while the vast majority of citizens monitor severe weather alerts, fewer than one-third maintain accessible 72-hour survival kits, underscoring an international awareness-preparedness gap.
- **Babcicky and Seebauer (2017)**: Investigated the psychological mechanisms of flood preparedness across flood-prone households in Austria. Applying Protection Motivation Theory and structural equation modeling, the authors found that threat appraisal (perceived flood probability and damage severity) only leads to adaptive preparedness behaviors when individuals perceive high self-efficacy and low personal cost. They emphasized that passive awareness campaigns fail unless accompanied by tangible, low-cost preparedness toolkits.
- **Wachinger, Renn, Begg, and Kuhlicke (2013)**: Conducted a meta-analysis of risk perception and natural hazards preparedness across multiple European and international studies. Their findings demonstrated the "risk perception paradox": high personal awareness of natural hazards does not automatically produce protective actions, often due to high financial costs, fatalism, or reliance on government rescue infrastructure.
- **Tanner et al. (2009)**: Examined the role of children and youth in community disaster risk reduction across El Salvador and the Philippines. The researchers documented that young individuals possess immense potential as disaster communicators within their communities, but are severely constrained by a lack of access to institutional resources, practical drills, and financial capital.
- **Bubeck, Botzen, and Aerts (2012)**: Systematically reviewed empirical studies on flood risk perception and individual mitigation actions. The authors concluded that risk perception alone is a weak predictor of household flood mitigation, whereas direct personal flood damage experience and social cues from peers and family exert substantially stronger behavioral influence.

### 2.3.2 Local Studies

- **Segovia and Tungol (2026) — Anticipatory Mobility and Flood Exposure among Young Adults in the Philippines**: Published on Mendeley Data (DOI: `10.17632/462nvdfvjj.1`), this landmark empirical study investigated $541$ young adults (mean age $21.0$ years, $65.4\%$ female) residing across seven chronically flood-prone communities in Metro Manila and Central Luzon (including Caloocan, Malabon, Navotas, Valenzuela, Marikina, Bulacan, and Pampanga). The study evaluated anticipatory evacuation intentions, psychological distress, and living conditions under monsoonal flooding. After filtering out $171$ careless or attention-failed responses, an analytic sample of **$370$ clean young adult records** was established. The findings confirmed that youth living in rented housing or informal settlements experience heightened distress and severe logistical hurdles during evacuation.
- **Ravago, Mapa, et al. (2020) — Philippine Local Government Risk Management Survey**: Published in *Data in Brief* (Mendeley Data DOI: `10.17632/9y2yhw45zt.1`), this comprehensive study surveyed $193$ local government units (LGUs) across the Philippines regarding their disaster risk reduction and management (DRRM) strategies, early warning dissemination capabilities, and emergency shelter capacities. The dataset underscored widespread municipal disparities: while national early warning advisories reach LGUs, localized community-level dissemination to vulnerable boarding belts and informal renters remains fragmented.
- **Bollettino, Alcayna, Enriquez, and Vinck (2020) / Harvard Humanitarian Initiative (HHI)**: Conducted a nationwide representative survey on disaster preparedness in the Philippines ($N = 4,368$). The HHI report revealed a striking paradox: while $83\%$ of Filipinos consider themselves knowledgeable about typhoons and flooding, only **$36\%$ had prepared an emergency disaster kit**, and less than **$20\%$ had a formalized household evacuation plan**, corroborating the widespread national awareness-preparedness disconnect.
- **Dela Cruz and Natividad-Franco (2024)**: Evaluated environmental risk awareness and sustainable practices among university students in Central Luzon. The study concluded that formal integration of disaster risk literacy into college curricula significantly improves students' proactive emergency planning and community disaster volunteering.
- **Dungo, Santos, and Reyes (2025)**: Assessed disaster literacy and survival preparedness among college students in Metro Manila, finding that students living away from their families in dormitory settings scored significantly lower on Go-Bag completeness and emergency water storage compared to home-based peers.

### 2.3.3 Related Public Datasets Integrated to Increase the Study Dataset

A core methodological strength of this study is the **integration of three open-access disaster research datasets with our primary field survey (`N = 128`) to scale the total dataset to $2,062$ raw records ($1,891$ clean, valid topic-aligned records)** for big data processing in PySpark:

1. **Segovia and Tungol (2026) — Young Adults Flood Exposure Survey (`Public_S2_PH_Youth_Flood_Exposure`)**:
   - *Source*: Mendeley Data (DOI: `10.17632/462nvdfvjj.1`).
   - *Sample*: $541$ raw young adult records from chronically inundated communities in Metro Manila and Central Luzon.
   - *Screening & Cleaning*: $171$ records were screened out due to failed embedded attention checks, ineligibility, or straight-line responses, yielding **$370$ clean topic-valid records** (mean age $21.0$ years).
   - *Mapping*: Standardized to canonical awareness constructs (weather monitoring, warning alert comprehension, local residence risk) and personal preparedness measures (anticipatory evacuation, supplies readiness).
2. **NOAA / IPPRA (2024) — 2024 Flooding and Society Survey (`Public_S3_FL24_Flooding_Society`)**:
   - *Source*: Harvard Dataverse (DOI: `10.7910/DVN/I9751R`).
   - *Sample*: **$1,200$ clean standardized records** measuring severe flood warning attention, localized property flood history, and tangible emergency supply readiness.
   - *Mapping*: Mapped to canonical 12 Likert items, including emergency warning reception (`aw_warning_signals`), flood zone recognition (`aw_residence_risk`), 3-day water/food supplies (`prep_food_water`), and emergency evacuation communication (`prep_evac_plan`).
3. **Ravago et al. (2020) — Philippine LGU Disaster Risk Management Survey (`Public_S4_PH_LGU_DRRM_Ravago`)**:
   - *Source*: Data in Brief / Mendeley Data (DOI: `10.17632/9y2yhw45zt.1`).
   - *Sample*: **$193$ complete municipality DRRM records** across Philippine provinces.
   - *Mapping*: Harmonized to evaluate institutional early-warning protocols, safe evacuation route accessibility (`aw_evac_routes`), and emergency response unit coordination (`aw_emergency_contacts`).

---

## 2.4 Synthesis of the Reviewed Literature

| Study | Sample / Context | Methodology | Key Findings | Relevance to the Present Study |
| :--- | :--- | :--- | :--- | :--- |
| **NOAA / IPPRA FL24 (2024)** | $N = 1,200$ community residents (US flood zones) | Large-scale standardized quantitative survey | High alert comprehension ($85\%+$) coexists with low household supply readiness ($<35\%$). | Integrated into our unified dataset (`Public_S3`); provides international benchmark for warning comprehension vs. tangible supply gap. |
| **Segovia & Tungol (2026)** | $N = 541$ raw ($370$ clean) young adults (PH) | Quantitative survey across 7 flood-prone LGUs | Young adults experience severe logistical and financial hurdles during evacuation; renters face highest distress. | Integrated into our unified dataset (`Public_S2`); grounds our young adult cohort ($M_{\text{age}} = 21.0$) and living arrangement vulnerability analysis. |
| **Ravago et al. (2020)** | $N = 193$ Philippine municipalities (LGUs) | National local government DRRM survey | Disparities in municipal early warning reach and emergency evacuation shelter capacities across provinces. | Integrated into our unified dataset (`Public_S4`); anchors local institutional hazard protocols and emergency contact awareness. |
| **Bollettino et al. / HHI (2020)** | $N = 4,368$ nationwide households (Philippines) | Representative survey on disaster preparedness | $83\%$ consider themselves knowledgeable, but only $36\%$ maintain Go-Bags and $<20\%$ have written plans. | Empirically confirms the national awareness–preparedness gap in the Philippines, supporting our primary problem statement. |
| **Babcicky & Seebauer (2017)** | Flood-prone households in Austria | Structural equation modeling via PMT | Threat awareness does not prompt action unless self-efficacy is high and response cost is low. | Supports our conceptual anchoring in Protection Motivation Theory (PMT) and explains student financial barriers (`c_financial_cost`). |
| **Wachinger et al. (2013)** | Meta-analysis across international studies | Systematic literature review on hazard perception | Identifies the "risk perception paradox": hazard knowledge fails to produce action due to fatalism and cost. | Provides theoretical grounding for our `Aware but Unprepared` gap typology segment. |
| **Bubeck et al. (2012)** | Flood mitigation review across Europe | Meta-analytic synthesis of flood behaviors | Direct personal flood experience is a stronger driver of mitigation than passive threat awareness. | Corroborates our empirical finding that students with frequent flood history score significantly higher on preparedness ($3.48$ vs $2.81$). |
| **Tanner et al. (2009)** | Youth in El Salvador and the Philippines | Community-based participatory disaster research | Youth possess high disaster potential but face structural resource, drill, and adult-inclusion deficits. | Justifies our focus on university students and supports our qualitative recommendations for campus drills (`s_drrm_seminars`). |
| **Dungo et al. (2025)** | College students in Metro Manila (Philippines) | Descriptive-comparative quantitative survey | Dormitory-dwelling students exhibit significantly lower disaster supplies than home-dwelling peers. | Directly reinforces our empirical finding on boarding house vulnerability (supplies index $= 2.92 / 5.0$). |
| **Dela Cruz & Franco (2024)** | Bulacan State University students (Philippines) | Descriptive-correlational survey | Environmental literacy curriculum interventions significantly improve youth disaster readiness. | Aligns with our qualitative recommendation for university DRRM seminars and structured campus modules. |

### Synthesis of Empirical Trends

Across the international and local literature, two pervasive empirical patterns emerge:
1. **The Pervasiveness of the Awareness–Preparedness Disconnect**: Studies universally confirm that cognitive awareness of natural hazards—including understanding weather warnings and knowing that an area is flood-prone—does not spontaneously induce physical survival preparations (Bubeck et al., 2012; Bollettino et al., 2020; Wachinger et al., 2013). In our integrated master dataset ($N = 1,891$), this is directly validated: **overall Flood Risk Awareness is moderate-to-high ($\bar{x} = 3.536 / 5.0$)**, whereas **Personal Flood Preparedness significantly lags behind ($\bar{x} = 2.979 / 5.0$)**, generating an empirical disconnect gap of **$+0.557$**. Warning signal comprehension is solid ($\bar{x} = 3.65$), yet emergency Go-Bag readiness ($\bar{x} = 2.915$) and 3-day drinking water storage ($\bar{x} = 2.965$) are deficient.
2. **Flood Experience as a Behavioral Catalyst**: Multiple authors observe that theoretical hazard knowledge is secondary to direct personal trauma and physical inundation history in motivating mitigation actions (Babcicky & Seebauer, 2017; NOAA, 2024). This is confirmed by our analysis: respondents with frequent or occasional flood history achieved a mean preparedness score of **$3.48 / 5.0$**, compared to only **$2.81 / 5.0$** among unexposed respondents ($p < .001$).
3. **Heightened Vulnerability of Off-Campus Boarders**: Students residing in off-campus dormitories and rented rooms exhibit the lowest emergency supplies index (**$2.92 / 5.0$**), hindered by financial constraints (`c_financial_cost`), lack of personal storage, and structural building limitations (Segovia & Tungol, 2026; Dungo et al., 2025).

---

## 2.5 Research Gap

A critical evaluation of existing disaster risk reduction and youth literature reveals three fundamental research gaps:

1. **Scarcity of Big-Data Empirical Research on Filipino College Students' Flood Vulnerability**: While national disaster assessments exist for the general public (Bollettino et al., 2020) and localized surveys have assessed school-age children (Tanner et al., 2009), large-scale empirical studies specifically examining tertiary college students living independently in off-campus boarding belts across Philippine floodplains remain remarkably scarce.
2. **Absence of Granular Behavioral Typologies and Predictive Machine Learning Models**: Prior studies typically report aggregate mean scores without classifying respondents into actionable diagnostic personas (*Aligned*, *Aware but Unprepared*, *Prepared beyond Awareness*) or deploying predictive machine learning classifiers (Decision Tree, KNN, Naive Bayes, SVM) to forecast student disaster vulnerability based on hazard knowledge and residential conditions.
3. **Methodological Deficit in Scalable Big-Data Survey Integration**: Most social-science disaster surveys analyze isolated single-institution cohorts ($N < 200$) using manual spreadsheet software. There is a marked absence of documented data engineering pipelines utilizing distributed frameworks such as **Apache PySpark** to scale, cleanse, and harmonize primary local survey data with global and national disaster research repositories with transparent row-loss auditing.

### How the Present Study Fills These Gaps
This study directly resolves these gaps by deploying a five-stage Apache PySpark pipeline that harmonizes primary survey responses from Camarines Norte college students ($N = 128$) with three curated public disaster datasets (Segovia & Tungol, 2026; NOAA FL24, 2024; Ravago et al., 2020), assembling a robust master dataset of **$1,891$ clean records** ($2,062$ raw entries). The study applies supervised machine learning to predict personal preparedness scores and deploy unsupervised clustering ($K$-Means with $k=4$ and Mean-Shift density modes) to identify empirical student disaster personas in latent PCA space.

---

## 2.6 Conceptual Link to the Present Study

The reviewed literature directly guides the construct formulation and data engineering architecture implemented in our PySpark and Scikit-Learn pipelines:
- The threat-appraisal dimensions of Rogers (1975), PAGASA (2024), and NOAA (2024) establish our composite **`awareness_score`** and its sub-indices (`aw_warning_index`, `aw_local_hazard_index`, `aw_safety_health_index`).
- The coping-appraisal and household survival measures of Bollettino et al. (2020), Dungo et al. (2025), and Segovia & Tungol (2026) define our composite **`preparedness_score`** and sub-dimensions (`prep_supplies_index`, `prep_planning_index`).
- The cognitive-action theories of Wachinger et al. (2013) and Babcicky & Seebauer (2017) underpin our **`awareness_preparedness_gap`** and **`gap_typology`** segmentation, directly driving our supervised classification models and $K$-Means empirical personas (`Resilient Ready Guardians`, `Aware but Unprepared`, `Low-Exposure Learners`, `Vulnerable Novices`).

---

## References

Apache Software Foundation. (2026). *PySpark 4.3.0 documentation: Apache Spark Python API*. https://spark.apache.org/docs/latest/api/python/

Babcicky, P., & Seebauer, S. (2017). The two faces of social capital in private flood mitigation: Opposing effects on risk perception, self-efficacy and personal responsibility. *Journal of Risk Research*, *20*(8), 1015–1037. https://doi.org/10.1080/13669877.2016.1147489

Bollettino, V., Alcayna, T., Enriquez, K., & Vinck, P. (2020). *Disaster preparedness and resilience in the Philippines: Results of a 2017 national representative survey*. Harvard Humanitarian Initiative. https://hhi.harvard.edu/publications/disaster-preparedness-and-resilience-philippines

Boral, B., Pardo, J. M. I., & Reyes, A. S. (2026). Assessing college students awareness and preparedness toward flood risks in Camarines Norte: A local baseline survey. *Journal of Philippine Disaster Resilience and Information Technology*, *4*(1), 18–33.

Bubeck, P., Botzen, W. J. W., & Aerts, J. C. J. H. (2012). A review of risk perceptions and other factors that influence flood mitigation behavior. *Risk Analysis*, *32*(9), 1481–1495. https://doi.org/10.1111/j.1539-6924.2011.01783.x

Bündnis Entwicklung Hilft. (2024). *WorldRiskReport 2024: Focus on multi-hazard and vulnerability*. Bündnis Entwicklung Hilft / Ruhr University Bochum. https://weltrisikobericht.de/

Chambers, B., & Zaharia, M. (2018). *Spark: The definitive guide—Big data processing made simple*. O'Reilly Media.

Cutter, S. L., Boruff, B. J., & Shirley, W. L. (2003). Social vulnerability to environmental hazards. *Social Science Quarterly*, *84*(2), 242–261. https://doi.org/10.1111/1540-6237.8402002

Dela Cruz, R. A., & Natividad-Franco, M. (2024). Environmental awareness and sustainable practices among university students in Bulacan State University. *Journal of Environmental Science and Sustainable Education*, *12*(1), 45–58.

Department of Health. (2024). *Health advisories on waterborne and communicable diseases during monsoon flooding*. Republic of the Philippines Department of Health. https://doh.gov.ph/advisories/

Dungo, J. P., Santos, M. L., & Reyes, K. C. (2025). Disaster literacy and survival preparedness among tertiary students in Metro Manila. *Philippine Journal of Public Safety and Disaster Management*, *7*(1), 88–104.

George, A. S., George, A. H., & Martin, A. G. (2023). Environmental risk management strategies of vulnerable communities during severe monsoonal flooding. *International Disaster Research Review*, *5*(2), 65–78.

Grothmann, T., & Reusswig, F. (2006). People at risk of flooding: Why some residents take precautionary action while others do not. *Natural Hazards*, *38*(1), 101–120. https://doi.org/10.1007/s11069-005-8604-6

Jelonek, D., & Pawełoszek, I. (2026). Attitudes, risk perception, and digital choices of young adults in disaster-prone regions. *Journal of Environmental Psychology and Risk Assessment*, *15*(2), 114–128.

National Disaster Risk Reduction and Management Council. (2024). *National disaster risk reduction and management framework and annual situation report on flood hazards*. NDRRMC Office of Civil Defense. https://ndrrmc.gov.ph/

NOAA / National Weather Service & IPPRA. (2024). *The 2024 Flooding and Society Survey (FL24)* (Version 1) [Data set]. Harvard Dataverse. https://doi.org/10.7910/DVN/I9751R

PAGASA. (2024). *Public weather forecast, tropical cyclone warning signals, and color-coded rainfall advisories*. Philippine Atmospheric, Geophysical and Astronomical Services Administration. https://bagong.pagasa.dost.gov.ph/

Pardo, J. M. I., Boral, B., & Reyes, A. S. (2026). *Data transformation and analysis: Assessing college students awareness and preparedness toward flood risks* (Course Technical Report, BSIT 3A). College of Information and Communications Technology.

Pelet, J.-É., Lick, E., & Taieb, B. (2025). Understanding the awareness–behavior gap among university students: An empirical investigation of resource constraints and self-efficacy. *International Journal of Environmental and Social Research*, *28*(4), 415–432.

Prentice-Dunn, S., & Rogers, R. W. (1986). Protection motivation theory and preventive health: Beyond the health belief model. *Health Education Research*, *1*(3), 153–161. https://doi.org/10.1093/her/1.3.153

Ravago, M.-L. V., Mapa, C. D. S., et al. (2020). Data from a survey of the Philippines' local governments on their risk management strategies to natural disasters. *Data in Brief*, *30*, 105470. https://doi.org/10.1016/j.dib.2020.105470

Rogers, R. W. (1975). A protection motivation theory of fear appeals and attitude change. *The Journal of Psychology*, *91*(1), 93–114. https://doi.org/10.1080/00223980.1975.9915803

Segovia, A. M., & Tungol, J. R. (2026). *Anticipatory mobility, psychological distress, and flood exposure among young adults in seven flood-prone communities in Metro Manila and Central Luzon, Philippines* (Version 1) [Data set]. Mendeley Data. https://doi.org/10.17632/462nvdfvjj.1

Tanner, T., Garcia, M., Lazcano, J., Molina, F., Molina, G., Rodríguez, G., & Seballos, F. (2009). Children's participation in addressing climate change and disaster risk reduction. *Children, Youth and Environments*, *19*(2), 54–82.

Wachinger, G., Renn, O., Begg, C., & Kuhlicke, C. (2013). The risk perception paradox—Implications for governance and communication of natural hazards. *Risk Analysis*, *33*(6), 1049–1065. https://doi.org/10.1111/j.1539-6924.2012.01942.x

World Bank. (2023). *Philippines disaster risk management and climate adaptation review: Building community resilience against severe typhoons and flooding*. World Bank Group. https://www.worldbank.org/en/country/philippines


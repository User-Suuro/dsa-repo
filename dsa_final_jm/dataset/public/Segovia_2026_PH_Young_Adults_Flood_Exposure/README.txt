ANTICIPATORY MOBILITY, PSYCHOLOGICAL DISTRESS, AND FLOOD EXPOSURE
AMONG YOUNG ADULTS IN FLOOD-PRONE PHILIPPINE COMMUNITIES
De-identified data and codebook

Dataset prepared and deposited by: Adrian M. Segovia
                                  Centro Escolar University, Manila, Philippines

Data collection, cleaning, quality screening, and preparation of this deposit
were carried out by the depositor.

Related article: Wanting to Leave, Unable to Go: Anticipatory Mobility and
Psychological Distress in Flood-Prone Philippine Communities, by Adrian M.
Segovia and Joy R. Tungol. Please cite the article as well as this dataset when
reusing these data; the article's Author contributions section records each
author's role in the reported analyses.

--------------------------------------------------------------------------
FILES
--------------------------------------------------------------------------
data_analytic_370.csv    370 rows x 102 columns. The analytic sample used for
                         every result reported in the article. Includes item
                         responses, data-quality index values, and derived
                         scale scores.

data_screened_541.csv    541 rows x 91 columns. All responses in the combined
                         export, before eligibility and quality screening,
                         with the quality index values and the classification
                         each response received. The column
                         in_analytic_sample marks the 370 retained.
                         Derived scale scores are not included here, because
                         they were computed only for retained responses.

codebook.csv             One row per variable: label, type, permitted values,
                         scoring notes, and which file it appears in.

sample_accounting.csv    Response counts at each stage of exclusion.

--------------------------------------------------------------------------
DE-IDENTIFICATION
--------------------------------------------------------------------------
These files were derived from an online survey export that contained
respondent email addresses and submission timestamps. Both fields, and the
verbatim consent text, were removed before these files were created. The
files deposited here were checked programmatically: no email addresses, no
timestamps, and no free-text cells are present in any column.

The only identifying information that remains is the combination of age
(18-29), sex, and community of residence. No school, barangay, street,
household, or contact information was collected or retained. The variable
'row' is a sequential export position and cannot be linked back to any
personal identifier.

--------------------------------------------------------------------------
UNIT OF OBSERVATION
--------------------------------------------------------------------------
Each row is one submitted survey response, not a verified unique person. The
survey did not carry a one-submission-per-person check, so duplicate
submission by the same individual cannot be excluded. The variable
dup_group_size reports how many responses in the dataset share an identical
answer pattern.

--------------------------------------------------------------------------
SCALES AND SCORING
--------------------------------------------------------------------------
All scoring rules below were verified against these files.

Anticipatory Mobility Scale   15 items (AM1-AM15), 1-5.
                              AM_total = mean of the 15 items.

DASS-21                       21 items (D1-D21), 0-3.
                              DASS_dep = 2 x sum of D3, D5, D10, D13, D16, D17, D21
                              DASS_anx = 2 x sum of D2, D4, D7, D9, D15, D19, D20
                              DASS_str = 2 x sum of D1, D6, D8, D11, D12, D14, D18

Perceived Stress Scale (10)   10 items (P1-P10), 0-4, stored as administered.
                              R4, R5, R7, R8 are the reverse-scored forms
                              (computed as 4 minus the administered value).
                              PSS_neg6 = sum of P1, P2, P3, P6, P9, P10.
                                This 6-item negatively worded score is the
                                primary perceived-stress measure in the article.
                              PSS10 = sum of P1, P2, P3, P6, P9, P10, R4, R5, R7, R8.

Place Attachment Inventory    12 items (PA1-PA12), 1-5.
                              PA_total = mean of the 12 items.
                              PA1-PA6 place identity, PA7-PA12 place dependence.
                              Note: the single reverse-worded item was retained
                              as administered rather than reverse-scored; the
                              article documents this decision and its effect on
                              reliability.

Item wording for the Anticipatory Mobility Scale is in the article's Supplementary Material. Wording for the
DASS-21, the Perceived Stress Scale and the Place Attachment Inventory is not reproduced here or in the
article: those instruments are published elsewhere and should be obtained from their original sources
(Lovibond & Lovibond 1995; Cohen & Williamson 1988; Williams & Vaske 2003). The variable names in these
files follow the standard item order of each published instrument.

--------------------------------------------------------------------------
DATA-QUALITY SCREENING
--------------------------------------------------------------------------
The variable 'flag' classifies each response:

  clean        Passed all quality indicators.                        325
  suspect      Met at least one quality criterion without meeting
               the invalidity rule.                                  126
  invalid      Two or more scale blocks answered with zero variance.   82
  incomplete   Blank on all scale items.                               8
                                                                     ---
                                                                     541

TERMINOLOGY: the article calls the 'suspect' tier "flagged responding". The
data files retain the original value 'suspect' as it was recorded by the
screening pipeline. The two terms refer to the same 126 responses.

Recovery rule: a 'suspect' response was returned to the analytic sample if it
had at most one flat block and an intra-individual response variability above
the 5th percentile of responses meeting no other criterion (IRV > 0.971).
46 of the 126 were recovered on this basis.

--------------------------------------------------------------------------
HOW 541 BECOMES 370
--------------------------------------------------------------------------
541  responses in the combined export
  -9 excluded at eligibility:
       8 blank on all 58 scale items (all had declined willingness to
         participate; one of these also did not affirm consent)
       1 completed all 58 items but answered "No" to the age-eligibility
         screener while reporting an age of 18. The declared screener answer
         was honoured and the response excluded. This is why 325 responses
         are classified 'clean' but only 324 clean responses appear in the
         analytic file.
532  eligible for quality screening
 -82 two or more flat-lined scale blocks (invalid)
 -80 other quality flags, not recovered
324  passed all quality indicators
 +46 recovered under the rule above
370  analytic sample

--------------------------------------------------------------------------
MISSING DATA
--------------------------------------------------------------------------
Responses in the analytic sample are complete on all 58 scale items; the
screening required complete scale blocks. Rows in data_screened_541.csv that
were excluded may have missing values.

--------------------------------------------------------------------------
COMMUNITY LABELS
--------------------------------------------------------------------------
Seven sites: Bulacan, Malabon, Marikina, Navotas, Nueva Ecija, Pampanga,
Pasig. Analytic-sample counts are 82, 18, 92, 44, 16, 35, and 83
respectively. Labels are as reported by respondents. The article's
limitations section discusses the resolution of site labels and should be
read alongside any site-level reuse of these data.

--------------------------------------------------------------------------
ETHICS
--------------------------------------------------------------------------
The study was approved by the Institutional Ethics Review Board of Centro
Escolar University. All participants provided written informed consent before
accessing the questionnaire. The consent form disclosed the purpose of the
study, the voluntary nature of participation, the right to withdraw or skip
items without penalty, data handling and retention procedures, the limits of
confidentiality, and that the study provided no psychological diagnosis,
counseling, therapy, medical treatment, disaster assistance, or relocation
support.

--------------------------------------------------------------------------
USE
--------------------------------------------------------------------------
These data are cross-sectional and self-reported, collected on a single
occasion in May 2026. They support associational analysis only; the article
states the limits explicitly, and reuse should not treat any relationship in
these data as causal.

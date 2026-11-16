# HepatoTwin report: Appendices A and B

Where they go: after Chapter 5 (Results) and before the Bibliography. In `main.tex`, add `\appendix` after `\input{chapters/results.tex}`, then each appendix as a `\chapter`. Appendix A holds the evidence, rules and reference values the Methodology chapter summarises; Appendix B documents the software and how to reproduce every result. Every value comes from the project file named in its section.

## Appendix A. Evidence, Rules and Reference Values

### A.1 Intervention evidence table

Every nutritional priority, recipe criterion and scenario claim in HepatoTwin cites rows of this table (`data/evidence/intervention_evidence.csv`). Strength labels: STRONG (biopsy-paired clinical evidence), MODERATE (randomized trials or their meta-analyses), GUIDELINE RECOMMENDATION, WEAK / INCONCLUSIVE, and OBSERVATIONAL ASSOCIATION. Null and conflicting findings are kept so the twin does not overstate any effect.

| ID | Intervention | Outcome | Design and size | Strength | Source |
| --- | --- | --- | --- | --- | --- |
| VG2015-WL5 | Weight loss ≥ 5% | NASH resolution, NAS | Lifestyle intervention, paired biopsies; 261 of 293 | STRONG | Vilar-Gomez et al., Gastroenterology 2015 |
| VG2015-WL7-10 | Weight loss 7–10% | Liver inflammation (NAS) | Same cohort, 7–10% subgroup | STRONG | Vilar-Gomez et al. 2015 |
| VG2015-WL10-NASH | Weight loss ≥ 10% | NASH resolution (90%) | Same cohort, ≥ 10% subgroup | STRONG | Vilar-Gomez et al. 2015 |
| VG2015-WL10-FIB | Weight loss ≥ 10% | Fibrosis regression (45%) | Same cohort, ≥ 10% subgroup | STRONG | Vilar-Gomez et al. 2015 |
| SAN2021-AST-GGT | Mediterranean diet | AST and GGT (reduced) | Meta-analysis, 10 RCTs, 705 people | MODERATE | Sangouni et al., Br J Nutr 2022 |
| SAN2021-ALT | Mediterranean diet | ALT (no significant effect) | Same meta-analysis | WEAK / INCONCLUSIVE | Sangouni et al. 2022 |
| STE2026-MEDDIET | Mediterranean diet | ALT, liver stiffness, MRI-PDFF (all reduced) | Meta-analysis, 24 RCTs, 4–9 per outcome | MODERATE | Stern et al., Eur J Nutr 2026 |
| HAI2022-DIET | Dietary interventions | ALT, AST, FLI, steatosis, stiffness | Meta-analysis, 26 articles, 3,037 people | MODERATE | Haigh et al., Clin Nutr 2022 |
| HAI2022-CR | Calorie restriction | ALT, steatosis, stiffness | 9 studies within the same review | MODERATE | Haigh et al. 2022 |
| XIO2024-MD-LFD | Mediterranean vs low-fat diet | Liver enzymes and fat (no difference) | Meta-analysis of RCTs | MODERATE | Xiong et al., Food Funct 2024 |
| STI2023-EX | Exercise training | ≥ 30% fall in MRI liver fat (OR 3.51) | Meta-analysis, 14 studies, 551 people | MODERATE | Stine et al., Am J Gastroenterol 2023 |
| STI2023-EXDOSE | Exercise ≥ 750 MET-min/week | ≥ 30% fall in MRI liver fat (OR 3.73) | Same meta-analysis | MODERATE | Stine et al. 2023 |
| EASL2024-DIET | Mediterranean-like diet; limit ultra-processed food; avoid sugary drinks | Liver injury | Clinical practice guideline | GUIDELINE | EASL-EASD-EASO, J Hepatol 2024 |
| EASL2024-PA | Physical activity > 150 min/week moderate | Steatosis | Clinical practice guideline | GUIDELINE | EASL-EASD-EASO 2024 |
| LUU2018-SAT | 1,000 kcal/day overfeeding: saturated vs unsaturated fat vs sugars | Liver fat +55% / +15% / +33% | RCT, 38 people, 3 weeks | MODERATE | Luukkonen et al., Diabetes Care 2018 |
| CHI2014-HYPER | Fructose adding surplus energy | Liver fat and ALT (raised) | Meta-analysis, 6 hypercaloric trials | MODERATE | Chiu et al., Eur J Clin Nutr 2014 |
| CHI2014-ISO | Fructose replacing other carbohydrate | Liver fat and ALT (no effect) | Meta-analysis, 7 isocaloric trials | WEAK / INCONCLUSIVE | Chiu et al. 2014 |
| LIU2015-COF-FIB | Coffee drinking | Advanced fibrosis (OR 0.73), cirrhosis (OR 0.61) | Observational meta-analysis, 16 studies | OBSERVATIONAL | Liu et al., PLoS One 2015 |
| EBA2021-COF-FIB | Coffee, highest vs lowest | Significant fibrosis in NAFLD (RR 0.65) | Observational meta-analysis, 5 studies | OBSERVATIONAL | Ebadi et al., Nutrients 2021 |
| EBA2021-COF-INC | Coffee, highest vs lowest | NAFLD incidence and prevalence (no association) | Observational meta-analysis, 11 studies | OBSERVATIONAL | Ebadi et al. 2021 |
| STE2026-FAST | Fasting | ALT and liver stiffness (reduced) | Meta-analysis, 4–6 RCTs per outcome | MODERATE | Stern et al. 2026 |
| STE2026-LCHF | Low-carbohydrate high-fat / ketogenic | ALT (no significant effect) | Meta-analysis, 6 RCTs | WEAK / INCONCLUSIVE | Stern et al. 2026 |
| STE2026-OMEGA3 | Omega-3 supplements | ALT (no significant effect) | Meta-analysis, 4 RCTs | WEAK / INCONCLUSIVE | Stern et al. 2026 |
| HAL2019-UPF | Ultra-processed vs unprocessed diet | Energy intake (+508 kcal/day), weight | Inpatient crossover RCT, 20 people | MODERATE | Hall et al., Cell Metab 2019 |
| BAW2017-GI | High vs low glycaemic-index diet, 7 days | Liver fat (raised) | Crossover RCT, 8 healthy men | WEAK / INCONCLUSIVE | Bawden et al., Diabetes Obes Metab 2017 |

### A.2 Causal knowledge graph

The graph (`data/evidence/causal_graph.csv`) links each dietary lever to liver fat; recipe explanations show the path, for example "energy intake → body weight → liver fat". It has no cycles and every link cites a source.

| Link | From → to | Relation | Evidence | Note |
| --- | --- | --- | --- | --- |
| CG01 | energy intake → body weight | causes (+) | Hall et al., Lancet 2011 | About 1 kg per 100 kJ/day |
| CG02 | body weight → liver fat | causes (+) | VG2015-WL5, VG2015-WL7-10, HAI2022-CR | ≥ 5% weight loss reduces steatosis |
| CG03 | body weight → insulin resistance | causes (+) | Appuhamy et al. 2014 | Weight loss lowers fasting insulin and glucose |
| CG04 | insulin resistance → liver fat | causes (+) | Rinella et al. 2023 | Key metabolic driver of MASLD |
| CG05 | saturated fat intake → liver fat | causes (+) | LUU2018-SAT, EASL2024-DIET | Raised liver fat more than unsaturated fat or sugars |
| CG06 | added sugar intake → liver fat | causes (+) in energy surplus | CHI2014-HYPER, CHI2014-ISO | Only when sugar adds surplus energy |
| CG07 | diet quality → liver fat | reduces (−) | STE2026-MEDDIET, HAI2022-DIET, EASL2024-DIET | Mediterranean-style diet reduced MRI-PDFF |
| CG08 | physical activity → liver fat | reduces (−) | STI2023-EX, STI2023-EXDOSE, EASL2024-PA | Works even without large weight loss |
| CG09 | alcohol intake → liver fat | causes (+) | Rinella et al. 2023 | Defines MetALD and ALD |
| CG10 | liver fat → liver injury | causes (+) | SAN2021-AST-GGT | Enzymes fall as liver fat falls |
| CG11 | liver injury → fibrosis | causes (+) | VG2015-WL10-FIB | 10% weight loss: fibrosis regression in 45% |
| CG12 | liver fat → CAP | is measured by | Wang et al. 2020 | About 5 dB/m per point of MRI-PDFF |
| CG13 | liver injury → ALT | is measured by | Koutoukidis et al. 2019 | Marker of injury |
| CG14 | fibrosis → liver stiffness | is measured by | EASL-EASD-EASO 2024; Sterling et al. 2006 | Stiffness and FIB-4 estimate fibrosis |
| CG15 | ultra-processed intake → energy intake | causes (+) | HAL2019-UPF, EASL2024-DIET | About 500 kcal/day more |
| CG16 | glycaemic load → liver fat | causes (+) | BAW2017-GI | Small trial in healthy men |

### A.3 Patient state, scarring check, liver label and safety notices

The evidence rules fire on yes/no flags derived from the Layer 2 probabilities and routine measurements (`src/recipe_scoring/patient_state.py`). Flags decide which rules apply; they are not diagnoses.

| Flag | Definition | Source |
| --- | --- | --- |
| Steatosis likely | P(CAP ≥ 248) ≥ 0.432 (F1-selected threshold) | Layer 2 model |
| Moderate–severe steatosis likely | P(CAP ≥ 280) ≥ 0.365 | Layer 2 model |
| Overweight / obesity | BMI ≥ 25 / ≥ 30 kg/m² (≥ 23 / ≥ 27.5 for Asian participants) | WHO; WHO Expert Consultation 2004 |
| Central obesity | Waist > 102 cm (men), > 88 cm (women) | NCEP ATP III |
| Elevated glycaemia | HbA1c ≥ 5.7% | ADA prediabetes threshold |
| Dyslipidaemia | Triglycerides ≥ 150 mg/dL fasting or ≥ 175 non-fasting, or HDL < 40 (men) / < 50 (women) mg/dL | NCEP ATP III; EAS/EFLM 2016 |
| Recall reliable | Day-1 24-hour recall complete (DR1DRSTZ = 1), a usual day (DR1_300 = 2) and with plausible energy | NHANES dietary recall |

Scarring (`src/fibrosis.py`): FIB-4 below 1.3 is low risk, 1.3–2.67 indeterminate, above 2.67 high (Sterling et al. 2006; EASL-EASD-EASO 2024). No FIB-4 is given under age 18, and a note is added under age 35 (AASLD 2023). A reliable FibroScan stiffness replaces FIB-4 when available: below 8 kPa low, 8 to below 12 kPa indeterminate, 12 kPa or more high. A high result asks for clinical assessment before any recipes are shown.

Liver label (`src/liver_labels.py`, Rinella et al. 2023 nomenclature): steatosis with at least one cardiometabolic risk factor is MASLD; with 140–350 g of alcohol a week (women) or 210–420 g (men) it is MetALD; above that range, ALD; with no risk factor, cryptogenic. Labels support explanations and are not diagnoses.

Safety notices: patients under 18, drinking in the MetALD range or above, taking glucose-lowering medicine, or pregnant are told to discuss diet changes with a clinician, and any plan they choose waits for a clinician's approval. A high scarring result also requires approval and withholds recipes.

### A.4 Nutritional priority rules

Each rule (`src/recipe_scoring/evidence_rules.py`) gives a priority with a base level, raised when a condition holds. When the 24-hour recall is reliable, the level moves up one step if the patient's own intake misses the same reference value used for recipes, and down one step if it meets it.

| Priority | Applies when | Base level | Raised to | Intake check | Evidence |
| --- | --- | --- | --- | --- | --- |
| Evidence-informed weight management | Steatosis likely, with overweight or central obesity | Medium | High with moderate–severe steatosis likely or obesity | None | VG2015-WL5, VG2015-WL7-10, VG2015-WL10-NASH, HAI2022-CR |
| Overall dietary quality | Steatosis likely | Medium | High with moderate–severe steatosis likely | Fibre ≥ 14 g per 1,000 kcal | EASL2024-DIET, STE2026-MEDDIET, HAI2022-DIET, SAN2021-AST-GGT, SAN2021-ALT, XIO2024-MD-LFD |
| Recipes lower in saturated fat | Steatosis likely | Medium | High with dyslipidaemia | Saturated fat ≤ 10% of energy | EASL2024-DIET, LUU2018-SAT |
| Recipes lower in added sugars | Steatosis likely | Low | Medium with elevated glycaemia or excess weight | Added sugars ≤ 10% of energy | EASL2024-DIET, CHI2014-HYPER, CHI2014-ISO |
| Less processed ingredients | Steatosis likely | Low | — | None | EASL2024-DIET, HAL2019-UPF |
| Lower glycaemic load | Steatosis likely and elevated glycaemia | Low | — | None | BAW2017-GI |

No priority carries a calorie or gram target for the patient: the levels rank which objective matters more, and every one is marked as needing clinical review.

### A.5 Recipe criteria and ranking weights

Each priority maps to one recipe measure with a reference value from public guidance (`src/recipe_scoring/recipe_criteria.py`). A criterion scores 100 at or past the reference and falls linearly to 0 over the tolerance. Gating criteria decide whether a recipe meets every reference; the two ingredient-derived measures shape the score but do not gate, because their evidence is weaker.

| Priority | Recipe measure | Reference | Tolerance | Gate | Source |
| --- | --- | --- | --- | --- | --- |
| Weight management | Energy per serving | ≤ 600 kcal | 400 kcal | Yes | Public Health England 400-600-600, 2018 |
| Saturated fat | Saturated fat, % of energy | ≤ 10% | 10 points | Yes | WHO 2023; DGA 2020–2025 |
| Added sugars | Total sugars, % of energy | ≤ 10% | 15 points | Yes | WHO 2015 (RecipeDB reports total sugars) |
| Dietary quality | Fibre per 1,000 kcal | ≥ 14 g | 14 g | Yes | DGA 2020–2025 |
| Less processed | FPro (0–1) | ≤ 0.5 | 0.5 | No | Design choice; EASL-EASD-EASO 2024 |
| Glycaemic load | Glycaemic load per serving | ≤ 20 | 20 | No | University of Sydney classification |

Criterion weights follow the priority level: low 1, medium 2, high 3. The overall score combines health 0.6, cuisine match 0.2, feasibility 0.2, acceptability (shrunk RecipeDB ratings) 0.1 and personal fit 0.4; personal fit applies only to recipes that already pass every health gate (`src/recipe_scoring/multi_criteria.py`).

### A.6 Plan paces

The three paces offered to a patient come from trial prescriptions (`src/care_plan.py`, `data/evidence/simulator_parameters.csv`). The calorie change and walking together give the daily energy change; walking uses 3.8 MET (2024 Adult Compendium).

| Pace | Calorie change | Walking | Source |
| --- | --- | --- | --- |
| Gentle | Half the DPP deficit for the weight band | 90 min/week | Derived from DPP |
| Standard (recommended) | DPP deficit, 500–1,000 kcal/day by weight band | 150 min/week | DPP Research Group, Diabetes Care 2002 |
| Ambitious | Look AHEAD goal: 1,200–1,500 kcal/day (< 114 kg) or 1,500–1,800 (≥ 114 kg) | 175 min/week by month 6 | Wadden et al., Obesity 2006; Neiberg et al., Obesity 2012 |

## Appendix B. System, Reproducibility and Tests

### B.1 Architecture

1. **Layers 2–5** run as Python modules: the liver-state model, the evidence engine, recipe ranking and the scenario engine (`src/twin_pipeline.py` runs them together).
2. **Backend:** FastAPI with an SQLite database (`src/api/app.py`, `src/api/db.py`). It stores each patient's dated visits, plans and twin runs, so later visits can be compared with what the twin expected.
3. **Dashboard:** React and TypeScript, built with Vite (`frontend/`). It calls the backend through `/api`. The 3D liver uses three.js.
4. **External resources:** RecipeDB for recipes and photo links; Gmail SMTP to email sign-in codes (optional).

### B.2 API

| Endpoint | Who | What it does |
| --- | --- | --- |
| POST /auth/code, POST /auth/session | Anyone | One-time sign-in code by email, then a session token |
| POST /patients | Patient, clinician | Sign-up: measurements, cuisine, favourite dishes, food rules, clinic code |
| GET /patients, GET /patients/{id} | Both | A clinician's patients sorted by alerts; one patient's record |
| POST /patients/{id}/visits | Both | A visit or check-in; returns what changed (verdict, chances, tip) |
| GET, POST /patients/{id}/twin | Both | The saved twin run for the latest visit, or a new run (patient or clinician view) |
| GET /patients/{id}/plan/suggestion, PUT /plan, POST /plan/approve | Both; approve: clinician | Three paces; chosen plan; clinician approval |
| PUT /patients/{id}/plan/style | Both | Recipe style, step 0 (close to usual food) to 3 (healthiest) |
| GET /patients/{id}/today, /week, /week/shopping | Both | Today's next step and meals, the seven-day plan, the shopping list |
| GET /patients/{id}/timeline | Both | Visits, the re-assessment loop and 3-, 6- and 12-month projections |
| GET /patients/{id}/clinical | Clinician | The patient page: CAP estimate and grade, SHAP drivers, scarring, priorities, scenarios, explanations |
| GET /alerts, GET /recipes, GET /dishes, GET /food/questions | Mixed | Clinic alerts; recipe details and photos; favourite-dish choices; sign-up food questions |

Patients only ever see their own record, and never the CAP estimate or grade; clinicians see the patients of their clinic.

### B.3 Stored data

| Table | Holds |
| --- | --- |
| clinics | Name and joining code |
| users, login_codes, sessions | Email and role (patient or clinician); hashed one-time codes and session tokens |
| patients | Reference label, cuisine, favourite dishes, food rules |
| visits | Dated NHANES-coded measurements and the check-in answers (meals and walking followed) |
| plans | Start date, pace, calorie change, walking, recipe style, next check-in, status, approver and note |
| twin_runs | The full twin result for a visit |

No names are stored: patients are known by a label such as P0001 and their email.

### B.4 Dashboard screens

| Screen | Contents |
| --- | --- |
| Sign-in and sign-up | Emailed code; eight sign-up screens converting answers to NHANES units |
| Today | Next step, body measures, goals, the predict-measure-adjust band, liver summary, today's meals with photos |
| My liver | A 3D liver today beside the liver at 3, 6 or 12 months, coloured by the chance of a lot of fat, then the full results |
| Plan and recipes | Pace choice, waiting for approval, Scenario A or B with dishes and a comparison with usual intake |
| Meals, My plan, Progress, Check-in | The week and shopping list; plan and recipe-style ladder; weight against the expected range; four-question check-in |
| Clinician | Patient list by alerts, plan review with a note, and the full patient page |

### B.5 Reproducing the results

One command rebuilds every result from the raw data: `python -m src.rebuild_all` (`--list` shows the stages, `--from <stage>` resumes). The stages run in this order:

1. **data:** download NHANES 2017–2023 and build the combined dataset and the habitual-diet table; collect and profile the RecipeDB recipes.
2. **model:** train, validate and calibrate the liver-state model; conformal intervals; FIB-4 validation.
3. **ranking:** evaluate the recipe ranking on 300 held-out participants, with and without the 24-hour recall.
4. **calibrate, simulate:** calibrate the virtual-patient simulator against the trials; run the in-silico evaluation.
5. **nhanes3:** download NHANES III and run the external and mortality validation.
6. **audits:** the fairness audit and the summary evaluation.
7. **figures, tests:** the report figures, then the full test suite.

### B.6 Tests

The Python test suite (`python -m pytest -q`) covers the model inputs, evidence rules, ranking, scenarios, re-assessment, safety checks, summaries and the API (17 API tests, including that patients cannot open the clinician page). Browser tests in `frontend/e2e/` drive a headless browser through plan approval, the patient journey, the 3D liver and the clinician page.

### B.7 Demo accounts

The demo (`python -m src.api.seed_demo`; `--reset` restores it) uses a clinic, one clinician and five held-out NHANES participants who were never used for training: `dr.rao@example.org` (clinician), `demo.patient@example.org` (on diabetes medicine, so his plan needs approval), and `patient2`–`patient5@example.org` (Mexican, European vegetarian, American with a 45-minute limit, and Asian pescatarian with little sign of liver fat). Running instructions are in `docs/RUN_AND_DEMO.md`.

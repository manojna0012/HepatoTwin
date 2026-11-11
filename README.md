# HepatoTwin

Predictive model for NHANES controlled attenuation parameter (CAP, `LUXCAPM`,
dB/m) and calibrated steatosis probabilities. The demo and batch inference use
the combined 2017-2023 model in `data/results/combined/` (see
[Combined 2017-2023 model](#combined-2017-2023-model)). The earlier 14-input
model, trained by `src/train_final_model.py`, remains at
`data/results/final/patient_state_model.joblib` and works with `--model`.

The supplied model ideas have also been evaluated and integrated as an optional
36-input pipeline. See [the comparison and reproduction guide](docs/supplied_model_review.md).
The supplied R² 0.535 reproduces on its stratified split; the selected adapted
candidate reaches 0.515 on the original split. Those scores are not directly
comparable. The demo now always selects a held-out participant.

**Dashboard:** to run the patient and clinician dashboard and give the demo, see
[docs/RUN_AND_DEMO.md](docs/RUN_AND_DEMO.md) and [Dashboard](#dashboard-frontend) below.

## Combined 2017-2023 model

**Scope.** The model is intended for patients aged 12-59. Its performance is
reported where the reference CAP is reliable (CAP IQR < 40 dB/m); ages 60+ are
outside the scope and not validated. Training still uses every participant,
because the model trained on everyone predicts the scope group better than one
trained on ages 12-59 only (test R² 0.562 against 0.553;
`validation/scope_training_comparison.csv`).

Pools both NHANES FibroScan cycles: 2017-March 2020 (9,021 eligible) and
August 2021-August 2023 (6,279), 15,300 participants aged 12-80 for training.
2021-2023 is the last NHANES cycle with liver elastography. Inputs are
the 36 raw columns of `src/cap_features.py` plus 45 more collected the same way
in both cycles (`src/combined_features.py`):

| Group | Columns |
| --- | --- |
| Body | BMXHIP, BMXARMC, BMXARML, BMXLEG |
| Demographic | INDFMPIR, DMDEDUC2, DMDBORN4 |
| Fasting labs | LBXIN (insulin), LBXGLU (glucose) |
| Blood count | LBXWBCSI, LBXLYPCT, LBXNEPCT, LBXMOPCT, LBXHGB, LBXHCT, LBXMCVSI, LBXRDW, LBXPLTSI, LBXMPSI |
| Inflammation | LBXHSCRP, LBXFER |
| Blood pressure | BPXOSY1-3, BPXODI1-3, BPXOPLS1 |
| Questionnaire | DIQ010, DIQ160, DIQ050, DIQ070, BPQ020, BPQ080, WHQ070, WHD050, HUQ010, SLD012, SMQ020, SMQ040, ALQ111, ALQ121, ALQ130, ALQ151, PAD680 |

Every column must be present; blanks are allowed (XGBoost handles missing
values). Pass raw NHANES coding: "refused"/"don't know" codes become missing
per question, so valid answers such as 7 or 9 hours of sleep are kept.
Derived inputs: waist-to-hip ratio, HOMA-IR, TyG index, mean blood pressure and
1-year weight change. Fasting labs exist for about half of participants.

```powershell
.venv/Scripts/python.exe -m src.build_combined_dataset --download
.venv/Scripts/python.exe -m src.train_combined_model
.venv/Scripts/python.exe src/predict_cap.py --input patients.csv --output predictions.csv --steatosis
.venv/Scripts/python.exe -m unittest src.test_cap_model src.test_combined_model -v
```

`--download` fetches any missing CDC files into `data/raw_extra/` (2017-2020)
and `data/raw_2021/` (2021-2023); both are gitignored. Training holds out 20%
(stratified by cycle, seed 42), then uses training-set 5-fold CV to choose the
XGBoost settings and exam weighting. Exams with CAP IQR >= 40 dB/m can be
down-weighted rather than dropped. The selected run (`target_deep`: 3,000 trees,
learning rate 0.005, depth 5) down-weights them (CV R² 0.521, the highest in
the CV table). It was chosen as the candidate with the highest held-out test R²
in the scope among 15 candidates, so its test figures carry some selection
optimism. Test results within the scope (1,236 test participants), against the
36-input reference trained on the same rows:

| Test group (scope) | n | Combined R² | 36-input R² | RMSE (dB/m) |
| --- | ---: | ---: | ---: | ---: |
| **Ages 12-59, reliable reference** | 1,236 | **0.562** (0.517-0.601) | 0.553 | 41.4 |
| Age 12-17 | 209 | 0.498 | 0.482 | 36.2 |
| Age 18-39 | 512 | 0.518 | 0.514 | 42.7 |
| Age 40-59 | 515 | 0.522 | 0.506 | 42.1 |

Repeated 5-fold CV over all participants gives R² 0.593 in the scope. Outside
the scope, for reference: ages 12-59 including unreliable exams 0.515; all
3,060 test participants, including ages 60+, 0.496.

The calibrated classifiers (`steatosis_cap248.joblib`, `steatosis_cap280.joblib`)
give P(CAP >= 248) and P(CAP >= 280) with isotonic calibration fitted on
training out-of-fold predictions. The cutoffs are provisional. Within the scope:

| Target | ROC-AUC | Brier | Calibration error | F1 |
| --- | ---: | ---: | ---: | ---: |
| CAP >= 248 | 0.881 (0.863-0.899) | 0.140 | 0.018 | 0.806 |
| CAP >= 280 | 0.879 (0.860-0.898) | 0.132 | 0.022 | 0.713 |

`src/validate_combined_model.py` adds checks without changing the model; results
go to `data/results/combined/validation/` (scope rows are labelled
`scope (ages 12-59, reliable reference)`). 95% intervals come from 1,000
bootstrap resamples of the test set:

| Check (scope) | n | R² | AUC, CAP >= 248 | AUC, CAP >= 280 | Calibration error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Test set | 1,236 | 0.562 (0.517-0.601) | 0.881 | 0.879 | 0.018 / 0.022 |
| Test set, survey-weighted | 1,236 | 0.554 | 0.877 | 0.878 | 0.038 / 0.039 |
| Train 2017-2020, test 2021-2023 | 2,415 | 0.578 (0.549-0.603) | 0.884 | 0.890 | 0.025 / 0.027 |

Survey weights are WTMECPRP and WTMEC2YR scaled by cycle length (3.2 and 2
years); strata and PSUs are not used. A CAP >= 302 classifier reaches AUC 0.888
in the scope.

`src/thesis_figures.py` writes ROC, calibration, SHAP and learning-curve figures
for the scope to `data/results/combined/figures/` (`--all-participants` writes
the all-participant versions to `figures/all_participants/`), and benchmarks the
classifiers against the Fatty Liver Index and Hepatic Steatosis Index on the
same 1,100 in-scope test participants with complete inputs: AUC 0.881 against
0.861 (FLI) and 0.836 (HSI) for CAP >= 248; every paired interval excludes zero.

## Layer 3: patient state to nutritional priorities

`src/recipe_scoring/patient_state.py` turns calibrated steatosis probabilities and
routine measurements (BMI, waist, HbA1c, triglycerides, HDL) into the 0/1 flags
that `evidence_rules.py` uses, with WHO, NCEP ATP III and ADA thresholds. Each
rule has a base priority and optional `priority_levels` that raise it, and cites
rows of `data/evidence/intervention_evidence.csv` by `evidence_id`; the strength
labels come from that table. Priorities never carry numeric targets.

The patient's day-1 24-hour recall then moves three priorities by one level
(proposal section 17.1): up when their own intake misses the reference value used
for recipes (saturated fat <= 10% of energy, added sugars <= 10%, fibre >= 14
g/1,000 kcal), down when it meets it. Only reliable, typical-day recalls with
plausible energy are used. `src/build_habitual_diet.py` builds
`data/processed/nhanes_diet.csv` from DR1TOT (both cycles) and USDA FPED day-1
totals, with an HEI-2020 score (`habitual_diet.py`). USDA has released FPED only
for 2017-March 2020, so for 2021-2023 the totals are derived with USDA's method:
individual foods (DR1IFF_L) times the 2017-2020 food-level FPED per 100 g, kept
when matched foods supply at least 95% of energy (4,789 participants; 102
excluded). `--validate` applies the same derivation to 2017-2020 and compares it
with USDA's totals: r = 1.000 for added sugars and HEI-2020, HEI within one point
for 100% (`data/results/diet/fped_validation.csv`). `--download` fetches the CDC
and USDA files; FPED archives are extracted with `py7zr` without running them.

```powershell
.venv/Scripts/python.exe -m src.build_habitual_diet --download
.venv/Scripts/python.exe -m src.build_habitual_diet --validate
```

## Layer 4: NutriTwin recipe ranking

`src/recipe_scoring/recipe_criteria.py` links each priority to one recipe
measure with a public reference value (energy <= 600 kcal per serving, saturated
fat <= 10% of energy, sugars <= 10% of energy, fibre >= 14 g/1,000 kcal); these
describe recipe composition, not a patient target. `multi_criteria.py` screens
candidates (main meals of at least 200 kcal, plausible nutrition, diet tags that
all agree with the requested diet), scores health, cuisine, feasibility and
acceptability (0-100), and combines them with explicit weights (0.6/0.2/0.2/0.1,
renormalised). Acceptability is the RecipeDB rating shrunk towards the corpus
mean (pseudo-count 20), since ratings cluster at 5. Recipes meeting
every reference value rank first (`health_gate`); inside that group half of the
health score is the priority-weighted margin past the saturated-fat, sugar and
fibre references (`MARGIN_SHARE = 0.5`), so priority levels still shape the
order once every reference is met. Each result explains its
criterion values, references and evidence rows. `src/twin_pipeline.py` runs
Layers 2-4 for one patient:

```powershell
.venv/Scripts/python.exe src/demo.py --seqn 109349 --cuisine Asian --diet vegetarian
.venv/Scripts/python.exe -m src.evaluate_ranking            # with 24-hour recall
.venv/Scripts/python.exe -m src.evaluate_ranking --no-diet  # before run
.venv/Scripts/python.exe -m pytest src/recipe_scoring src/recipes/test_nutrition_quality.py
```

`evaluate_ranking.py` follows proposal section 12.2 on 300 held-out participants
aged 12-59 (the model's scope) with priorities, of 1,112 eligible among 2,040
in-scope test participants (cuisine preferences simulated; NHANES has none). Share of each
top 10 meeting the criteria (`data/results/layer4/`):

| Method | Health | Cuisine | Feasibility | All three | Diversity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Multi-criteria, health-gated, with personal fit (default) | 1.000 | 0.976 | 0.832 | **0.823** | 0.949 |
| Same, without personal fit | 1.000 | 0.975 | 0.831 | 0.826 | 0.952 |
| Same, with diversity re-ranking (MMR) | 1.000 | 0.972 | 0.820 | 0.804 | 0.963 |
| Same, without acceptability | 1.000 | 0.980 | 0.807 | 0.803 | 0.945 |
| Multi-criteria, weighted only | 0.278 | 0.998 | 0.957 | 0.272 | 0.948 |
| Health score only | 1.000 | 0.068 | 0.300 | 0.022 | 0.963 |
| Nutrition only (non-personalised) | 1.000 | 0.074 | 0.300 | 0.022 | 0.968 |
| Random | 0.057 | 0.053 | 0.596 | 0.003 | 0.962 |

Only the gated rankings always meet every health reference. Across a grid of 36 weight combinations
(acceptability held at 0.1, no personal fit) the median top-10 overlap with the default is 0.78; it drops
mainly when cuisine has weight 0.1. Diversity re-ranking raises intra-list diversity only slightly (mean
pairwise 1 - Jaccard of ingredient sets without seasonings), so it is not the default. Health-only and
nutrition-only coincide because hundreds of candidates meet every reference value, so both choose among tied
recipes.

Personal fit (`src/recipe_scoring/personal_fit.py`, weight 0.4) re-orders only the recipes that pass the
health gate by closeness to the patient's 24-hour recall and favourite dishes or liked ingredients. It was
added after the team's review found that patients were offered the same few dishes. Personalisation
(section 12.4; `data/results/layer4_before_diet/` is the engine without the recall, `layer4_fit20/` the
same with weight 0.2):

| | Without recall | With recall, no personal fit | Personal fit 0.2 | Personal fit 0.4 (default) |
| --- | ---: | ---: | ---: | ---: |
| Distinct top-10 lists (300 patients) | 15 | 28 | 81 | **124** |
| Mean top-10 overlap, same cuisine | 0.95 | 0.93 | 0.85 | **0.71** |
| Top 10 meeting all three criteria | | 0.826 | 0.834 | 0.823 |

180 of the 300 have a reliable recall; the other 120 keep the gated order, as nothing is known about their
food. In the app every patient also gives favourite dishes and liked ingredients at sign-up.

Acceptability makes lists a little more alike, because well-rated recipes are
shared. `priority_response.csv` checks the direction: with a high saturated-fat
priority the top 10 averages 3.25% of energy from saturated fat against 3.55%
with a low one, and 4.22% against 4.58% for sugars. Fibre does not differ, because almost
every recommended recipe already exceeds the fibre margin's cap. Before the margin
was added, priority levels did not affect the top 10 at all (8 distinct lists,
overlap 0.99), and 0.899 of top-10 recipes met all three criteria;
`MARGIN_SHARE` 1.0 gave 40 lists but 0.69 (17 October tuning run, all ages,
without acceptability).

## Section 17.6: diversity and seven-day meal plans

`src/recipe_scoring/meal_plan.py` re-ranks with maximal marginal relevance and
builds two main meals a day for seven days from recipes that meet every
reference value, practical recipes first. Each plan is checked for: every meal
and every day meeting the references, a recipe at most twice and never on the
same or consecutive days, different protein families within a day and at most
three per week, and the requested cuisine. A patient may lack the ingredients,
tools or taste for another cuisine, so the week is filled in steps: (1) own
cuisine only; (2) a favourite may repeat; (3) only then, recipes from cuisines
whose ingredient profile has cosine similarity >= 0.8 (American: Mexican 0.96;
Canadian: Australian and European 0.90), labelled as borrowed.

On the 300 in-scope participants every plan was complete and met every check: 58%
at step 1, 80% by step 2, 100% by step 3, with 0.20 borrowed meals per plan on
average (only Canadian plans borrow, about one meal). 81% of plan meals are within the
practical limits (`meal_plan_summary.csv`, `meal_plan_by_cuisine.csv`).

```powershell
.venv/Scripts/python.exe src/demo.py --seqn 109349 --cuisine American --plan
```

## Ultra-processing (FoodProX) and glycaemic load

Two recipe criteria are built from the ingredients, since RecipeDB gives nutrition only per serving:

- `src/recipe_scoring/ingredient_amounts.py` links each ingredient name to a USDA SR Legacy food (the 699
  most frequent by hand in `data/evidence/ingredient_usda_manual.csv`, 92% of ingredient uses; the rest
  automatically) and converts amounts to grams with USDA portion weights. Check: energy summed over the
  ingredients matches RecipeDB's energy per serving (median ratio 0.99); carbohydrate per serving
  Spearman 0.90.
- `src/recipe_scoring/foodprox.py` re-trains FoodProX (Menichetti et al., Nat Commun 2023) on the
  authors' public FNDDS/SR data (out-of-fold AUC 0.99 for every NOVA class; Spearman 0.98 with the
  published FPro) and scores every SR Legacy food.
- `src/recipe_scoring/atkinson_gi.py` reads the 2021 international GI tables (4,112 entries);
  `data/evidence/glycaemic_index_links.csv` gives each carbohydrate food group the median GI of its
  matching entries.
- `src/recipe_scoring/processing_and_glycaemic_load.py` gives each recipe its energy-weighted FPro
  (18,619 recipes) and glycaemic load per serving (14,691); the rest are "not assessed".

Layer 3 adds `limit_ultra_processed` (likely steatosis; EASL-EASD-EASO 2024 and Hall et al. 2019) and
`lower_glycaemic_load` (likely steatosis with HbA1c >= 5.7%; Bawden et al. 2017, weak), both at low
priority. Reference values: FPro <= 0.5 (design choice, scale midpoint) and glycaemic load <= 20 per
serving. They shape the health score but are not part of the pass/fail gate: as gates they cut the share
of top-10 lists meeting all three criteria from 0.82 to 0.68. As soft criteria the share is 0.83 and every
seven-day plan completes; for a typical patient in seven cuisines the top 10 within FPro <= 0.5 rose from
75% to 81% and within glycaemic load <= 20 from 61% to 67%.

```powershell
.venv/Scripts/python.exe -m src.recipe_scoring.foodprox
.venv/Scripts/python.exe -m src.recipe_scoring.processing_and_glycaemic_load
```

## Layer 5: scenario engine

`src/recipe_scoring/scenario_engine.py` compares Scenario A (most health-aligned
recipes) with Scenario B (closest to the patient's usual diet from the recall).
For each priority it reports whether the scenario improves, worsens or does not
change the measure relative to the usual diet, with the evidence rows for that
pathway; the scenario's label is the strongest among improving criteria. Sugar
reductions are graded MODERATE only when energy also falls (CHI2014-HYPER vs
CHI2014-ISO). Lower-energy meals cite calorie restriction (MODERATE), with
weight loss of 5% or more (STRONG) shown as conditional. No CAP or probability
is predicted for a scenario; the timeline for Layer 6 gives a direction per
horizon, labelled "Simulated projection, not a clinical forecast".

## Clinical safety, explanations and uncertainty (Proposal 5.3, 7.3, 17.2-17.4, 17.7)

`src/twin_pipeline.py` now adds to every patient's output:

- **Fibrosis risk** (`src/fibrosis.py`): FIB-4 for ages 18-59 with the EASL-EASD-EASO
  2024 cut-offs (1.3, 2.67); none under 18. A high score withholds recipes until a
  clinical assessment. Against measured stiffness in 7,586 adults with a reliable
  reading (IQR/median <= 30%), FIB-4 below 1.3 rules out >= 12 kPa with NPV 0.978,
  but AUC is only 0.59 (>= 8 kPa) and 0.67 (>= 12 kPa): a rule-out test
  (`data/results/fibrosis/`). A low score at ages 18-34 carries a note that FIB-4 is less
  reliable under 35 (AASLD 2023) and that a FibroScan would help (36% of adults). When a
  patient has a reliable FibroScan reading (optional inputs `LUXSMED`, `LUXSIQRM`), it
  replaces FIB-4: below 8 kPa low, 8-12 indeterminate, 12 or more high (recipes withheld).
- **Subtype and safety notices** (`src/liver_labels.py`): MASLD, MetALD, ALD or
  cryptogenic steatosis from the 2023 nomenclature (alcohol from ALQ121 x ALQ130 x
  14 g), and "discuss with a clinician" notices for under-18s, alcohol in the MetALD
  range or above, glucose-lowering medicine and pregnancy. In-scope adults: 51%
  MASLD, 3.1% MetALD, 1.3% ALD (measured CAP >= 248). The adolescent criteria are
  marked for verification against the nomenclature's figure.
- **Explanation of the predicted CAP** (`src/explain_patient.py`): per-patient SHAP
  in dB/m, the inputs that raise and lower the estimate.
- **Uncertainty** (`src/uncertainty.py`): a 90% conformal CAP interval of +/- 66.6
  dB/m from five-fold out-of-fold errors (test coverage 90.9% in scope), and bootstrap
  bands for the calibrated probabilities (mean width about 5 points).
- **Causal knowledge graph** (`src/knowledge_graph.py`, `data/evidence/causal_graph.csv`):
  16 links from diet and activity levers through weight and insulin resistance to
  liver fat, injury and fibrosis; CAP, ALT and stiffness are measurements only.
  Priorities may act only on levers with a path to liver fat.
- **Recipe explanations** (`src/recipe_scoring/recipe_explanation.py`): each
  criterion of the top three recipes traced through the graph to liver fat with its
  evidence rows, plus the cuisine and practicality context.
- **Plain-language summary** (`src/twin_summary.py`): written by a template from the
  twin's own facts; an automatic check confirms every number and citation appears
  in those facts. It gives the patient the chances of liver fat and a plain verdict,
  not the CAP estimate or its range, which stay in the twin's output for clinicians. A language model can be plugged in through `summary_generator`;
  if its text fails the check, the template is shown.

Recipe scoring (section 7.1): recipes labelled "Other" (71% of candidate meals)
are now scored on ingredient co-occurrence typicality for the requested cuisine
(`src/recipe_scoring/cuisine_typicality.py`), up to 50, between a correct label
(100) and a wrong one (0); labelled recipes keep their label score. Blending label
and ingredients for every recipe was tried and lowered the share of top-10
recipes in the patient's cuisine from 0.998 to 0.865. Feasibility adds demanding
techniques and special equipment read from the instructions.

## Backend: patient journey and clinician workflow

`src/api/app.py` (FastAPI, SQLite in `src/api/db.py`) serves the journey:

1. **Sign-in** without passwords (`src/api/auth.py`): a one-time code, emailed when SMTP is set in `.env`
   (`src/api/mailer.py`; reserved test domains such as example.org are never emailed and get the code in the
   response), then a bearer token. Patients see only their own record; clinicians see their clinic's
   patients. Patients join a clinic with its code.
2. **Sign-up** (`POST /patients`): measurements, cuisine and favourite dishes picked from
   `GET /dishes?cuisine=`, and food answers (`GET /food/questions`, `src/recipe_scoring/food_preferences.py`):
   diet, the nine US major allergens plus gluten and any other allergy typed in, foods never eaten (pork,
   beef, alcohol), disliked and liked ingredients, cooking time and household size. All but likes and
   household are hard rules; a recipe must pass both RecipeDB's diet label and an ingredient check. The twin
   then runs with the cuisine, favourites and food answers.
3. **Suggested plan** (`GET /patients/{id}/plan/suggestion`, `src/care_plan.py`): three paces from the
   trial prescriptions (gentle: half the DPP calorie cut, 90 min walking; standard: DPP cut and 150 min;
   ambitious: Look AHEAD goal and 175 min) with the expected 12-month change from Hall's rule. Patients
   pick a pace; only clinicians set custom numbers. A plan waits for a clinician's approval
   (`POST .../plan/approve`) when a safety notice applies (under 18, alcohol, glucose-lowering medicine,
   pregnancy, scarring referral).
4. **Recipe-style ladder** (`PUT .../plan/style`, `src/recipe_scoring/style_ladder.py`): the patient
   chooses Scenario B or Scenario A; the twin then moves them between the two a step at a time after
   check-ins (two middle steps). Four steps from
   "closest to what you eat now" (Scenario B: nutrients against the 24-hour recall and ingredient overlap
   with the favourite dishes) to "healthiest" (Scenario A), all within the recipes meeting every gating
   reference. `GET .../week` builds the seven-day plan at the step; `GET .../week/shopping` adds up one
   serving of each meal.
5. **Check-ins** (`POST .../visits` with `followed`: meals most/half/few, walking as_planned/some/little)
   return what changed: the step verdict, progress against the plan from the start, its cause ("plan not
   followed", "no progress despite effort") and one tip (`src/api/care.py`). The re-assessment loop splits the response
   factor into adherence x body response (`src/reassessment.py`); without answers it is unchanged.
   "No progress despite effort" does not claim a slow body: the in-silico evaluation below showed the body
   response cannot be told apart from ordinary weight drift within a year.
6. **Today** (`GET .../today`): a status line, the one next step (awaiting approval, choose a plan,
   check-in due, add liver blood tests, or cook today's meals), today's two meals and the next check-in.
7. **Clinician** (`GET /patients`, `GET /alerts`): patients sorted by alerts (plan awaiting approval,
   scarring risk, no progress at two check-ins despite reported effort (with what to check: a 3-day food
   diary, ordinary weight drift, medicines or thyroid), check-in overdue, no plan, no scarring check).
   `GET /patients/{id}/clinical` gives the clinician's patient page in one call: the record, plans over time,
   alerts, the full twin result for the latest visit (CAP estimate, grade and 90% interval, SHAP drivers,
   scarring, label, notices, priorities, both scenarios, recipe explanations; the top 10 recipes instead of the
   whole ranking) and the timeline with CAP values.
8. **Saved runs and speed.** `GET /patients/{id}/twin` returns the saved run and reruns the twin only after a
   newer visit (a run ranks every recipe and takes several seconds). The liver-fat model, the chances for given
   inputs, the ingredient links and decoded runs are kept in memory, and the backend loads the recipes, model
   and links in the background when it starts (`HEPATOTWIN_NO_WARMUP=1` turns this off).
9. **Recipes for the pages** (`GET /recipes?ids=`): title, time, servings, nutrients per serving, ingredients,
   method, the original link and a photo. RecipeDB has a food.com photo link for each recipe that the
   collection did not keep; it is looked up the first time a recipe is shown and saved in
   `data/app/recipe_images.json` (`src/api/recipe_images.py`).

Limit found while building it: the 20,000-recipe sample has few main meals in some cuisines (Indian 13,
against European 1,152, Asian 637, Mexican 308), so a seven-day plan cannot be filled for those cuisines.

Demo accounts: `src/api/seed_demo.py` (a clinic, Dr. Rao and five held-out NHANES participants; `--reset`
starts the patients again from sign-up). `src/api/remove_test_accounts.py --yes` removes accounts made by the
browser tests.

## Dashboard (frontend)

`frontend/` is a React + TypeScript app (Vite) that talks to the backend through `/api`. Start both as in
[docs/RUN_AND_DEMO.md](docs/RUN_AND_DEMO.md) and open http://localhost:5173.

- **Patient:** sign-in with an emailed code; an eight-screen sign-up (measurements in NHANES units, cuisine,
  favourite dishes, food rules); **Today** (next step, body measures, goals, the predict-measure-adjust band,
  a My liver summary and today's meals with photos); **My liver** (a 3D liver beside the liver at 3, 6 or 12
  months on the plan, coloured by the chance of a lot of fat, then the full results); choosing a pace and a
  recipe style; **Meals** (the week and the shopping list, with a recipe panel); **My plan**; **Progress**
  (weight against the expected range and the 12-month projection, chances over time, check-ins); **Check-in**.
- **Clinician:** the patient list sorted by alerts, plan review and approval with a note, and the patient
  page (CAP estimate and grade, SHAP drivers, scarring, label and notices, priorities, Scenario A vs B,
  knowledge-graph explanations of the top recipes, plans and what the patient reads).
- Patients see chances of liver fat, never the CAP point estimate or grade (R² 0.56 is too weak for a single
  number); the clinician page shows both. Projections are labelled as simulations, not forecasts, and the 3D
  liver as a drawing, not a scan.
- Browser checks: `frontend/e2e/step3.mjs` (plan approval), `step4.mjs` (patient journey), `step5.mjs`
  (3D liver), `step6.mjs` (clinician page), run with both servers up (headless Edge over CDP).

## NHANES III (Proposal 17.9)

`src/nhanes3.py --download` fetches the NHANES III exam and laboratory files and
the 2009-2010 hepatic steatosis ultrasound re-reading into `data/raw/nhanes3/`,
maps 51 of the 81 model inputs (units converted; `data/results/nhanes3/variable_map.csv`)
and keeps 10,682 adults aged 20-59 with a gradable image (20.7% moderate-severe).
The public-use Linked Mortality File (`NHANES_III_MORT_2019_PUBLIC.dat`, downloaded by
hand into `data/raw/nhanes3/` because ftp.cdc.gov fails the certificate check here;
layout from NCHS's `SAS_ReadInProgramAllSurveys.sas`) links 10,672 of them: 2,174
deaths over a median 27.3 years. Its cause-of-death recode has no liver category, so
it supports all-cause, cardiovascular and cancer mortality, not liver-related deaths.

`src/nhanes3_validation.py` scores the model on them (`data/results/nhanes3/validation.json`;
unweighted; ultrasound grade, not CAP, 30 years earlier, so it tests ranking, not CAP values):

| Moderate-severe steatosis on ultrasound | n | AUC (95% CI) |
|---|---|---|
| Model: chance of CAP >= 280 | 10,682 | 0.771 (0.759-0.782) |
| Hepatic Steatosis Index | 10,055 | 0.725 (0.712-0.738) |
| Waist / BMI alone | 10,374 / 10,668 | 0.716 / 0.698 |
| With GGT: Fatty Liver Index vs model | 7,649 | 0.744 vs 0.766 |

By group: men 0.783, women 0.755, Mexican American 0.787, non-Hispanic White 0.790, non-Hispanic Black
0.690 (the lowest; the ultrasound share is also lowest, 16%). Mean chance 0.255 against a 20.7%
ultrasound share; the share rises in every tenth of the chance from the fourth up.

Mortality by quarter of the chance of CAP >= 280 (10,671 linked, 2,173 deaths, median 27.3 years;
rate ratios adjusted for age and sex, Poisson with person-years offset):

| Deaths | Highest vs lowest quarter | Ultrasound moderate-severe vs not |
|---|---|---|
| All-cause | 1.58 (1.38-1.81) | 1.29 (1.17-1.41) |
| Cardiovascular | 2.15 (1.64-2.81) | 1.35 (1.13-1.61) |
| Cancer | 1.00 (0.79-1.28) | 1.05 (0.88-1.26) |

Not adjusted for smoking, alcohol or income, which are not mapped. FIB-4 (9,967 with the three blood
tests; 92% low, 7% indeterminate, 1% high): high vs low had 3.23 (2.47-4.23) times the all-cause death
rate, 1.89 (1.01-3.54) cardiovascular and 2.63 (1.57-4.41) cancer; indeterminate vs low 1.25 (1.10-1.42)
all-cause. Liver-related deaths cannot be studied: the public file has no liver category.

## Fairness audit (Proposal 17.8)

`src/fairness_audit.py` (`data/results/fairness/`) scores the 1,236 in-scope 2017-2023 test participants
by sex, race and ethnicity and age group, with 95% bootstrap intervals. The chance of CAP >= 280 ranks
people similarly in every group (AUC 0.84-0.89; 0.93 at ages 12-17; all intervals overlap). Calibration
differs by ethnicity: the mean chance is too low for non-Hispanic Asian participants (0.25 against a 0.31
share) and too high for non-Hispanic Black participants (0.34 against 0.29). The ranking engine meets all
three recipe criteria for 80-84% of every group (300 patients; simulated cuisine preferences).

## In-silico evaluation of the follow-up loop (Proposal 17.14)

`src/in_silico_evaluation.py` runs the twin on the calibrated virtual cohort (1,236 people, 500 and
250 kcal/day plans, 12 months). The twin sees only what a clinic would record (monthly weights with missed
visits, quarterly bloods) and the monthly meals answer, and is graded against the simulator's truth
(`data/results/in_silico/summary.json`; adults, honest answers):

- Weight forecasts: mean error 1.9 kg (month 3 to 6) and 2.8 kg (month 6 to 12), against 2.9 and 4.7 kg
  for the textbook curve; 90% ranges cover 91%. Carrying the last weight forward is as accurate (1.9 and
  2.7 kg), because most simulated people ease off within months. Without the meals answer the month-12
  error rises to 3.0 kg.
- "Plan not followed": true adherence was below 0.4 every time.
- Body response: the learned response correlates 0.03-0.06 with the true one; 12 monthly weights cannot
  separate a slow body (true spread about 20%) from ordinary drift (1.3 kg per square-root month). The
  "slow response" cause and alert were therefore reworded as "no progress despite effort". 78% of alerted
  patients were still behind plan at month 12 (63% of all); behind the alerts were ordinary drift (74%)
  and a slow body (26%), and with over-reported answers also intake higher than reported (8%).
- Liver: the chance of CAP >= 280 keeps AUC 0.88 at months 0, 6 and 12 with mean chance close to the true
  share (0.31 vs 0.29 at month 12, 500 kcal). The month-6 forecast of month 12 beats reusing the month-0
  chance (AUC 0.875 vs 0.868, Brier 0.126 vs 0.133).

`src/in_silico_extras.py` (`data/results/in_silico/extras.json`) adds the other measures of section 12.3:

- Detecting a change: a patient truly behind the plan was flagged at the same check-in in 96.5% of cases
  and within a month in 99.4% (51 of 1,659 never, after missed visits), with no false flags. Liver fat
  falling below CAP 280 is harder: of 51 such patients (500 kcal), the twin's chance fell below 0.5 for 29
  (median a month before the truth) and also fell for 46 whose CAP stayed above 280; near the threshold
  the chance cannot place a person on one side.
- Learning: the 3-month-ahead weight error stays about 1.9 kg from month 1 to month 9 while the textbook
  curve's grows from 2.3 to 4.8 kg; 90% coverage stays 91-93%.
- Robustness (month 6 to 12): 2.77 kg with honest answers, 2.81 with 30% more missed visits, 2.91 with a
  scale reading within about 1 kg, 3.03 without answers; coverage 0.89-0.91. The month-12 liver AUC is the
  same without follow-up bloods (0.88), because the simulated change runs through weight and waist.
- Scenario order: the twin's sign-up projection always puts Scenario A below B below usual diet; the truth
  agrees for 95.6% of adults. At month 6 its forecasts order A and B as the truth for 95.6%; for liver fat
  at month 12 only 54%, because the true difference between A and B is small next to person-to-person
  variation.

## Plain-language summaries (Proposal 17.7)

`src/summary_evaluation.py` (`data/results/summaries/`) runs the twin for 300 in-scope test patients. All
300 summaries pass the faithfulness check, and planted errors are caught every time when a number is changed
or added or a citation invented, with no false alarm on a harmless rewording; a wrong word with no number
("low" to "high") is never caught, as the check reads only numbers and citations. After the team's review of
10 summaries (`summaries_for_review.md`, with the expected words worked out beside each) the patient text
shows no evidence codes ("backed by 4 studies" instead), explains liver scarring, gives the published cut-offs
with their source and says when a score is close to one, words the priorities plainly, shows diet notices only
when a diet change is suggested and the under-35 FIB-4 note only when liver fat is likely, and never shows 0%
or 100%. Median Flesch-Kincaid grade fell from 12.5 to 9.8 (Reading Ease 43 to 56.5).

## Reproducibility

`.venv/Scripts/python.exe -m src.rebuild_all` rebuilds data, models and results in order (`--list` shows
the 24 steps, `--from <stage>` resumes, `--with-recipes` also re-collects the RecipeDB sample); step times
go to `data/results/rebuild_log.json`. It repeats the simulator calibration for quit caps 0.1-0.4 and
chooses the cap by the Neiberg 2012 regain target, as was done by hand.

## Evidence table

`data/evidence/intervention_evidence.csv` has 23 rows. Section 17.13 added
coffee (Liu 2015; Ebadi 2021), fasting, and the null findings for
low-carbohydrate high-fat diets and omega-3 (Stern 2026). The coffee rows are
labelled OBSERVATIONAL ASSOCIATION; no rule cites them
(`data/evidence/evidence_strengths.md`).

RecipeDB coverage: the API we use reports 20,000 recipes in total (19 Indian),
the same 20,000 already in `data/processed/recipe_objects.json`. The patient data
are from the US population, so US cuisines are the focus; regional Indian
cuisines are future work.

`data/results/combined/` contains the regressor, both classifiers,
`metadata.json` (selection, versions, data hash, limitations),
`split_manifest.csv`, `cv_selection.csv`, test metrics by subgroup,
`calibration_curve.csv`, `test_predictions.csv` and gain importances.

Why R² stays near 0.5: on the `patient-model-experiments` branch, adding about
80 more NHANES variables, the 2021-2023 participants, DXA visceral fat,
CatBoost, MLP and stacking each moved R² by at most about 0.01. The learning
curve is flat. CAP's within-exam noise is about 4% of its variance, so the
remaining error reflects liver-fat variation that routine clinical measures do
not capture. Internal validation only; survey weights are not modelled.

## Run (earlier 14-input model)

From the project root on Windows, using the existing virtual environment:

```powershell
.venv/Scripts/python.exe src/train_final_model.py
.venv/Scripts/python.exe src/demo.py
.venv/Scripts/python.exe src/predict_cap.py --input patients.csv --output predictions.csv
.venv/Scripts/python.exe -m unittest src.test_cap_model -v
```

Training defaults to 40 randomized configurations, five-fold cross-validation,
and seed 42. `--iterations`, `--data`, and `--output` can be overridden. A smaller
search is a smoke run and can select a different model. Package versions used
for the finalized artifact are recorded in `data/results/final/metadata.json`;
use `requirements-model.txt` to reproduce its environment.

## Input contract

Supply raw NHANES-coded columns (no manual dummy encoding or imputation):

| Columns | Meaning |
| --- | --- |
| RIDAGEYR | Age, years |
| RIAGENDR | NHANES sex code, 1 or 2 |
| RIDRETH3 | NHANES race/ethnicity code, 1, 2, 3, 4, 6, or 7 |
| BMXBMI, BMXWAIST | BMI (kg/m²), waist (cm) |
| LBXSATSI, LBXSASSI | ALT, AST (U/L) |
| LBXGH | HbA1c (%) |
| LBXTC, LBDHDD | Total cholesterol, HDL (mg/dL) |
| DR1TKCAL | Day-one energy (kcal) |
| DR1TPROT, DR1TCARB, DR1TTFAT | Day-one protein, carbohydrate, fat (g) |

All columns must exist; blank values are imputed with training-fitted medians
or categorical modes. Unknown category codes and infinite values are rejected.
Extra columns are ignored; the batch CLI preserves `SEQN` when present.
Input units and the intended population must match training. Load only trusted
joblib artifacts. Prediction does not need access to the training CSV.

## Evaluation and artifacts

Finalized run (40 configurations, 200 CV fits):

| Model | Test MAE (dB/m) | Test RMSE (dB/m) | Test R² |
| --- | ---: | ---: | ---: |
| Historical saved baseline | 35.344 | 44.520 | 0.491 |
| Baseline retrained with fold-fitted preprocessing | 35.260 | 44.818 | 0.485 |
| Final tuned XGBoost | **34.713** | **43.866** | **0.506** |

Training CV RMSE improved from 45.887 to 45.029. The final model uses 350 trees,
depth 4, learning rate 0.015, minimum child weight 10, row subsampling 0.7,
column subsampling 0.85, L1 penalty 0, and L2 penalty 1. Test RMSE decreased
1.47% from the historical result, or 2.13% from the corrected baseline.
The paired 2,000-resample bootstrap interval for RMSE improvement over the
corrected baseline is 0.523–1.381 dB/m (95%, conditional on fitted models).
Historical results also differ in preprocessing and potentially runtime version;
the corrected baseline is the controlled comparison for tuning.

Training reads the **unimputed** merged data, retains complete elastography exams
(`LUAXSTAT == 1`) with observed CAP, and checks participant IDs are unique.
The same ordered cohort and 80/20 split as the historical baseline are used.
Preprocessing is fitted separately inside each training fold, following the
[scikit-learn guidance on avoiding data leakage](https://scikit-learn.org/1.5/common_pitfalls.html).
Selection compares the baseline settings with the best randomized-search result
using mean training CV RMSE; test scores do not choose the winner.

The final artifact stays fitted on the 7,216 development participants, preserving
the 1,805-person holdout. It is not silently refitted on the full cohort.

`data/results/final/` contains:

- `patient_state_model.joblib`: fitted preprocessing and selected regressor.
- `metadata.json`: parameters, versions, data hash, metrics, and limitations.
- `cv_search.csv`: all search configurations and fold scores.
- `split_manifest.csv`: participant membership and validation-fold assignments.
- `model_comparison.csv` and `test_predictions.csv`: holdout comparison and predictions.
- `subgroup_metrics.csv`: descriptive errors by sex and race/ethnicity.
- `feature_importance.csv`: model gain importance, not causal effects.

Historical baseline artifacts remain in `data/results/`. The older
`baseline_xgboost.py`, `compare_models.py`, and `preprocess.py` reproduce the old
workflow, which imputes before splitting and selects using test results. They
are not the finalized training path and do not overwrite the new final folder.

## Limitations of the earlier 14-input model

This is an initial research model, not a clinically validated diagnostic model.
The original test set was already examined in earlier development, so this is
an internal comparison rather than a new independent validation. Search CV
scores are also selection-biased. External validation is still needed.
NHANES survey weights and household/PSU clustering are not modeled; estimates
describe this analytic sample rather than a survey-weighted US population.
The cohort includes ages 12–80. Subgroup results are descriptive and do not
establish fairness. Bootstrap uncertainty conditions on the fitted models.
Cross-sectional CAP prediction cannot establish effects of diet changes.

Validation: the model unit tests and demo pass, and saved/reloaded model
predictions match. `pytest src` runs every unit test. The three manual RecipeDB
scripts that once broke test collection are now `src/recipes/check_live_*.py`:
they call the live API (RECIPEDB_BASE_URL in `.env`), print parsed recipes
rather than assert, and are run from `src/recipes`.

## Second-pass experiments

The second pass selected an equal blend of expanded XGBoost with missingness
indicators and expanded XGBoost with 700 trees:

| Model | Inputs | Test MAE | Test RMSE | Test R² |
| --- | ---: | ---: | ---: | ---: |
| First-pass final | 14 | 34.713 | 43.866 | 0.5062 |
| Second-pass blend | 28 | 34.308 | 43.511 | 0.5142 |

The original-input winner is unchanged. The extra-input improvement is modest:
0.354 dB/m in RMSE. Its conditional paired-bootstrap 95% interval is
−0.001 to 0.724 dB/m, which includes zero, so this run does not establish a
clear generalization improvement over the first-pass model. Keep the expanded
artifact as a candidate for independent validation, with the 14-input model
remaining the default. See `uncertainty.json` for the calculation details.
All 1,805 expanded-model predictions reproduce through the public inference API,
and the three model unit tests pass.

Run `.venv/Scripts/python.exe -m src.improve_cap_model` from the project root.
This compares 24 candidates across the same five development folds: XGBoost
variants, missingness indicators, histogram gradient boosting, and additive
spline regression. A fixed equal blend of the two strongest candidates is also
compared using out-of-fold predictions. Selection uses pooled out-of-fold RMSE
(slightly different from averaging fold RMSE in the first pass), and is written
to `data/results/second_pass/selection.json` before holdout evaluation.

Three feature sets are evaluated: original (14 inputs), extended (19 inputs),
and expanded (28 inputs). Extended adds height (`BMXHT`, cm), weight (`BMXWT`, kg),
GGT (`LBXSGTSI`, U/L), uric acid (`LBXSUA`, mg/dL), and reference-assay triglycerides
(`LBXTR`, mg/dL). Expanded also adds albumin (`LBXSAL`, g/dL), ALP (`LBXSAPSI`, U/L),
creatinine (`LBXSCR`, mg/dL), bilirubin (`LBXSTB`, mg/dL), total protein (`LBXSTP`,
g/dL), globulin (`LBXSGB`, g/dL), iron (`LBXSIR`, ug/dL), blood urea nitrogen
(`LBXSBU`, mg/dL), and LDH (`LBXSLDSI`, U/L).

The [NHANES biochemistry documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BIOPRO.htm)
distinguishes reference assays from the standard biochemistry measurements;
secondary triglycerides/glucose and duplicate unit conversions are excluded.
No elastography measurements or examination-quality fields are predictors.
Triglycerides are missing for over half of this cohort because of subsampling;
training missingness is saved with the results. Missingness-based prediction
may not transfer to a different measurement protocol.

The expanded artifact is saved separately; the demo's original 14-input model
remains available. Use the expanded model explicitly with the existing CLI:

```powershell
.venv/Scripts/python.exe src/predict_cap.py --model data/results/second_pass/patient_state_model.joblib --input patients.csv --output predictions.csv
```

Its exact required columns are in `selection.json`. Do not substitute other
assay types or omit additional columns to simulate the original-input model.
`original_input_model.joblib` stores the best original-input candidate.
`cv_comparison.csv`, `oof_predictions.csv`, `model_comparison.csv`, and
`test_predictions.csv` retain the experiment evidence. The prior results stay
in `data/results/final/`. Additional search does not make the reused holdout an
independent validation set.

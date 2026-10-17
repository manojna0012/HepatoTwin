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

## Combined 2017-2023 model

Pools both NHANES FibroScan cycles: 2017-March 2020 (9,021 eligible) and
August 2021-August 2023 (6,279), 15,300 participants aged 12-80, nobody excluded
by age. 2021-2023 is the last NHANES cycle with liver elastography. Inputs are
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
down-weighted rather than dropped. The selected run down-weights them (CV R²
0.520). Test results (3,060 participants, scored once), against the 36-input
reference trained on the same rows:

| Test group | n | Combined R² | 36-input R² |
| --- | ---: | ---: | ---: |
| All | 3,060 | **0.493** | 0.487 |
| Reliable exams (IQR < 40) | 1,891 | 0.532 | 0.522 |
| Age 12-17 | 400 | 0.376 | 0.366 |
| Age 18-39 | 868 | 0.456 | 0.461 |
| Age 40-59 | 772 | 0.490 | 0.486 |
| Age 60+ | 1,020 | 0.418 | 0.400 |

This held-out split is harder than average: CV R² on the training rows is
0.520. Within-age R² is lower than overall partly because CAP varies less
inside an age group; 60+ RMSE is about 46 dB/m against 42-44 dB/m for adults
under 60.

The calibrated classifiers (`steatosis_cap248.joblib`, `steatosis_cap280.joblib`)
give P(CAP >= 248) and P(CAP >= 280) with isotonic calibration fitted on
training out-of-fold predictions. The cutoffs are provisional.

| Target | ROC-AUC | Brier | Calibration error | F1 | AUC, age 60+ |
| --- | ---: | ---: | ---: | ---: | ---: |
| CAP >= 248 | 0.857 | 0.154 | 0.023 | 0.792 | 0.823 |
| CAP >= 280 | 0.868 | 0.137 | 0.020 | 0.704 | 0.844 |

`src/validate_combined_model.py` adds checks without changing the model; results
go to `data/results/combined/validation/`. 95% intervals come from 1,000 bootstrap
resamples of the test set:

| Check | R² | AUC, CAP >= 248 | AUC, CAP >= 280 | Calibration error |
| --- | ---: | ---: | ---: | ---: |
| Test set, with intervals | 0.493 (0.461-0.521) | 0.857 (0.844-0.869) | 0.868 (0.855-0.881) | 0.023 / 0.020 |
| Test set, survey-weighted | 0.507 | 0.863 | 0.875 | 0.027 / 0.022 |
| Train 2017-2020, test 2021-2023 | 0.502 | 0.858 | 0.870 | 0.023 / 0.019 |

Survey weights are WTMECPRP and WTMEC2YR scaled by cycle length (3.2 and 2
years); strata and PSUs are not used. A CAP >= 302 classifier reaches AUC 0.872.
Separate calibration per age group did not improve calibration, so the global
calibration stays. The 60+ calibration error (0.037/0.050 on 1,020 people) is
0.030/0.029 on the larger temporal test, so it is mostly small-sample noise.

`src/thesis_figures.py` writes ROC, calibration, SHAP and learning-curve figures
to `data/results/combined/figures/`, and benchmarks the classifiers against the
Fatty Liver Index and Hepatic Steatosis Index on the same 2,703 test participants:
AUC 0.860 against 0.842 (FLI) and 0.813 (HSI) for CAP >= 248; every paired
interval excludes zero.

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
with priorities (cuisine preferences simulated; NHANES has none). Share of each
top 10 meeting the criteria (`data/results/layer4/`):

| Method | Health | Cuisine | Feasibility | All three |
| --- | ---: | ---: | ---: | ---: |
| Multi-criteria, health-gated (default) | 1.000 | 0.998 | 0.832 | **0.830** |
| Same, without acceptability | 1.000 | 0.993 | 0.802 | 0.795 |
| Same, with diversity re-ranking (MMR) | 1.000 | 0.989 | 0.819 | 0.808 |
| Multi-criteria, weighted only | 0.736 | 1.000 | 0.971 | 0.727 |
| Health score only | 1.000 | 0.057 | 0.400 | 0.017 |
| Nutrition only (non-personalised) | 1.000 | 0.057 | 0.400 | 0.017 |
| Random | 0.057 | 0.053 | 0.596 | 0.003 |

Only the gated ranking always meets every health reference. Across a grid of 36
weight combinations (acceptability held at 0.1) the median top-10 overlap with
the default is 0.78; it drops mainly when cuisine has weight 0.1. Diversity
re-ranking raised intra-list diversity only from 0.947 to 0.956 (mean pairwise
1 - Jaccard of ingredient sets without seasonings), so it is not the default. Health-only and nutrition-only
coincide because 565 candidates meet every reference value, so both choose among
tied recipes.

Personalisation (section 12.4; `data/results/layer4_before_diet/` is the same
engine without the recall):

| | Without recall | With recall |
| --- | ---: | ---: |
| Distinct priority profiles | 11 | 35 |
| Most common profile (share of patients) | 0.34 | 0.15 |
| Distinct top-10 lists | 16 | 29 |
| Mean top-10 overlap, same cuisine | 0.95 | 0.92 |

Acceptability makes lists a little more alike, because well-rated recipes are
shared. `priority_response.csv` checks the direction: with a high saturated-fat
priority the top 10 averages 3.20% of energy from saturated fat against 3.66%
with a low one, and 4.17% against 4.62% for sugars. Fibre does not differ, because almost
every recommended recipe already exceeds the fibre margin's cap. Before the margin
was added, priority levels did not affect the top 10 at all (8 distinct lists,
overlap 0.99), and 0.899 of top-10 recipes met all three criteria;
`MARGIN_SHARE` 1.0 gave 40 lists but 0.69 (both without acceptability).

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

On 300 participants every plan was complete and met every check: 58% at step 1,
81% by step 2, 100% by step 3, with 0.19 borrowed meals per plan on average
(only Canadian plans borrow, about one meal). 82% of plan meals are within the
practical limits (`meal_plan_summary.csv`, `meal_plan_by_cuisine.csv`).

```powershell
.venv/Scripts/python.exe src/demo.py --seqn 110006 --cuisine American --plan
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

## Scope

This is an initial research model, not a clinically validated diagnostic model.
The original test set was already examined in earlier development, so this is
an internal comparison rather than a new independent validation. Search CV
scores are also selection-biased. External validation is still needed.
NHANES survey weights and household/PSU clustering are not modeled; estimates
describe this analytic sample rather than a survey-weighted US population.
The cohort includes ages 12–80. Subgroup results are descriptive and do not
establish fairness. Bootstrap uncertainty conditions on the fitted models.
Cross-sectional CAP prediction cannot establish effects of diet changes.
The demo's evidence and recipe outputs remain explicitly labeled placeholders.

Validation: the model unit tests and demo pass, and saved/reloaded model
predictions match. Full `unittest discover -s src` encounters three pre-existing
recipe-test import errors (`recipedb_client` cannot be imported); those scripts
are outside the predictive-model changes.

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

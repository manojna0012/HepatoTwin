# Review of the supplied model ideas

The supplied training script reproduced R² **0.534759**, RMSE **43.205** and
MAE **33.466 dB/m** using the existing data and Python environment. Its CAP-quintile
stratified split differs from the unstratified seed-42 split used for the earlier
project models. Both splits have 7,216 training and 1,805 test participants.

## Separating the split from the model

Each entry below uses a model fitted on that split's training records. The
cross-split checks are diagnostic, not another model-selection exercise.

| Model | Original split R² | Supplied stratified split R² |
| --- | ---: | ---: |
| Previous expanded model | 0.514164 | 0.540836 |
| Supplied model, exact settings and feature code | 0.511310 | 0.534759 |

The reported 0.534 is reproducible. It does not show that the supplied model
outperforms the previous expanded model: on each matched split, the previous
model has the higher point estimate. A seed alone does not identify a test set;
stratification and participant membership also matter. Neither split is now a
fresh external validation set.

## Ideas incorporated

- Additional biochemistry inputs and ten deterministic derived features:
  waist/height, enzyme and lipid ratios, index-like combinations, energy
  fractions, and energy per kg.
- Feature engineering inside the persisted pipeline, before fold-fitted
  imputation, so raw patient input is sufficient for inference.
- Explicit handling of zero denominators, nonpositive log inputs, and stable
  logistic evaluation. Undefined derived values become missing and are imputed
  by the fitted training imputer.
- Held-out participant selection in the demo, with `--seqn`, `--random`, and
  `--model`. Training participants are rejected using the model's split manifest.
- Descriptive CAP-decile errors and subgroup metrics.

The original files are preserved unchanged in `references/supplied_model/` for
reference. Their CLI defaults belong to the supplied workflow; use the explicit
isolated output directory shown below when reproducing them.

## Controlled feature and parameter comparison

The experiment kept the original holdout fixed and selected on pooled
out-of-fold RMSE across the same five development folds. Existing OOF records
were aligned by participant ID. Five new candidates were compared with the
previous expanded model.

| Candidate | Development OOF RMSE |
| --- | ---: |
| Previous expanded model | 44.455 |
| Supplied settings, raw labs without ratios | 44.662 |
| Supplied settings with ratios | 44.584 |
| Ratios, 600 trees | 44.386 |
| Ratios, 800 trees with stronger regularization | 44.434 |
| **Ratios, 600 trees and missing-value indicators** | **44.276** |

The selected model retains depth 4, learning rate 0.02, minimum child weight 8,
subsample 0.8, column subsample 0.6, L1 penalty 0.5 and L2 penalty 3. It uses
600 trees, 36 raw input columns and ten internally generated features, plus
training-fitted missing-value indicators.

On the original holdout it achieves **R² 0.515061, RMSE 43.471, MAE 34.317**.
Compared with the previous expanded model, RMSE improves by only 0.040 dB/m,
while MAE is slightly worse (34.308 previously). The conditional paired-bootstrap
95% interval for RMSE improvement is **−0.305 to 0.380 dB/m**, including zero.
The feature ideas help development CV, but the holdout does not establish a
clear improvement over the previous expanded model. The original 14-input model
remains the default, and this expanded candidate is available explicitly.

## Interpretation corrections

`HSI_partial` is deliberately not labeled full HSI: the supplied expression
omits the diabetes term in the [published formula](https://www.sciencedirect.com/science/article/pii/S1590865809003363).
`FLI_proxy` uses the supplied index-like formula with biochemistry triglycerides.
Both are internal prediction features, not validated clinical scores here.

The [NHANES biochemistry documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/P_BIOPRO.htm)
generally recommends the reference triglyceride assay and cautions against
using `LBXSGL` to identify undiagnosed diabetes or prediabetes. `LBXSTR` and
`LBXSGL` were evaluated here as explicitly assay-specific research predictors
with much less missingness, not interchangeable clinical measurements. This
choice needs validation for any intended deployment measurement protocol.

The supplied script's threshold AUROCs compare predicted CAP with measured-CAP
threshold labels. They do not independently validate histological steatosis
grades. Its claim that differences below twice the fold standard deviation are
necessarily noise is not a valid general significance test. Likewise, target
censoring does not prove that observed-target prediction error cannot improve.
Those interpretations were not adopted. The new model evaluates unmodified
predictions consistently; it does not clip only the test report.

## Reproduction and validation

From the project root:

```powershell
# Reproduce the supplied result in an isolated folder (skips CV only).
.venv/Scripts/python.exe references/supplied_model/train_final_model.py --no-cv --outdir data/results/supplied_reference

# Select the adapted candidate on the original training folds.
.venv/Scripts/python.exe -m src.train_engineered_model

# Demonstrate or predict with raw input; no manual derived columns needed.
.venv/Scripts/python.exe src/demo.py --model data/results/engineered/patient_state_model.joblib
.venv/Scripts/python.exe src/predict_cap.py --model data/results/engineered/patient_state_model.joblib --input patients.csv --output predictions.csv
.venv/Scripts/python.exe -m unittest src.test_cap_model -v
```

Training uses the prior split and OOF artifacts in `data/results/second_pass/`.
`selection.json`, `cv_comparison.csv`, `model_comparison.csv`, prediction CSVs,
`uncertainty.json`, and diagnostics are in `data/results/engineered/`. The exact
36-column raw input list is in `selection.json`; preserve the recorded assay
types and NHANES units. Keep this project's `src` package importable when loading
the artifact, because it contains the serialized feature transformer class.

Five model tests pass, including ratio edge cases, serialization and held-out
demo selection. The real saved artifact and both CLI entry points were also
checked. All previous internal-validation and survey-design limitations remain.

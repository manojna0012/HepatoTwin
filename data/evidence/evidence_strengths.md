# Evidence Strength Definitions

The evidence-strength labels used by HepatoTwin are an internal
classification scheme for organizing intervention evidence. They are
not intended to replace established clinical evidence-grading systems.

## STRONG

Biopsy-paired or large randomized controlled trial evidence with a
consistent intervention effect and, where applicable, evidence of a
dose-response relationship.

Examples in the current evidence table include the biopsy-paired
weight-loss study by Vilar-Gomez et al. (2015).

## MODERATE

Evidence supported by a systematic review or meta-analysis of
randomized controlled trials, or by a well-powered controlled
intervention study, where the evidence is relevant but may have
heterogeneity or other limitations.

## WEAK / INCONCLUSIVE

Evidence exists, but the reported intervention effect is not
statistically significant, confidence intervals cross the null
effect, or the available evidence is otherwise insufficiently
consistent to support a clear beneficial effect.

A WEAK / INCONCLUSIVE entry is retained in the evidence table so that
conflicting or negative evidence is not silently discarded.

## INSUFFICIENT

There is no adequate evidence for the specific intervention-outcome
relationship.

The eventual ranking engine should not assign a quantitative health
benefit or penalty when the evidence is INSUFFICIENT. The intervention
or recipe should instead be flagged as having insufficient evidence.

## GUIDELINE RECOMMENDATION

Guideline recommendations are recorded separately from the STRONG /
MODERATE / WEAK / INSUFFICIENT evidence labels.

A guideline may make a strong clinical recommendation while drawing
on an evidence base that includes multiple study designs. Therefore,
"strong recommendation" in a clinical guideline should not
automatically be converted into the project's STRONG evidence label.

## Evidence interpretation principles

1. Evidence is recorded at the intervention and outcome level.
2. A finding from one study should not automatically be generalized
   to every population or every recipe.
3. Meta-analyses are interpreted together with their heterogeneity,
   study populations, intervention differences, and outcome
   measurement methods.
4. Observational associations should not automatically be treated as
   causal intervention effects.
5. Guideline recommendations provide clinical context but are not
   themselves individual intervention trials.
6. An intervention effect should not be extrapolated directly to an
   individual recipe without an explicit mapping rule.
7. Conflicting evidence is retained rather than resolved by
   selecting only the favourable result.
8. INSUFFICIENT evidence should not be converted into an invented
   numerical score.
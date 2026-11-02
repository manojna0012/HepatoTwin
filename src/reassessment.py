"""Re-assessment loop: predict, observe, compare, adjust (Proposal section 17.5).

At each visit the twin first predicts the patient's weight from what it has learned so far, then
compares the prediction with the weight measured, and finally updates what it has learned.

- Predict. The twin's forecast is Hall's published rule of thumb (WT-EQ, WT-HALF, WT-95): a
  sustained change in intake moves weight by 1 kg per 100 kJ/day in the end, half of it in about a
  year. The plan's full effect is scaled by a personal response factor k, which absorbs how
  closely the patient follows the plan and how strongly their body responds (k = 1: full plan,
  full textbook response; k = 0: no change).
- Compare. A measured weight outside the 90% range of the prediction is flagged as slower or
  faster than expected. The range combines the uncertainty in k with ordinary weight drift, set
  from the spread of weight change in Look AHEAD usual care (5.0 kg SD over 12 months,
  LA-Y1-KG-SD), assumed to grow like a random walk.
- Adjust. k is re-estimated by Bayesian linear regression on the weight changes between visits,
  from a weakly informative prior (mean 0.5, SD 0.35: on average half the plan's textbook effect).
  Older changes count less (weight halves every RECENCY_HALF_LIFE months), because adherence
  drifts over time.

The twin does not use the simulator's code or its calibrated settings, only the published rule
and trial spread, so the in-silico evaluation tests it fairly. Projections for the liver re-run
Layer 2 with the projected weight, BMI and waist (WC-KG, Lim 2011).
"""
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ROOT / 'data' / 'evidence' / 'simulator_parameters.csv'
TARGETS = ROOT / 'data' / 'evidence' / 'calibration_targets.csv'
KJ_PER_KCAL = 4.184
DAYS_PER_MONTH = 30.4375
PRIOR_MEAN, PRIOR_SD = 0.5, 0.35
RECENCY_HALF_LIFE = 6.0     # months
Z90 = 1.645
HORIZONS = (3, 6, 12)


def _single_exponential(half_months, ninety_five_months):
    """Time constant of the single exponential closest to Hall's 'half in a year, 95% in three'."""
    taus = np.linspace(1, 60, 5901)
    loss = ((1 - np.exp(-half_months / taus) - 0.5) ** 2 + (1 - np.exp(-ninety_five_months / taus) - 0.95) ** 2)
    return float(taus[np.argmin(loss)])


@dataclass(frozen=True)
class WeightRule:
    kg_per_100kj: float
    tau_months: float
    drift_sd: float            # kg per sqrt(month)
    waist_cm_per_kg: float

    @classmethod
    def from_tables(cls, parameters=PARAMETERS, targets=TARGETS):
        p = pd.read_csv(parameters).set_index('param_id')['value']
        t = pd.read_csv(targets).set_index('target_id')
        usual_care_sd = 5.0
        assert f'{usual_care_sd} (DSE)' in t.loc['LA-Y1-KG-SD', 'spread']
        return cls(kg_per_100kj=float(p['WT-EQ']),
                   tau_months=_single_exponential(float(p['WT-HALF']), float(p['WT-95'])),
                   drift_sd=float(usual_care_sd / np.sqrt(12)), waist_cm_per_kg=float(p['WC-KG']))

    def full_effect(self, energy_change_kcal, months):
        """Weight change (kg) after `months` of following the plan fully, with a textbook response."""
        eventual = self.kg_per_100kj * energy_change_kcal * KJ_PER_KCAL / 100
        return eventual * (1 - np.exp(-np.asarray(months, dtype=float) / self.tau_months))


def months_between(start, end):
    return (date.fromisoformat(str(end)) - date.fromisoformat(str(start))).days / DAYS_PER_MONTH


@dataclass
class Response:
    """Posterior belief about the personal response factor k."""
    mean: float = PRIOR_MEAN
    sd: float = PRIOR_SD


def learn(steps, rule):
    """Posterior for k from weight changes between visits. steps: (plan_effect_kg, observed_kg,
    months_between, months_ago) per step; recent steps count more."""
    precision = 1 / PRIOR_SD ** 2
    weighted = PRIOR_MEAN / PRIOR_SD ** 2
    for effect, observed, gap, ago in steps:
        if gap <= 0:
            continue
        w = 0.5 ** (ago / RECENCY_HALF_LIFE) / (rule.drift_sd ** 2 * gap)
        precision += w * effect ** 2
        weighted += w * effect * observed
    return Response(mean=float(weighted / precision), sd=float(np.sqrt(1 / precision)))


def predict(last_weight, k, effect, gap, rule):
    """Expected weight after a step and its 90% range."""
    mean = float(last_weight + k.mean * effect)
    sd = float(np.sqrt((effect * k.sd) ** 2 + rule.drift_sd ** 2 * max(gap, 0)))
    return mean, mean - Z90 * sd, mean + Z90 * sd


def reassess(weights, plan_start, energy_change_kcal, rule=None):
    """Run the loop over dated weights [(visit_date, kg or None), ...] in date order.

    Returns one record per visit with a weight (the first is the starting point), what the twin
    expected before seeing it, the verdict, and k after learning from it; plus projections from the
    last weight at 3, 6 and 12 months.
    """
    rule = WeightRule.from_tables() if rule is None else rule
    measured = [(d, w) for d, w in weights if w is not None]
    if not measured:
        return {'visits': [], 'response': Response().__dict__, 'projection': []}
    t = [months_between(plan_start, d) for d, _ in measured]
    plan = [float(rule.full_effect(energy_change_kcal, max(x, 0))) for x in t]
    records = [{'visit_date': str(measured[0][0]), 'weight_kg': measured[0][1], 'expected_kg': None,
                'expected_range_kg': None, 'verdict': 'start', 'k_mean': PRIOR_MEAN, 'k_sd': PRIOR_SD}]
    steps = []
    k = Response()
    for i in range(1, len(measured)):
        effect, gap = plan[i] - plan[i - 1], t[i] - t[i - 1]
        mean, low, high = predict(measured[i - 1][1], k, effect, gap, rule)
        observed = measured[i][1]
        verdict = ('slower than expected' if observed > high else 'faster than expected' if observed < low
                   else 'on track')
        steps.append((effect, observed - measured[i - 1][1], gap, t[i]))
        k = learn([(e, o, g, t[i] - at) for e, o, g, at in steps], rule)
        records.append({'visit_date': str(measured[i][0]), 'weight_kg': observed, 'expected_kg': round(mean, 2),
                        'expected_range_kg': [round(low, 2), round(high, 2)], 'verdict': verdict,
                        'k_mean': round(k.mean, 3), 'k_sd': round(k.sd, 3)})
    last_t, last_w = t[-1], measured[-1][1]
    projection = []
    for h in HORIZONS:
        effect = float(rule.full_effect(energy_change_kcal, max(last_t + h, 0)) - plan[-1])
        mean, low, high = predict(last_w, k, effect, h, rule)
        projection.append({'months_ahead': h, 'weight_kg': round(mean, 2), 'range_kg': [round(low, 2), round(high, 2)],
                           'change_kg': round(mean - last_w, 2)})
    return {'visits': records, 'response': {'mean': round(k.mean, 3), 'sd': round(k.sd, 3)},
            'projection': projection}


def projected_inputs(inputs, change_kg, rule):
    """Twin inputs after a weight change: weight, BMI (same height) and waist (WC-KG) move together."""
    out = dict(inputs)
    weight = inputs.get('BMXWT')
    if weight is None or not np.isfinite(weight):
        return out
    out['BMXWT'] = weight + change_kg
    height = inputs.get('BMXHT')
    if height is not None and np.isfinite(height) and height > 0:
        out['BMXBMI'] = out['BMXWT'] / (height / 100) ** 2
    waist = inputs.get('BMXWAIST')
    if waist is not None and np.isfinite(waist):
        out['BMXWAIST'] = waist + rule.waist_cm_per_kg * change_kg
    return out

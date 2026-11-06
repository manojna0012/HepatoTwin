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

What the patient followed. k is the product of adherence a (the share of the plan followed) and the
body's response r: k = a x r. The prior above is r ~ 1.0 +/- 0.7 (the textbook response) at a typical
adherence of 0.5. When the patient reports at a check-in how much of the plan they followed
(src/care_plan.py), that step uses the reported adherence and r is learned from it; without a report
the step uses 0.5, which gives exactly the k of the version without reports. This separates the two
reasons for slow progress: a weight above the range with low reported adherence means the plan was not
followed ("plan not followed"); with good adherence it means the body responds slowly ("slow
response"). Reported adherence is self-reported and likely optimistic, as food logs are.

Progress against the plan. The step verdict compares each weight with what the twin expected after
learning, so once it has learned that a patient is not losing weight it stops flagging them. Progress
is therefore also judged from the start: the weight change since the plan began against the plan's
textbook effect at the adherence reported so far (prior response, no learning). From month
BEHIND_FROM_MONTHS on, a patient is "behind plan" when they have lost less than BEHIND_SHARE (half) of
the loss the plan typically gives (typical adherence and response: the result shown when the plan was
chosen), the kind of early-response check weight programmes use; the reported adherence then gives the
cause; before then, and for "ahead of plan", the 90% range is used. The 90% range alone is too wide for
this: it includes no loss at all, so a patient whose weight stalled for months while following the plan
was still "on plan". The cause of slow progress and the alerts use this progress check.

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
TYPICAL_ADHERENCE = 0.5
RESPONSE_MEAN, RESPONSE_SD = 1.0, 0.7          # body response r; textbook = 1
PRIOR_MEAN, PRIOR_SD = TYPICAL_ADHERENCE * RESPONSE_MEAN, TYPICAL_ADHERENCE * RESPONSE_SD   # k = a x r
LOW_ADHERENCE, GOOD_ADHERENCE = 0.4, 0.6
RECENCY_HALF_LIFE = 6.0     # months
Z90 = 1.645
BEHIND_FROM_MONTHS = 1.5    # the half-of-expected check starts at the second monthly check-in
BEHIND_SHARE = 0.5          # behind plan: lost less than half of the loss the plan typically gives
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
    """Posterior belief about the body's response r (k = adherence x r)."""
    mean: float = RESPONSE_MEAN
    sd: float = RESPONSE_SD

    def at(self, adherence):
        """The response factor k at a given adherence."""
        return Response(mean=adherence * self.mean, sd=adherence * self.sd)


def learn(steps, rule):
    """Posterior for r from weight changes between visits. steps: (plan_effect_kg, observed_kg,
    months_between, months_ago[, adherence]) per step; recent steps count more."""
    precision = 1 / RESPONSE_SD ** 2
    weighted = RESPONSE_MEAN / RESPONSE_SD ** 2
    for effect, observed, gap, ago, *rest in steps:
        if gap <= 0:
            continue
        x = effect * (rest[0] if rest and rest[0] is not None else TYPICAL_ADHERENCE)
        w = 0.5 ** (ago / RECENCY_HALF_LIFE) / (rule.drift_sd ** 2 * gap)
        precision += w * x ** 2
        weighted += w * x * observed
    return Response(mean=float(weighted / precision), sd=float(np.sqrt(1 / precision)))


def predict(last_weight, k, effect, gap, rule):
    """Expected weight after a step and its 90% range."""
    mean = float(last_weight + k.mean * effect)
    sd = float(np.sqrt((effect * k.sd) ** 2 + rule.drift_sd ** 2 * max(gap, 0)))
    return mean, mean - Z90 * sd, mean + Z90 * sd


def is_behind(start, observed, typical, high, months):
    """Behind the plan: above the 90% range, or from BEHIND_FROM_MONTHS on, less than BEHIND_SHARE of the
    typical loss (only when the plan expects a loss)."""
    if observed > high:
        return True
    expected_loss = start - typical
    return months >= BEHIND_FROM_MONTHS and expected_loss > 0 and start - observed < BEHIND_SHARE * expected_loss


def cause(progress, adherence):
    """Why progress is behind the plan, when the patient reported what they followed."""
    if adherence is None or progress != 'behind plan':
        return None
    if adherence < LOW_ADHERENCE:
        return 'plan not followed'
    if adherence >= GOOD_ADHERENCE:
        return 'slow response'
    return 'partly followed'


def plan_schedule(plan_start, energy_change_kcal):
    """[(months after plan_start, kcal/day from then on), ...] from one energy change, or from a list of
    (start date, kcal/day) when the plan was changed along the way."""
    if np.isscalar(energy_change_kcal):
        return [(0.0, float(energy_change_kcal))]
    return [(months_between(plan_start, d), float(e)) for d, e in sorted(energy_change_kcal, key=lambda x: str(x[0]))]


def scheduled_effect(rule, schedule, months):
    """Full-plan weight change after `months` when the energy change steps at the scheduled times. Hall's rule
    is linear, so each change adds its own exponential approach from the day it starts."""
    total, before = 0.0, 0.0
    for start, kcal in schedule:
        if months > start:
            total += float(rule.full_effect(kcal - before, months - start))
        before = kcal
    return total


def reassess(weights, plan_start, energy_change_kcal, rule=None, adherence=None):
    """Run the loop over dated weights [(visit_date, kg or None), ...] in date order.

    energy_change_kcal: the plan's daily energy change, or a list of (start date, kcal/day) when the
    plan was changed (the first start is plan_start), so earlier months are judged against the plan in
    force at the time.

    adherence: the share of the plan the patient reported following at each of those visits (None
    where not reported), the same length as weights. Returns one record per visit with a weight (the
    first is the starting point), what the twin expected before seeing it given what was followed,
    the verdict and its cause, and k after learning from it; plus projections from the last weight at
    3, 6 and 12 months, assuming the most recently reported adherence continues.
    """
    rule = WeightRule.from_tables() if rule is None else rule
    adherence = list(adherence) if adherence is not None else [None] * len(weights)
    measured = [(d, w, a) for (d, w), a in zip(weights, adherence) if w is not None]
    if not measured:
        return {'visits': [], 'response': {'mean': PRIOR_MEAN, 'sd': PRIOR_SD},
                'body_response': Response().__dict__, 'projection': []}
    t = [months_between(plan_start, d) for d, _, _ in measured]
    schedule = plan_schedule(plan_start, energy_change_kcal)
    plan = [scheduled_effect(rule, schedule, max(x, 0)) for x in t]
    records = [{'visit_date': str(measured[0][0]), 'weight_kg': measured[0][1], 'expected_kg': None,
                'expected_range_kg': None, 'verdict': 'start', 'progress': None, 'cause': None, 'adherence': None,
                'k_mean': PRIOR_MEAN, 'k_sd': PRIOR_SD}]
    start = measured[0][1]
    steps = []
    r = Response()
    for i in range(1, len(measured)):
        effect, gap = plan[i] - plan[i - 1], t[i] - t[i - 1]
        a = measured[i][2]
        mean, low, high = predict(measured[i - 1][1], r.at(TYPICAL_ADHERENCE if a is None else a), effect, gap, rule)
        observed = measured[i][1]
        verdict = ('slower than expected' if observed > high else 'faster than expected' if observed < low
                   else 'on track')
        steps.append((effect, observed - measured[i - 1][1], gap, t[i], a))
        reported_so_far = [s[4] for s in steps if s[4] is not None]
        a_plan = sum(reported_so_far) / len(reported_so_far) if reported_so_far else TYPICAL_ADHERENCE
        _, low_plan, high_plan = predict(start, Response().at(a_plan), plan[i], t[i], rule)
        typical = predict(start, Response().at(TYPICAL_ADHERENCE), plan[i], t[i], rule)[0]
        progress = 'behind plan' if is_behind(start, observed, typical, high_plan, t[i]) \
            else 'ahead of plan' if observed < low_plan else 'on plan'
        r = learn([(e, o, g, t[i] - at, aa) for e, o, g, at, aa in steps], rule)
        k = r.at(TYPICAL_ADHERENCE)
        records.append({'visit_date': str(measured[i][0]), 'weight_kg': observed, 'expected_kg': round(mean, 2),
                        'expected_range_kg': [round(low, 2), round(high, 2)], 'verdict': verdict,
                        'progress': progress, 'cause': cause(progress, a),
                        'adherence': None if a is None else round(a, 2),
                        'k_mean': round(k.mean, 3), 'k_sd': round(k.sd, 3)})
    reported = [a for _, _, a in measured[1:] if a is not None]
    ahead = reported[-1] if reported else TYPICAL_ADHERENCE
    last_t, last_w = t[-1], measured[-1][1]
    projection = []
    for h in HORIZONS:
        effect = scheduled_effect(rule, schedule, max(last_t + h, 0)) - plan[-1]
        mean, low, high = predict(last_w, r.at(ahead), effect, h, rule)
        projection.append({'months_ahead': h, 'weight_kg': round(mean, 2), 'range_kg': [round(low, 2), round(high, 2)],
                           'change_kg': round(mean - last_w, 2)})
    k = r.at(TYPICAL_ADHERENCE)
    return {'visits': records, 'response': {'mean': round(k.mean, 3), 'sd': round(k.sd, 3)},
            'body_response': {'mean': round(r.mean, 3), 'sd': round(r.sd, 3)},
            'adherence_ahead': round(ahead, 2), 'projection': projection}


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

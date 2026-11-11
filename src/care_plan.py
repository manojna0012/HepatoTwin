"""The patient's plan: a suggestion from the twin, the patient's pace, and when a clinician must approve.

Suggestion. Patients cannot be expected to know how large a calorie change to make, so the twin
suggests one from the trial prescriptions in data/evidence/simulator_parameters.csv, at three paces:

- gentle: half the DPP calorie cut for the patient's weight band, walking 90 minutes a week
  (the Vilar-Gomez starting goal, RX-EX-VG low);
- standard: the DPP cut (RX-DPP-DEFICIT, 500-1,000 kcal/day by weight band) and 150 minutes of
  brisk walking a week (RX-EX-DPP);
- ambitious: the Look AHEAD calorie goal below the DPP maintenance calories (RX-LA-GOAL; maintenance
  is the DPP goal plus deficit for the band) and 175 minutes a week (RX-EX-LA).

Walking adds (MET - 1) x weight x hours of energy use (EX-MET, 3.8 MET). The expected weight change
uses the re-assessment rule (Hall's rule, src/reassessment.py): "if followed fully" is the textbook
effect, and "typical" multiplies it by the prior response factor (0.5), what people usually achieve.

Under 18 the twin suggests no calorie cut (children and adolescents are still growing; the trials
enrolled adults), only the recipe priorities, and a clinician sets any energy change.

Approval. A plan needs a clinician's approval before it applies when any safety notice is present
(under 18, alcohol in the MetALD range or above, glucose-lowering medicine, pregnancy) or recipes are
withheld for a scarring referral (src/liver_labels.py guardrails).

Adherence. At each check-in the patient says how much of the plan they followed, for meals and for
walking. The answers become fractions (ADHERENCE; design choices for three coarse answers) and are
combined in proportion to each part's share of the plan's energy change.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.reassessment import PRIOR_MEAN, WeightRule

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ROOT / 'data' / 'evidence' / 'simulator_parameters.csv'
LB_PER_KG = 2.20462
ADULT_AGE = 18
PACES = ('gentle', 'standard', 'ambitious')
MEAL_ANSWERS = {'most': 0.9, 'half': 0.5, 'few': 0.15}
WALK_ANSWERS = {'as_planned': 0.9, 'some': 0.5, 'little': 0.15}
APPROVAL_NOTICES = {'under_18', 'alcohol', 'glucose_lowering_medicine', 'pregnant', 'fibrosis_referral'}
MIN_ENERGY_CHANGE = -1500


def _numbers(text):
    return [float(x) for x in str(text).split(';')]


@dataclass(frozen=True)
class Prescriptions:
    dpp_goal: tuple
    dpp_deficit: tuple
    la_goal: tuple
    walk_gentle: float
    walk_standard: float
    walk_ambitious: float
    met: float

    @classmethod
    def from_table(cls, path=PARAMETERS):
        p = pd.read_csv(path).set_index('param_id')
        return cls(dpp_goal=tuple(_numbers(p.loc['RX-DPP-GOAL', 'value'])),
                   dpp_deficit=tuple(_numbers(p.loc['RX-DPP-DEFICIT', 'value'])),
                   la_goal=tuple(_numbers(p.loc['RX-LA-GOAL', 'value'])),
                   walk_gentle=float(p.loc['RX-EX-VG', 'low']), walk_standard=float(p.loc['RX-EX-DPP', 'value']),
                   walk_ambitious=float(p.loc['RX-EX-LA', 'value']), met=float(p.loc['EX-MET', 'value']))

    def band(self, kg):
        """DPP weight band: < 175, 175-220, 220-250, > 250 lb."""
        lb = kg * LB_PER_KG
        return sum(lb >= edge for edge in (175, 220, 250))

    def diet_kcal(self, kg, pace):
        b = self.band(kg)
        if pace == 'gentle':
            return -self.dpp_deficit[b] / 2
        if pace == 'standard':
            return -self.dpp_deficit[b]
        goal = self.la_goal[0] if kg < 114 else self.la_goal[1]
        return max(goal - (self.dpp_goal[b] + self.dpp_deficit[b]), MIN_ENERGY_CHANGE)

    def walking_minutes(self, pace):
        return {'gentle': self.walk_gentle, 'standard': self.walk_standard, 'ambitious': self.walk_ambitious}[pace]


def walking_kcal(kg, minutes_per_week, met):
    """Daily energy used by walking (kcal/day, negative: it lowers the balance)."""
    return -(met - 1) * kg * minutes_per_week / 60 / 7


def total_energy_change(kg, diet_kcal, walking_minutes, rx=None):
    rx = rx or Prescriptions.from_table()
    return diet_kcal + walking_kcal(kg, walking_minutes, rx.met)


@dataclass
class Option:
    pace: str
    diet_kcal: float
    walking_min_per_week: float
    total_kcal: float
    expected_change_12m_kg: float
    if_followed_change_12m_kg: float
    recommended: bool


def suggest(weight_kg, age=None, rx=None, rule=None):
    """The twin's three plan options for a patient of this weight, standard first and recommended."""
    rx = rx or Prescriptions.from_table()
    rule = rule or WeightRule.from_tables()
    if age is not None and age < ADULT_AGE:
        return {'options': [], 'note': 'Under 18: no calorie cut is suggested; a clinician sets the plan.'}
    options = []
    for pace in ('standard', 'gentle', 'ambitious'):
        diet = rx.diet_kcal(weight_kg, pace)
        walk = rx.walking_minutes(pace)
        total = total_energy_change(weight_kg, diet, walk, rx)
        full = float(rule.full_effect(total, 12))
        options.append(asdict(Option(pace, round(diet), walk, round(total), round(PRIOR_MEAN * full, 1),
                                     round(full, 1), pace == 'standard')))
    return {'options': options, 'note': 'Typical change assumes people follow about half the plan, as in trials.'}


def needs_approval(notice_ids):
    """Safety notices that require a clinician to approve the plan first."""
    return sorted(set(notice_ids) & APPROVAL_NOTICES)


def reported_adherence(followed, diet_kcal, walk_kcal):
    """Share of the plan followed (0-1) from the check-in answers, or None when not answered.

    followed: {'meals': 'most'|'half'|'few', 'walking': 'as_planned'|'some'|'little'}; a missing part
    takes the other part's value."""
    if not followed:
        return None
    meals, walk = MEAL_ANSWERS.get(followed.get('meals')), WALK_ANSWERS.get(followed.get('walking'))
    if meals is None and walk is None:
        return None
    meals = walk if meals is None else meals
    walk = meals if walk is None else walk
    total = abs(diet_kcal) + abs(walk_kcal)
    if total == 0:
        return (meals + walk) / 2
    return (abs(diet_kcal) * meals + abs(walk_kcal) * walk) / total

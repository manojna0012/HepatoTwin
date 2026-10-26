"""Normal growth of adolescents from the CDC 2000 growth charts (data/evidence/cdc_growth).

A patient aged 12-17 keeps their weight-for-age and height-for-age z-scores as they grow, so on
their usual diet they gain the weight and height that a child on the same centile gains. The
charts end at 20 years; adults (18+) do not grow.
"""
import numpy as np
import pandas as pd

from src.simulation.parameters import ROOT

GROWTH_DIR = ROOT / 'data' / 'evidence' / 'cdc_growth'
ADULT_AGE = 18


def load_lms(name):
    table = pd.read_csv(GROWTH_DIR / f'{name}.csv', usecols=['Sex', 'Agemos', 'L', 'M', 'S'])
    return {sex: group.sort_values('Agemos') for sex, group in table.groupby('Sex')}


def lms_at(tables, sex, age_months):
    t = tables[int(sex)]
    return (np.interp(age_months, t.Agemos, t.L), np.interp(age_months, t.Agemos, t.M),
            np.interp(age_months, t.Agemos, t.S))


def z_score(value, l, m, s):
    return np.log(value / m) / s if abs(l) < 1e-8 else ((value / m) ** l - 1) / (l * s)


def from_z(z, l, m, s):
    return m * np.exp(s * z) if abs(l) < 1e-8 else m * (1 + l * s * z) ** (1 / l)


class Growth:
    """Expected change in weight (kg) and height (cm) after a number of months, by centile."""

    def __init__(self):
        self.weight = load_lms('wtage')
        self.height = load_lms('statage')

    def path(self, tables, value, sex, age_years, months):
        """Change from month 0 at each month in `months` for a patient on a fixed centile."""
        months = np.asarray(months, dtype=float)
        if age_years >= ADULT_AGE or not np.isfinite(value):
            return np.zeros_like(months)
        start = 12 * age_years + 6       # NHANES gives age in whole years: take the middle of the year
        z = z_score(value, *lms_at(tables, sex, start))
        return np.array([from_z(z, *lms_at(tables, sex, min(start + m, 240))) for m in months]) \
            - from_z(z, *lms_at(tables, sex, start))

    def weight_path(self, weight_kg, sex, age_years, months):
        return self.path(self.weight, weight_kg, sex, age_years, months)

    def height_path(self, height_cm, sex, age_years, months):
        return self.path(self.height, height_cm, sex, age_years, months)

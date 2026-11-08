import numpy as np
import pandas as pd

from src.nhanes3_validation import fatty_liver_index, hepatic_steatosis_index, poisson_fit


def test_the_published_scores_follow_their_formulas():
    d = pd.DataFrame({'LBXSTR': [150.0], 'BMXBMI': [30.0], 'LBXSGTSI': [40.0], 'BMXWAIST': [100.0],
                      'LBXSATSI': [40.0], 'LBXSASSI': [20.0], 'RIAGENDR': [2], 'LBXGH': [7.0], 'LBXGLU': [np.nan]})
    logit = 0.953 * np.log(150) + 0.139 * 30 + 0.718 * np.log(40) + 0.053 * 100 - 15.745
    assert abs(fatty_liver_index(d)[0] - 100 / (1 + np.exp(-logit))) < 1e-9
    assert hepatic_steatosis_index(d)[0] == 8 * 2 + 30 + 2 + 2          # female and diabetic


def test_poisson_fit_recovers_a_known_rate_ratio():
    rng = np.random.default_rng(1)
    n = 20000
    group = rng.integers(0, 2, n).astype(float)
    years = rng.uniform(5, 25, n)
    events = rng.poisson(0.01 * years * np.exp(np.log(2.0) * group))
    beta, se = poisson_fit(pd.DataFrame({'g': group}), events, years)
    assert abs(np.exp(beta[0]) - 2.0) < 0.15 and 0 < se[0] < 0.1

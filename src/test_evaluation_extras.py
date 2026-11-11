import json

import numpy as np
import pandas as pd

from src.fairness_audit import gaps, layer2_metrics
from src.in_silico_extras import perturbed
from src.rebuild_all import choose_calibration
from src.summary_evaluation import planted, readability
from src.twin_summary import article, check


def test_fairness_metrics_and_largest_gap_between_groups():
    rng = np.random.default_rng(0)
    y = np.array([0, 0, 1, 1] * 25)
    out = layer2_metrics(y, np.where(y == 1, 0.8, 0.2), rng, boot=50)
    assert out['auc']['value'] == 1 and abs(out['brier']['value'] - 0.04) < 1e-9 and out['n'] == 100
    table = {'sex: men': {'auc': {'value': 0.9}}, 'sex: women': {'auc': {'value': 0.85}}}
    assert gaps(table, 'auc') == {'sex': {'gap': 0.05, 'highest': 'men', 'lowest': 'women'}}


def test_planted_number_and_citation_errors_are_caught_and_rewording_is_not():
    facts = {'p_cap248_pct': 40, 'p_cap280_pct': 20, 'priorities': [{'evidence_ids': ['VG2015-WL5']}]}
    text = 'The chance is 40%, and of a lot 20% (evidence VG2015-WL5). Your results suggest little fat.'
    versions = planted(text)
    assert check(text, facts)['faithful']
    assert not check(versions['number changed'], facts)['faithful'] and not check(versions['citation invented'], facts)['faithful']
    assert check(versions['harmless rewording'], facts)['faithful']
    assert readability('One short line. Another one.')['sentences'] == 2
    assert article('indeterminate') == 'an indeterminate' and article('low') == 'a low'


def test_perturbed_records_keep_the_first_visit():
    rec = {'weights': [(m, 90.0) for m in range(13)]}
    out = perturbed(rec, np.random.default_rng(1), missed=1.0)
    assert out['weights'] == [(0, 90.0)]


def test_calibration_is_chosen_by_the_closest_regain(tmp_path):
    for cap, regain in ((0.1, 1.9), (0.2, 2.8)):
        folder = tmp_path / f'quit_cap_{round(100 * cap):02d}'
        folder.mkdir()
        pd.DataFrame([{'target_id': 'LA-Y1-REGAIN-T1', 'use': 'held back', 'published': 1.5, 'after': regain}]).to_csv(folder / 'targets.csv', index=False)
        (folder / 'settings.json').write_text(json.dumps({'calibration': {'adherence_mean': 0.7, 'cap': cap},
                                                          'adherence_by_trial': {'vilar_gomez': 0.5}}))
    assert choose_calibration((0.1, 0.2), tmp_path) == 0.1
    chosen = json.loads((tmp_path / 'chosen.json').read_text())
    assert chosen['calibration'] == {'adherence_mean': 0.5, 'cap': 0.1}

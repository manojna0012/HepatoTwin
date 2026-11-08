"""Rebuild every dataset, model and result of the project with one command (reproducibility check).

    .venv/Scripts/python.exe -m src.rebuild_all              # everything, in order
    .venv/Scripts/python.exe -m src.rebuild_all --list       # show the steps
    .venv/Scripts/python.exe -m src.rebuild_all --from simulate   # resume from a step

Steps run in order as separate processes and stop at the first failure; each step's time is written to
data/results/rebuild_log.json. Downloads come from the CDC and USDA (NHANES 2017-2023, FPED, NHANES III);
the NHANES III mortality file must be placed by hand if ftp.cdc.gov fails the certificate check (README).

The recipe sample is collected from RecipeDB with RECIPEDB_BASE_URL in .env; it is slow and the API may
change, so it is rebuilt only with --with-recipes; otherwise the saved sample in data/recipes is used.

Simulator calibration is repeated for quit caps 0.1-0.4 and the cap is chosen as before: the one whose
Look AHEAD third with the most regain regains closest to Neiberg 2012 (target LA-Y1-REGAIN-T1).
"""
import argparse
import json
import subprocess
import sys
import time

import pandas as pd

from src.train_final_model import ROOT

PY = sys.executable
CALIBRATION = ROOT / 'data' / 'results' / 'simulation' / 'calibration'
LOG = ROOT / 'data' / 'results' / 'rebuild_log.json'
QUIT_CAPS = (0.1, 0.2, 0.3, 0.4)
REGAIN_TARGET = 'LA-Y1-REGAIN-T1'
SCENARIO_TRIAL = 'vilar_gomez'


def module(name, *args):
    return [PY, '-m', name, *args]


# The recipe scripts import their neighbours directly, so they run as files.
RECIPES = [('recipes', [PY, f'src/recipes/{name}.py']) for name in
           ('collect_recipe_ids', 'collect_recipes', 'profile_recipes', 'profile_nutritional_features')]
STEPS = [
    ('data', module('src.build_combined_dataset', '--download')),
    ('data', module('src.build_habitual_diet', '--download')),
    ('model', module('src.train_combined_model')),
    ('model', module('src.validate_combined_model')),
    ('model', module('src.uncertainty')),
    ('model', module('src.fibrosis')),
    ('recipes-scores', module('src.recipe_scoring.foodprox')),
    ('recipes-scores', module('src.recipe_scoring.processing_and_glycaemic_load')),
    ('ranking', module('src.evaluate_ranking', '--no-diet')),
    ('ranking', module('src.evaluate_ranking')),
    *[('calibrate', module('src.simulation.calibrate', '--quit-cap', str(c))) for c in QUIT_CAPS],
    ('calibrate', 'choose_calibration'),
    ('simulate', module('src.simulation.run_simulation')),
    ('in-silico', module('src.in_silico_evaluation')),
    ('in-silico', module('src.in_silico_extras')),
    ('nhanes3', module('src.nhanes3', '--download')),
    ('nhanes3', module('src.nhanes3_validation')),
    ('audits', module('src.fairness_audit')),
    ('audits', module('src.summary_evaluation')),
    ('figures', module('src.thesis_figures')),
    ('tests', [PY, '-m', 'pytest', '-q']),
]


def choose_calibration(caps=QUIT_CAPS, folder=CALIBRATION):
    """Write chosen.json from the cap whose simulated first-year regain is closest to its target."""
    gaps = {}
    for cap in caps:
        t = pd.read_csv(folder / f'quit_cap_{round(100 * cap):02d}' / 'targets.csv').set_index('target_id')
        gaps[cap] = (float(t.loc[REGAIN_TARGET, 'after']), abs(float(t.loc[REGAIN_TARGET, 'after']) - float(t.loc[REGAIN_TARGET, 'published'])))
    best = min(gaps, key=lambda c: gaps[c][1])
    settings = json.loads((folder / f'quit_cap_{round(100 * best):02d}' / 'settings.json').read_text())
    chosen = {'chosen_from': f'quit_cap_{round(100 * best):02d}',
              'reason': 'Of the quit caps ' + ', '.join(f'{c:g}' for c in caps) + ', the one whose Look AHEAD third with '
                        'the most regain regained closest to Neiberg 2012 (' + REGAIN_TARGET + '): '
                        + ', '.join(f'{gaps[c][0]:.2f} at cap {c:g}' for c in caps) + '. Written by src.rebuild_all.',
              'scenario_adherence_from': f'{SCENARIO_TRIAL} (routine clinical advice to patients with fatty liver; no intensive-contact phase)',
              'calibration': {**settings['calibration'], 'adherence_mean': settings['adherence_by_trial'][SCENARIO_TRIAL]},
              'adherence_by_trial': settings['adherence_by_trial']}
    (folder / 'chosen.json').write_text(json.dumps(chosen, indent=2))
    return best


def label(command):
    return command if isinstance(command, str) else ' '.join(command[1:]).replace('-m ', '')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--list', action='store_true', help='show the steps and stop')
    parser.add_argument('--from', dest='start', default=None, help='first stage to run (e.g. simulate)')
    parser.add_argument('--with-recipes', action='store_true', help='also re-collect the RecipeDB sample')
    args = parser.parse_args()
    steps = (RECIPES if args.with_recipes else []) + STEPS
    if args.start:
        stages = [s for s, _ in steps]
        if args.start not in stages:
            raise SystemExit(f'unknown stage {args.start}; stages: {", ".join(dict.fromkeys(stages))}')
        steps = steps[stages.index(args.start):]
    if args.list:
        for i, (stage, command) in enumerate(steps, 1):
            print(f'{i:2d}. [{stage}] {label(command)}')
        return
    log = []
    for stage, command in steps:
        began = time.time()
        print(f'[{stage}] {label(command)}', flush=True)
        if command == 'choose_calibration':
            print(f'  chose quit cap {choose_calibration()}', flush=True)
        else:
            done = subprocess.run(command, cwd=ROOT)
            if done.returncode:
                log.append({'stage': stage, 'step': label(command), 'seconds': round(time.time() - began), 'failed': True})
                LOG.write_text(json.dumps(log, indent=2))
                raise SystemExit(f'step failed: {label(command)}')
        log.append({'stage': stage, 'step': label(command), 'seconds': round(time.time() - began)})
        LOG.write_text(json.dumps(log, indent=2))
    print(f'rebuilt in {sum(x["seconds"] for x in log) / 60:.0f} minutes')


if __name__ == '__main__':
    main()

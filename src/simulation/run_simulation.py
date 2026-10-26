"""Generate the virtual cohort's true monthly trajectories.

Month 0 = the in-scope NHANES test patients; each follows every scenario for 12 months. The
scenario sizes below are placeholders until the evaluation links them to Layer 5 recipe
scenarios (Weekly Plan, meeting 12). Outputs go to data/results/simulation/:

- truth.csv.gz: one row per patient, scenario and month (the simulator's hidden truth)
- traits.csv: one row per patient (adherence pattern and response multipliers)
- summary.csv: cohort means by scenario and month, for checking against the trials

    .venv/Scripts/python.exe -m src.simulation.run_simulation
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.simulation.cohort import month_zero
from src.simulation.parameters import ROOT, Calibration, Evidence
from src.simulation.simulator import simulate_cohort

OUTPUT = ROOT / 'data' / 'results' / 'simulation'
SEED = 42
# Placeholder sizes, replaced by Layer 5 recipe scenarios in the evaluation (meeting 12).
SCENARIOS = [{'name': 'usual diet', 'energy_change_kcal': 0},
             {'name': 'scenario A', 'energy_change_kcal': -500},
             {'name': 'scenario B', 'energy_change_kcal': -250}]


def summarise(truth, baseline):
    w0 = baseline.set_index('SEQN').BMXWT
    cap0 = baseline.set_index('SEQN').LUXCAPM
    t = truth.assign(weight_change_pct=100 * (truth.BMXWT / truth.SEQN.map(w0) - 1),
                     cap_change=truth.LUXCAPM - truth.SEQN.map(cap0))
    return (t.groupby(['scenario', 'month'])
             .agg(n=('SEQN', 'size'), weight_change_pct=('weight_change_pct', 'mean'),
                  lost_5pct=('weight_change_pct', lambda s: 100 * (s <= -5).mean()),
                  lost_10pct=('weight_change_pct', lambda s: 100 * (s <= -10).mean()),
                  cap=('LUXCAPM', 'mean'), cap_sd=('LUXCAPM', 'std'), cap_change=('cap_change', 'mean'),
                  alt=('LBXSATSI', 'mean'), hba1c=('LBXGH', 'mean'))
             .reset_index())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--out', default=str(OUTPUT))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cohort = month_zero()
    ev, cal = Evidence.from_table(), Calibration()
    print(f'Month 0: {len(cohort)} in-scope test patients; weight time constant '
          f'{ev.weight_tau_months:.1f} months', flush=True)
    truth, traits = simulate_cohort(cohort, SCENARIOS, args.seed, ev, cal)
    truth.to_csv(out / 'truth.csv.gz', index=False)
    traits.to_csv(out / 'traits.csv', index=False)
    summary = summarise(truth, cohort)
    summary.to_csv(out / 'summary.csv', index=False)
    (out / 'run_info.json').write_text(json.dumps({
        'seed': args.seed, 'patients': len(cohort), 'scenarios': SCENARIOS,
        'weight_tau_months': ev.weight_tau_months, 'calibration': cal.__dict__,
        'note': 'Calibration settings are starting guesses, not yet tuned to the trial targets.'},
        indent=2, default=str))
    print(summary[summary.month.isin([0, 6, 12])].round(2).to_string(index=False))


if __name__ == '__main__':
    main()

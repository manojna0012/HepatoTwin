"""Build the habitual-diet table for the FibroScan cohort (proposal section 17.1).

Merges the day-1 24-hour recall totals (DR1TOT) of both cycles with USDA Food
Patterns Equivalents (FPED) day-1 totals, then adds the composition measures
and HEI-2020 score from src/recipe_scoring/habitual_diet.py. Rows are keyed
by SEQN and join onto data/processed/nhanes_combined.csv.

FPED source by cycle:
- 2017-March 2020: USDA's participant-level FPED totals (fped_dr1tot_1720).
- 2021-2023: USDA has not released FPED for this cycle, so the totals are
  derived with USDA's own method: each food in the day-1 individual-foods
  file (DR1IFF_L) contributes its FPED per-100 g values from the 2017-March
  2020 food-level database (fped_1720) times grams eaten / 100. About 99.6%
  of 2021-2023 food energy has a matching food code. A participant's FPED
  values are kept only when foods with a match supply at least
  MIN_FPED_COVERAGE of their recall energy; unmatched foods contribute 0.

--validate applies the same derivation to the 2017-2020 individual foods
(P_DR1IFF) and compares it with USDA's official totals, writing
data/results/diet/fped_validation.csv.

USDA distributes FPED as 7-Zip self-extracting archives; --download fetches
them and extracts the SAS files with py7zr, without running the executables.
Run from the project root after build_combined_dataset.py:

    .venv/Scripts/python.exe -m src.build_habitual_diet --download
    .venv/Scripts/python.exe -m src.build_habitual_diet --download --validate
"""
import argparse
import io
import urllib.request

import numpy as np
import pandas as pd

from src.build_combined_dataset import CYCLES, OUTPUT as COHORT_FILE, component_path
from src.recipe_scoring.habitual_diet import habitual_intake
from src.train_final_model import ROOT

OUTPUT = ROOT / 'data' / 'processed' / 'nhanes_diet.csv'
VALIDATION = ROOT / 'data' / 'results' / 'diet' / 'fped_validation.csv'
USDA = 'https://www.ars.usda.gov/ARSUserFiles/80400530/apps'
FPED_TOTALS = (ROOT / 'data' / 'raw' / 'fped_dr1tot_1720.sas7bdat', f'{USDA}/FPED_DR1TOT_1720_sas.EXE')
FPED_FOODS = (ROOT / 'data' / 'raw' / 'fped_1720.sas7bdat', f'{USDA}/FPED_1720_sas.EXE')
MIN_FPED_COVERAGE = 0.95
RECALL_COLUMNS = ['DR1DRSTZ', 'DR1_300', 'DR1TKCAL', 'DR1TSFAT', 'DR1TMFAT', 'DR1TPFAT',
                  'DR1TSUGR', 'DR1TFIBE', 'DR1TSODI']
FPED_COLUMNS = ['DR1T_F_TOTAL', 'DR1T_F_CITMLB', 'DR1T_F_OTHER', 'DR1T_V_TOTAL', 'DR1T_V_LEGUMES',
                'DR1T_V_DRKGR', 'DR1T_G_WHOLE', 'DR1T_G_REFINED', 'DR1T_D_TOTAL', 'DR1T_PF_TOTAL',
                'DR1T_PF_LEGUMES', 'DR1T_PF_SEAFD_HI', 'DR1T_PF_SEAFD_LOW', 'DR1T_PF_SOY',
                'DR1T_PF_NUTSDS', 'DR1T_ADD_SUGARS']
MEASURES = ['recall_reliable', 'saturated_fat_pct_energy', 'sugar_pct_energy', 'added_sugar_pct_energy',
            'fiber_per_1000kcal', 'hei2020']


def usda_file(spec, download):
    path, url = spec
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError(f'{path} not found; rerun with --download.')
    import py7zr
    print(f'Downloading {url}', flush=True)
    with urllib.request.urlopen(url) as response:
        payload = response.read()
    start = payload.find(b'7z\xbc\xaf\x27\x1c')
    if start < 0:
        raise ValueError(f'{url} is not a 7-Zip archive.')
    path.parent.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(io.BytesIO(payload[start:])) as archive:
        archive.extract(path=path.parent, targets=[path.name])
    return path


def zero_tiny(data):
    # The SAS readers return exact zeros as ~1e-79.
    numeric = data.select_dtypes('number').columns
    data[numeric] = data[numeric].mask(data[numeric].abs() < 1e-12, 0.0)
    return data


def derive_fped(foods_eaten, fped_foods):
    """Day-1 FPED totals from individual foods: per-100 g equivalents x grams / 100.

    Returns one row per SEQN with the DR1T_* FPED columns and
    fped_energy_matched, the share of recall energy from foods with an FPED code.
    """
    eaten = foods_eaten.dropna(subset=['DR1IFDCD', 'DR1IGRMS'])[['SEQN', 'DR1IFDCD', 'DR1IGRMS', 'DR1IKCAL']]
    per_100g = fped_foods.rename(columns={'FOODCODE': 'DR1IFDCD'})
    per_100g = per_100g[['DR1IFDCD'] + [c.removeprefix('DR1T_') for c in FPED_COLUMNS]]
    merged = eaten.merge(per_100g, on='DR1IFDCD', how='left', validate='many_to_one', indicator=True)
    matched = merged['_merge'].eq('both')
    totals = pd.DataFrame({'SEQN': merged.SEQN})
    for column in FPED_COLUMNS:
        totals[column] = merged[column.removeprefix('DR1T_')].fillna(0) * merged.DR1IGRMS / 100
    totals['matched_kcal'] = merged.DR1IKCAL.fillna(0) * matched
    totals['kcal'] = merged.DR1IKCAL.fillna(0)
    day = totals.groupby('SEQN').sum()
    with np.errstate(invalid='ignore', divide='ignore'):
        day['fped_energy_matched'] = np.where(day.kcal > 0, day.matched_kcal / day.kcal, np.nan)
    return day.drop(columns=['matched_kcal', 'kcal']).reset_index()


def load_cycle(cycle, download=False):
    recall = pd.read_sas(component_path(cycle, 'DR1TOT', download))[['SEQN'] + RECALL_COLUMNS]
    demo = pd.read_sas(component_path(cycle, 'DEMO', download))[['SEQN', 'RIAGENDR']]
    data = recall.merge(demo, on='SEQN', how='left', validate='one_to_one')
    if cycle == '2017-2020':
        fped = pd.read_sas(usda_file(FPED_TOTALS, download))[['SEQN'] + FPED_COLUMNS]
        fped['fped_source'] = 'USDA'
    else:
        foods = pd.read_sas(usda_file(FPED_FOODS, download), encoding='latin1')
        fped = derive_fped(pd.read_sas(component_path(cycle, 'DR1IFF', download)), foods)
        low = ~(fped.fped_energy_matched >= MIN_FPED_COVERAGE)
        fped.loc[low, FPED_COLUMNS] = np.nan
        fped['fped_source'] = np.where(low, 'insufficient food-code coverage',
                                       'derived from 2017-2020 food-level FPED')
    data = data.merge(fped, on='SEQN', how='left', validate='one_to_one')
    data = zero_tiny(data)
    data.insert(1, 'cycle', cycle)
    return data


def build(download=False):
    cohort = set(pd.read_csv(COHORT_FILE, usecols=['SEQN']).SEQN)
    data = pd.concat([load_cycle(cycle, download) for cycle in CYCLES], ignore_index=True)
    data = data[data.SEQN.isin(cohort)].reset_index(drop=True)
    measures = pd.DataFrame([habitual_intake(row) for row in data.to_dict('records')])
    return pd.concat([data, measures[MEASURES]], axis=1)


def validate(download=False):
    """Derive FPED for 2017-2020 from individual foods and compare with USDA's totals."""
    official = zero_tiny(pd.read_sas(usda_file(FPED_TOTALS, download)))
    foods = pd.read_sas(usda_file(FPED_FOODS, download), encoding='latin1')
    derived = zero_tiny(derive_fped(pd.read_sas(component_path('2017-2020', 'DR1IFF', download)), foods))
    recall = zero_tiny(pd.read_sas(component_path('2017-2020', 'DR1TOT', download))[['SEQN'] + RECALL_COLUMNS])
    demo = pd.read_sas(component_path('2017-2020', 'DEMO', download))[['SEQN', 'RIAGENDR']]
    base = recall.merge(demo, on='SEQN')
    cohort = set(pd.read_csv(COHORT_FILE, usecols=['SEQN']).SEQN)
    pair = (base.merge(official[['SEQN'] + FPED_COLUMNS], on='SEQN')
            .merge(derived, on='SEQN', suffixes=('_usda', '_derived')))
    pair = pair[pair.SEQN.isin(cohort) & (pair.fped_energy_matched >= MIN_FPED_COVERAGE)]
    rows = []
    for column in FPED_COLUMNS:
        a, b = pair[f'{column}_usda'], pair[f'{column}_derived']
        rows.append(dict(measure=column, n=len(pair), pearson_r=a.corr(b),
                         mean_usda=a.mean(), mean_derived=b.mean(),
                         median_abs_diff=(a - b).abs().median(),
                         share_within_1pct=((a - b).abs() <= 0.01 * a.abs().clip(lower=1e-9)).mean()))

    def scores(suffix):
        out = []
        for row in pair.to_dict('records'):
            r = {**row, **{c: row[f'{c}{suffix}'] for c in FPED_COLUMNS}}
            intake = habitual_intake(r)
            out.append((intake['added_sugar_pct_energy'], intake['hei2020']))
        return pd.DataFrame(out, columns=['added', 'hei'])
    usda, ours = scores('_usda'), scores('_derived')
    keep = usda.added.notna() & ours.added.notna()
    rows.append(dict(measure='added_sugar_pct_energy', n=int(keep.sum()), pearson_r=usda.added[keep].corr(
        ours.added[keep]), mean_usda=usda.added[keep].mean(), mean_derived=ours.added[keep].mean(),
        median_abs_diff=(usda.added - ours.added)[keep].abs().median(),
        share_within_1pct=float(((usda.added > 10) == (ours.added > 10))[keep].mean())))
    keep = usda.hei.notna() & ours.hei.notna()
    rows.append(dict(measure='hei2020', n=int(keep.sum()), pearson_r=usda.hei[keep].corr(ours.hei[keep]),
                     mean_usda=usda.hei[keep].mean(), mean_derived=ours.hei[keep].mean(),
                     median_abs_diff=(usda.hei - ours.hei)[keep].abs().median(),
                     share_within_1pct=float(((usda.hei - ours.hei)[keep].abs() <= 1).mean())))
    result = pd.DataFrame(rows)
    VALIDATION.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(VALIDATION, index=False)
    print('Derived vs USDA FPED, 2017-2020 cohort participants with >= 95% energy matched.')
    print('share_within_1pct: within 1% of USDA; for added sugars, agreement on > 10% of energy;')
    print('for HEI-2020, within 1 point.')
    print(result.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'Saved {VALIDATION}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Fetch missing CDC and USDA files.')
    parser.add_argument('--validate', action='store_true', help='Compare derived FPED with USDA (2017-2020).')
    args = parser.parse_args()
    if args.validate:
        validate(args.download)
        return
    data = build(args.download)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT, index=False)
    reliable = data[data.recall_reliable]
    print(f'Saved {len(data)} cohort participants to {OUTPUT}; {len(reliable)} with a reliable recall')
    print(data.groupby('cycle').fped_source.value_counts().to_string())
    print(reliable.groupby('cycle')[MEASURES[1:]].median().round(1).to_string())


if __name__ == '__main__':
    main()

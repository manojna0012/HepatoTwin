"""Build the habitual-diet table for the FibroScan cohort (proposal section 17.1).

Merges the day-1 24-hour recall totals (DR1TOT) of both cycles with the USDA
FPED day-1 totals for 2017-March 2020, then adds the composition measures
and HEI-2020 score from src/recipe_scoring/habitual_diet.py. Rows are keyed
by SEQN and join onto data/processed/nhanes_combined.csv.

USDA distributes FPED as a 7-Zip self-extracting archive; --download fetches
it and extracts the SAS file with py7zr, without running the executable.
Run from the project root after build_combined_dataset.py:

    .venv/Scripts/python.exe -m src.build_habitual_diet --download
"""
import argparse
import io
import urllib.request

import pandas as pd

from src.build_combined_dataset import CYCLES, OUTPUT as COHORT_FILE, component_path
from src.recipe_scoring.habitual_diet import habitual_intake
from src.train_final_model import ROOT

OUTPUT = ROOT / 'data' / 'processed' / 'nhanes_diet.csv'
FPED_FILE = ROOT / 'data' / 'raw' / 'fped_dr1tot_1720.sas7bdat'
FPED_URL = 'https://www.ars.usda.gov/ARSUserFiles/80400530/apps/FPED_DR1TOT_1720_sas.EXE'
RECALL_COLUMNS = ['DR1DRSTZ', 'DR1_300', 'DR1TKCAL', 'DR1TSFAT', 'DR1TMFAT', 'DR1TPFAT',
                  'DR1TSUGR', 'DR1TFIBE', 'DR1TSODI']
FPED_COLUMNS = ['DR1T_F_TOTAL', 'DR1T_F_CITMLB', 'DR1T_F_OTHER', 'DR1T_V_TOTAL', 'DR1T_V_LEGUMES',
                'DR1T_V_DRKGR', 'DR1T_G_WHOLE', 'DR1T_G_REFINED', 'DR1T_D_TOTAL', 'DR1T_PF_TOTAL',
                'DR1T_PF_LEGUMES', 'DR1T_PF_SEAFD_HI', 'DR1T_PF_SEAFD_LOW', 'DR1T_PF_SOY',
                'DR1T_PF_NUTSDS', 'DR1T_ADD_SUGARS']
MEASURES = ['recall_reliable', 'saturated_fat_pct_energy', 'sugar_pct_energy', 'added_sugar_pct_energy',
            'fiber_per_1000kcal', 'hei2020']


def fped_path(download):
    if FPED_FILE.exists():
        return FPED_FILE
    if not download:
        raise FileNotFoundError(f'{FPED_FILE} not found; rerun with --download.')
    import py7zr
    print(f'Downloading {FPED_URL}', flush=True)
    with urllib.request.urlopen(FPED_URL) as response:
        payload = response.read()
    start = payload.find(b'7z\xbc\xaf\x27\x1c')
    if start < 0:
        raise ValueError('FPED download is not a 7-Zip archive.')
    FPED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(io.BytesIO(payload[start:])) as archive:
        archive.extract(path=FPED_FILE.parent, targets=[FPED_FILE.name])
    return FPED_FILE


def load_cycle(cycle, download=False):
    recall = pd.read_sas(component_path(cycle, 'DR1TOT', download))[['SEQN'] + RECALL_COLUMNS]
    demo = pd.read_sas(component_path(cycle, 'DEMO', download))[['SEQN', 'RIAGENDR']]
    data = recall.merge(demo, on='SEQN', how='left', validate='one_to_one')
    if cycle == '2017-2020':
        fped = pd.read_sas(fped_path(download))[['SEQN'] + FPED_COLUMNS]
        data = data.merge(fped, on='SEQN', how='left', validate='one_to_one')
    numeric = data.select_dtypes('number').columns
    # The SAS readers return exact zeros as ~1e-79.
    data[numeric] = data[numeric].mask(data[numeric].abs() < 1e-12, 0.0)
    data.insert(1, 'cycle', cycle)
    return data


def build(download=False):
    cohort = set(pd.read_csv(COHORT_FILE, usecols=['SEQN']).SEQN)
    data = pd.concat([load_cycle(cycle, download) for cycle in CYCLES], ignore_index=True)
    data = data[data.SEQN.isin(cohort)].reset_index(drop=True)
    measures = pd.DataFrame([habitual_intake(row) for row in data.to_dict('records')])
    return pd.concat([data, measures[MEASURES]], axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Fetch missing CDC and USDA files.')
    args = parser.parse_args()
    data = build(args.download)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT, index=False)
    reliable = data[data.recall_reliable]
    print(f'Saved {len(data)} cohort participants to {OUTPUT}; {len(reliable)} with a reliable recall')
    print(reliable.groupby('cycle')[MEASURES[1:]].median().round(1).to_string())


if __name__ == '__main__':
    main()

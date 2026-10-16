"""Build the pooled 2017-2023 NHANES FibroScan cohort for the combined CAP model.

Merges the 2017-March 2020 pre-pandemic files (P_ prefix) and the August
2021-August 2023 files (_L suffix). Both cycles use the same FibroScan exam and
variable names; 2021-2023 is the last NHANES cycle with liver elastography.
Keeps complete exams (LUAXSTAT == 1) with observed CAP, plus the CAP IQR used
for exam-quality weighting. Values stay in raw NHANES coding;
CombinedCAPFeatures handles missing-value codes.

Run from the project root (--download fetches any missing CDC files):

    .venv/Scripts/python.exe -m src.build_combined_dataset --download
"""
import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.train_final_model import ROOT, TARGET
from src.combined_features import COMBINED_FEATURES

CDC = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public'
OUTPUT = ROOT / 'data' / 'processed' / 'nhanes_combined.csv'
COMPONENTS = ['LUX', 'DEMO', 'BMX', 'BIOPRO', 'GHB', 'HDL', 'TCHOL', 'DR1TOT',
              'INS', 'GLU', 'CBC', 'HSCRP', 'FERTIN', 'BPXO',
              'DIQ', 'BPQ', 'WHQ', 'HUQ', 'SLQ', 'SMQ', 'ALQ', 'PAQ']
CYCLES = {
    '2017-2020': dict(name='P_{}', year=2017, folders=['raw', 'raw_extra']),
    '2021-2023': dict(name='{}_L', year=2021, folders=['raw_2021']),
}
EXAM_COLUMNS = ['LUAXSTAT', TARGET, 'LUXCPIQR']


def component_path(cycle, component, download):
    spec = CYCLES[cycle]
    filename = spec['name'].format(component) + '.xpt'
    for folder in spec['folders']:
        path = ROOT / 'data' / folder / filename
        if path.exists():
            return path
    if not download:
        raise FileNotFoundError(f'{filename} not found in data/{spec["folders"]}; rerun with --download.')
    path = ROOT / 'data' / spec['folders'][-1] / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f'{CDC}/{spec["year"]}/DataFiles/{filename}'
    print(f'Downloading {url}', flush=True)
    urllib.request.urlretrieve(url, path)
    return path


def load_cycle(cycle, download=False):
    wanted = set(EXAM_COLUMNS + COMBINED_FEATURES)
    data = None
    for component in COMPONENTS:
        part = pd.read_sas(component_path(cycle, component, download))
        keep = [c for c in part.columns if c in wanted and (data is None or c not in data)]
        part = part[['SEQN'] + keep]
        data = part if data is None else data.merge(part, on='SEQN', how='left', validate='one_to_one')
    missing = sorted(wanted - set(data.columns))
    if missing:
        raise ValueError(f'{cycle} files lack expected columns: {missing}')
    data = data.loc[data.LUAXSTAT.eq(1) & data[TARGET].notna()].copy()
    # The SAS transport reader returns exact zeros as ~1e-79.
    numeric = data.select_dtypes('number').columns
    data[numeric] = data[numeric].mask(data[numeric].abs() < 1e-12, 0.0)
    data.insert(1, 'cycle', cycle)
    return data


def build(download=False):
    data = pd.concat([load_cycle(cycle, download) for cycle in CYCLES], ignore_index=True)
    # 2021-2023 SEQN values continue past the 2017-2020 range.
    if data.SEQN.duplicated().any():
        raise ValueError('Participant IDs must be unique across cycles.')
    if np.isinf(data[COMBINED_FEATURES].to_numpy(dtype=float)).any():
        raise ValueError('Infinite values in source data.')
    return data[['SEQN', 'cycle'] + EXAM_COLUMNS + COMBINED_FEATURES]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Fetch missing files from the CDC.')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    data = build(args.download)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(data.cycle.value_counts().to_string())
    print(f'Saved {len(data)} eligible participants to {args.output}')


if __name__ == '__main__':
    main()

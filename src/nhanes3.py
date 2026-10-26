"""NHANES III (1988-1994) data for external and long-term validation (Proposal section 17.9).

Downloads the NHANES III examination and laboratory files, the hepatic steatosis re-reading of the
archived gallbladder ultrasound (HGUHS, 2009-2010) and, when available, the public-use Linked
Mortality File (follow-up to 31 December 2019). Maps NHANES III variables onto the Layer 2 model's
input names, converting units where they differ, and keeps adults aged 20-59, inside the model's
scope, with a gradable ultrasound.

Outcome: GUPHSPFR, the recode NCHS recommends (1 = normal or mild, 2 = moderate or severe steatosis).
It is a different reference standard (ultrasound grade, not CAP) in a different era, so it tests
whether the model ranks steatosis correctly, not whether it reproduces CAP values.

Not mapped: the 24-hour recall, the questionnaire items and some blood counts; XGBoost treats them as
missing. NHANES III assays were not recalibrated to the 2017-2023 methods, and its CRP is not a
high-sensitivity assay.

    .venv/Scripts/python.exe -m src.nhanes3 [--download]
"""
import argparse
import json
import re
import shutil
import urllib.request

import numpy as np
import pandas as pd

from src.train_final_model import ROOT

RAW = ROOT / 'data' / 'raw' / 'nhanes3'
OUTPUT = ROOT / 'data' / 'processed' / 'nhanes3_mapped.csv'
MAP_OUTPUT = ROOT / 'data' / 'results' / 'nhanes3' / 'variable_map.csv'
CDC = 'https://wwwn.cdc.gov/nchs/data/nhanes3'
FILES = {'lab.dat': '1a', 'lab.sas': '1a', 'exam.dat': '1a', 'exam.sas': '1a', 'HGUHS.xpt': '34a'}
MORTALITY_FILE = 'NHANES_III_MORT_2019_PUBLIC.dat'
MORTALITY_URL = ('https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/'
                 + MORTALITY_FILE)
AGES = (20, 59)
FASTING_HOURS = 8
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

# NHANES III variable -> (file, model input, multiplier, note)
VARIABLES = [
    ('HSAGEIR', 'exam', 'RIDAGEYR', 1, 'Age in years'),
    ('HSSEX', 'exam', 'RIAGENDR', 1, '1 male, 2 female in both surveys'),
    ('DMARETHN', 'exam', 'RIDRETH3', None, 'Recoded: 1 NH white -> 3, 2 NH black -> 4, 3 Mexican American -> 1, 4 other -> 7'),
    ('BMPWT', 'exam', 'BMXWT', 1, 'kg'), ('BMPHT', 'exam', 'BMXHT', 1, 'cm'), ('BMPBMI', 'exam', 'BMXBMI', 1, 'kg/m2'),
    ('BMPWAIST', 'exam', 'BMXWAIST', 1, 'cm'), ('BMPBUTTO', 'exam', 'BMXHIP', 1, 'Buttocks circumference, cm'),
    ('BMPARMC', 'exam', 'BMXARMC', 1, 'cm'),
    ('PEPMNK1R', 'exam', 'BPXOSY1', 1, 'Average systolic BP; used for all three readings'),
    ('PEPMNK5R', 'exam', 'BPXODI1', 1, 'Average diastolic BP; used for all three readings'),
    ('ATPSI', 'lab', 'LBXSATSI', 1, 'ALT, U/L (assay not recalibrated)'),
    ('ASPSI', 'lab', 'LBXSASSI', 1, 'AST, U/L (assay not recalibrated)'),
    ('GGPSI', 'lab', 'LBXSGTSI', 1, 'GGT, U/L'), ('TGP', 'lab', 'LBXSTR', 1, 'Triglycerides, mg/dL'),
    ('HDP', 'lab', 'LBDHDD', 1, 'HDL cholesterol, mg/dL'), ('TCP', 'lab', 'LBXTC', 1, 'Total cholesterol, mg/dL'),
    ('GHP', 'lab', 'LBXGH', 1, 'HbA1c, %'), ('SGP', 'lab', 'LBXSGL', 1, 'Serum glucose, mg/dL'),
    ('G1P', 'lab', 'LBXGLU', 1, 'Plasma glucose, mg/dL; kept only after >= 8 h fasting'),
    ('I1P', 'lab', 'LBXIN', 1, 'Insulin, uU/mL; kept only after >= 8 h fasting'),
    ('UAP', 'lab', 'LBXSUA', 1, 'Uric acid, mg/dL'),
    ('CRP', 'lab', 'LBXHSCRP', 10, 'CRP mg/dL x 10 = mg/L; not a high-sensitivity assay'),
    ('WCP', 'lab', 'LBXWBCSI', 1, '1000 cells/uL'), ('PLP', 'lab', 'LBXPLTSI', 1, '1000 cells/uL'),
    ('LMPPCNT', 'lab', 'LBXLYPCT', 1, 'Lymphocytes, %'),
    ('MOPPCNT', 'lab', 'LBXMOPCT', 1, 'Mononuclear %, closest to the monocyte share'),
    ('HGP', 'lab', 'LBXHGB', 1, 'g/dL'), ('HTP', 'lab', 'LBXHCT', 1, '%'), ('MVPSI', 'lab', 'LBXMCVSI', 1, 'fL'),
    ('RWP', 'lab', 'LBXRDW', 1, '%'), ('PVPSI', 'lab', 'LBXMPSI', 1, 'fL'),
    ('FEP', 'lab', 'LBXSIR', 1, 'Iron, ug/dL'), ('FRP', 'lab', 'LBXFER', 1, 'Ferritin, ng/mL'),
    ('CAPSI', 'lab', 'LBXSCA', 4.008, 'Calcium mmol/L x 4.008 = mg/dL'),
    ('NAPSI', 'lab', 'LBXSNASI', 1, 'mmol/L'), ('SKPSI', 'lab', 'LBXSKSI', 1, 'mmol/L'),
    ('CLPSI', 'lab', 'LBXSCLSI', 1, 'mmol/L'), ('C3PSI', 'lab', 'LBXSC3SI', 1, 'mmol/L'),
    ('PSP', 'lab', 'LBXSPH', 1, 'Phosphorus, mg/dL'), ('BUP', 'lab', 'LBXSBU', 1, 'Urea nitrogen, mg/dL'),
    ('TBP', 'lab', 'LBXSTB', 1, 'Bilirubin, mg/dL'), ('CEP', 'lab', 'LBXSCR', 1, 'Creatinine, mg/dL'),
    ('TPP', 'lab', 'LBXSTP', 1, 'Total protein, g/dL'), ('AMP', 'lab', 'LBXSAL', 1, 'Albumin, g/dL'),
    ('APPSI', 'lab', 'LBXSAPSI', 1, 'Alkaline phosphatase, U/L'), ('LDPSI', 'lab', 'LBXSLDSI', 1, 'LDH, U/L'),
    ('PHPFAST', 'lab', None, 1, 'Hours since last meal (fasting filter only)'),
]
RACE = {1: 3, 2: 4, 3: 1, 4: 7}


def download(names=FILES, mortality=True):
    RAW.mkdir(parents=True, exist_ok=True)
    targets = [(name, f'{CDC}/{folder}/{name}') for name, folder in names.items()]
    if mortality:
        targets.append((MORTALITY_FILE, MORTALITY_URL))
    for name, url in targets:
        path = RAW / name
        if path.exists():
            continue
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=1800) as r, \
                    open(path, 'wb') as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            print(f'Downloaded {name}', flush=True)
        except Exception as error:
            print(f'Could not download {name} ({error}); place it in {RAW} by hand.', flush=True)


def column_positions(sas_file):
    """{variable: (start, end)} from the INPUT statement of an NHANES III SAS program (1-based)."""
    text = (RAW / sas_file).read_text(encoding='latin-1')
    block = text[text.upper().index('INPUT'):]
    block = block[:block.index(';')]
    return {name: (int(start), int(end or start))
            for name, start, end in re.findall(r'\b([A-Z][A-Z0-9]*)\s+(?:\$\s*)?(\d+)(?:-(\d+))?', block)}


def read_fixed(dat_file, sas_file, names):
    positions = column_positions(sas_file)
    names = ['SEQN'] + [n for n in names if n != 'SEQN']
    missing = [n for n in names if n not in positions]
    if missing:
        raise ValueError(f'{sas_file} has no positions for {missing}')
    raw = pd.read_fwf(RAW / dat_file, colspecs=[(positions[n][0] - 1, positions[n][1]) for n in names],
                      names=names, dtype=str)
    out = pd.DataFrame({'SEQN': pd.to_numeric(raw.SEQN)})
    for name in names[1:]:
        width = positions[name][1] - positions[name][0] + 1
        values = raw[name].str.strip()
        # NHANES III codes "blank but applicable" and "don't know" as a full field of 8s or 9s.
        blank = values.str.replace('.', '', regex=False).str.fullmatch(r'8+|9+') & (values.str.len() >= width - 1)
        out[name] = pd.to_numeric(values.mask(blank), errors='coerce')
    return out


def build():
    lab_vars = [v for v, f, *_ in VARIABLES if f == 'lab']
    exam_vars = [v for v, f, *_ in VARIABLES if f == 'exam']
    exam = read_fixed('exam.dat', 'exam.sas', exam_vars)
    lab = read_fixed('lab.dat', 'lab.sas', lab_vars)
    ultrasound = pd.read_sas(RAW / 'HGUHS.xpt')[['SEQN', 'GUPHSPFR']]
    data = exam.merge(lab, on='SEQN', how='inner').merge(ultrasound, on='SEQN', how='inner')
    data = data[data.HSAGEIR.between(*AGES) & data.GUPHSPFR.isin([1, 2])].reset_index(drop=True)
    fasting = data.PHPFAST >= FASTING_HOURS
    out = pd.DataFrame({'SEQN': data.SEQN.astype(int),
                        'steatosis_moderate_severe': (data.GUPHSPFR == 2).astype(int)})
    for var, _, target, factor, _ in VARIABLES:
        if target is None:
            continue
        values = data[var].map(RACE) if var == 'DMARETHN' else data[var] * factor
        if var in ('G1P', 'I1P'):
            values = values.where(fasting)
        out[target] = values
    for i in (2, 3):
        out[f'BPXOSY{i}'], out[f'BPXODI{i}'] = out.BPXOSY1, out.BPXODI1
    mortality = RAW / MORTALITY_FILE
    if mortality.exists():
        out = out.merge(read_mortality(mortality), on='SEQN', how='left')
    return out


def read_mortality(path):
    """Public-use Linked Mortality File (fixed width). Positions from the NHANES section of NCHS's
    SAS_ReadInProgramAllSurveys.sas (May 2022). UCOD_LEADING is the NCHS leading-cause recode (1 heart
    disease, 2 cancer, ... 10 all other causes); it has no liver-disease category, so the public file
    supports all-cause, cardiovascular and cancer mortality but not liver-related mortality."""
    spec = [('SEQN', 1, 6), ('ELIGSTAT', 15, 15), ('MORTSTAT', 16, 16), ('UCOD_LEADING', 17, 19),
            ('DIABETES', 20, 20), ('HYPERTEN', 21, 21), ('PERMTH_INT', 43, 45), ('PERMTH_EXM', 46, 48)]
    raw = pd.read_fwf(path, colspecs=[(s - 1, e) for _, s, e in spec], names=[n for n, *_ in spec], dtype=str)
    return raw.apply(pd.to_numeric, errors='coerce')


def variable_map():
    from src.combined_features import COMBINED_FEATURES
    mapped = {target: (var, note) for var, _, target, _, note in VARIABLES if target}
    mapped |= {f'BPXOSY{i}': ('PEPMNK1R', 'Average systolic BP') for i in (2, 3)}
    mapped |= {f'BPXODI{i}': ('PEPMNK5R', 'Average diastolic BP') for i in (2, 3)}
    return pd.DataFrame([{'model_input': c, 'nhanes3_variable': mapped.get(c, ('', ''))[0],
                          'status': 'mapped' if c in mapped else 'not available (treated as missing)',
                          'note': mapped.get(c, ('', ''))[1]} for c in COMBINED_FEATURES])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.download:
        download()
    data = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT, index=False)
    table = variable_map()
    MAP_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(MAP_OUTPUT, index=False)
    summary = {'participants_20_59_gradable': len(data),
               'moderate_severe_share': round(float(data.steatosis_moderate_severe.mean()), 4),
               'model_inputs_mapped': int(table.status.eq('mapped').sum()), 'model_inputs_total': len(table),
               'mortality_linked': 'MORTSTAT' in data,
               'input_coverage': {c: round(float(data[c].notna().mean()), 3) for c in data.columns
                                  if c not in ('SEQN', 'steatosis_moderate_severe')}}
    (MAP_OUTPUT.parent / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'input_coverage'}, indent=2))


if __name__ == '__main__':
    main()

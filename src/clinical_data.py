"""NHANES measurements used by the clinical layers but not by the Layer 2 model.

- LUXSMED, LUXSIQRM: median liver stiffness (kPa) from the same FibroScan exam as CAP, and its
  IQR/median (%); a reading is reliable when IQR/median <= 30% (Boursier et al., J Hepatol 2013).
- RIDEXPRG: pregnancy status at the exam (1 = pregnant).

Read from the raw NHANES files already used to build the cohort (src/build_combined_dataset.py).
"""
import pandas as pd

from src.build_combined_dataset import CYCLES, component_path

STIFFNESS_COLUMNS = ['LUXSMED', 'LUXSIQRM']
RELIABLE_STIFFNESS_IQR_RATIO = 30


def load_extras(download=False):
    """One row per SEQN with stiffness and pregnancy status, for both FibroScan cycles."""
    parts = []
    for cycle in CYCLES:
        lux = pd.read_sas(component_path(cycle, 'LUX', download))[['SEQN'] + STIFFNESS_COLUMNS]
        demo = pd.read_sas(component_path(cycle, 'DEMO', download))[['SEQN', 'RIDEXPRG']]
        parts.append(lux.merge(demo, on='SEQN', how='left', validate='one_to_one'))
    out = pd.concat(parts, ignore_index=True)
    out['SEQN'] = out.SEQN.astype(int)
    return out


def reliable_stiffness(data):
    return (data.LUXSMED.notna() & (data.LUXSIQRM <= RELIABLE_STIFFNESS_IQR_RATIO)).to_numpy()

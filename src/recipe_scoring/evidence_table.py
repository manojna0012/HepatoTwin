"""Read the intervention evidence table that the evidence rules cite."""

import csv
from functools import lru_cache
from pathlib import Path

EVIDENCE_FILE = Path(__file__).resolve().parents[2] / "data" / "evidence" / "intervention_evidence.csv"
STRENGTHS = {"STRONG", "MODERATE", "WEAK / INCONCLUSIVE", "INSUFFICIENT", "GUIDELINE RECOMMENDATION"}


@lru_cache(maxsize=None)
def load_evidence(path=EVIDENCE_FILE):
    """Return {evidence_id: row} and check that every strength label is defined."""
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    table = {}
    for row in rows:
        if row["evidence_strength"] not in STRENGTHS:
            raise ValueError(f"Undefined evidence strength for {row['evidence_id']}: "
                             f"{row['evidence_strength']}")
        if row["evidence_id"] in table:
            raise ValueError(f"Duplicate evidence_id: {row['evidence_id']}")
        table[row["evidence_id"]] = row
    return table


def cite(evidence_ids):
    """Return the strength, finding and source of each cited evidence row."""
    table = load_evidence()
    missing = [e for e in evidence_ids if e not in table]
    if missing:
        raise ValueError(f"Evidence rows not in the table: {missing}")
    return [{"evidence_id": e,
             "evidence_strength": table[e]["evidence_strength"],
             "outcome": table[e]["outcome_measured"],
             "effect_direction": table[e]["effect_direction"],
             "source": table[e]["source"]}
            for e in evidence_ids]

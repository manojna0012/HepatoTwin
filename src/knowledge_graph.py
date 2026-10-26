"""Causal knowledge graph of MASLD (Proposal section 5.3).

A small, hand-written directed graph (data/evidence/causal_graph.csv, 14 links) of the well-known
causal direction of MASLD biology: diet and activity levers act on weight, insulin resistance and
liver fat; liver fat drives injury and fibrosis; CAP, ALT and stiffness only measure them. Each link
names its evidence rows (data/evidence/intervention_evidence.csv) or, where the evidence table has no
row, the paper it rests on.

The graph does two jobs. It restricts the twin's dietary priorities to levers with a causal path to
liver fat, so a recipe criterion never targets a downstream measurement such as ALT or CAP. And it
supplies the chain behind each recommendation (recipe -> lever -> ... -> liver fat) for explanations.
"""
from functools import lru_cache

import pandas as pd

from src.recipe_scoring.evidence_table import load_evidence
from src.train_final_model import ROOT

GRAPH_FILE = ROOT / 'data' / 'evidence' / 'causal_graph.csv'
NODE_TYPES = {
    'energy_intake': 'lever', 'saturated_fat_intake': 'lever', 'added_sugar_intake': 'lever',
    'diet_quality': 'lever', 'physical_activity': 'lever', 'alcohol_intake': 'lever',
    'body_weight': 'mediator', 'insulin_resistance': 'mediator',
    'liver_fat': 'outcome', 'liver_injury': 'outcome', 'fibrosis': 'outcome',
    'CAP': 'measurement', 'ALT': 'measurement', 'liver_stiffness': 'measurement',
}
NODE_LABELS = {
    'energy_intake': 'energy intake', 'saturated_fat_intake': 'saturated fat intake',
    'added_sugar_intake': 'added sugar intake', 'diet_quality': 'Mediterranean-style diet quality',
    'physical_activity': 'physical activity', 'alcohol_intake': 'alcohol intake', 'body_weight': 'body weight',
    'insulin_resistance': 'insulin resistance', 'liver_fat': 'liver fat', 'liver_injury': 'liver injury',
    'fibrosis': 'liver scarring (fibrosis)', 'CAP': 'CAP', 'ALT': 'ALT', 'liver_stiffness': 'liver stiffness',
}
# The Layer 3 priorities (src/recipe_scoring/evidence_rules.py) and the lever each one acts on.
PRIORITY_LEVER = {'weight_management': 'energy_intake', 'limit_saturated_fat': 'saturated_fat_intake',
                  'limit_excess_sugar': 'added_sugar_intake', 'diet_quality': 'diet_quality'}
STRENGTH_ORDER = ('STRONG', 'MODERATE', 'GUIDELINE RECOMMENDATION', 'WEAK / INCONCLUSIVE',
                  'OBSERVATIONAL ASSOCIATION')


@lru_cache(maxsize=None)
def load_graph(path=GRAPH_FILE):
    edges = pd.read_csv(path, dtype=str, keep_default_na=False)
    unknown = (set(edges.source) | set(edges.target)) - set(NODE_TYPES)
    if unknown:
        raise ValueError(f'Unknown nodes in the causal graph: {sorted(unknown)}')
    known_ids = set(load_evidence())
    for ids in edges.evidence_ids:
        for eid in filter(None, ids.split(';')):
            if eid not in known_ids:
                raise ValueError(f'Causal graph cites unknown evidence row {eid}')
    graph = {}
    for row in edges.itertuples(index=False):
        graph.setdefault(row.source, []).append(row._asdict())
    if _has_cycle(graph):
        raise ValueError('The causal graph must be acyclic.')
    return graph


def _has_cycle(graph):
    state = {}

    def visit(node):
        if state.get(node) == 1:
            return True
        if state.get(node) == 2:
            return False
        state[node] = 1
        found = any(visit(edge['target']) for edge in graph.get(node, []))
        state[node] = 2
        return found
    return any(visit(node) for node in list(graph))


def paths(source, target='liver_fat', graph=None):
    """Every directed path from source to target, as lists of edges."""
    graph = load_graph() if graph is None else graph
    if source == target:
        return [[]]
    found = []
    for edge in graph.get(source, []):
        for rest in paths(edge['target'], target, graph):
            found.append([edge] + rest)
    return found


def actionable(node, graph=None):
    """A node may be targeted by a recipe criterion only if it is a lever with a path to liver fat."""
    return NODE_TYPES.get(node) == 'lever' and bool(paths(node, 'liver_fat', graph))


def evidence_strength(evidence_ids):
    """Strongest label among the cited evidence rows (None when the link rests on a paper only)."""
    if not evidence_ids:
        return None
    table = load_evidence()
    labels = {table[e]['evidence_strength'] for e in evidence_ids}
    return next((s for s in STRENGTH_ORDER if s in labels), None)


def chain(priority_id, graph=None):
    """The shortest causal chain from a priority's lever to liver fat, with evidence per link."""
    lever = PRIORITY_LEVER.get(priority_id)
    if lever is None or not actionable(lever, graph):
        return None
    best = min(paths(lever, 'liver_fat', graph), key=len)
    links = []
    for edge in best:
        ids = [e for e in edge['evidence_ids'].split(';') if e]
        links.append({'edge_id': edge['edge_id'], 'from': NODE_LABELS[edge['source']],
                      'to': NODE_LABELS[edge['target']], 'direction': edge['direction'],
                      'relation': edge['relation'], 'evidence_ids': ids,
                      'evidence_strength': evidence_strength(ids), 'reference': edge['source_reference']})
    text = ' -> '.join([NODE_LABELS[lever]] + [NODE_LABELS[e['target']] for e in best])
    return {'priority_id': priority_id, 'lever': lever, 'path': text, 'links': links}

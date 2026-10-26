import pandas as pd
import pytest

from src.knowledge_graph import NODE_TYPES, PRIORITY_LEVER, actionable, chain, load_graph, paths
from src.recipe_scoring.recipe_explanation import explain_recipe
from src.twin_summary import check, facts, summarise, template_summary

TWIN = {
    'current_state': {'predicted_cap': 313.7, 'predicted_grade': 'S3', 'p_cap248': 0.96, 'p_cap280': 0.78},
    'uncertainty': {'cap_interval_90': (247.0, 380.3)},
    'clinical': {'liver_label': {'label': 'MASLD'},
                 'fibrosis': {'fib4': 0.46, 'category': 'low'},
                 'guardrails': {'notices': [], 'block_recipes': False}},
    'nutritional_priorities': [{'title': 'Prefer recipes lower in saturated fat', 'priority': 'high',
                                'supporting_evidence': [{'evidence_id': 'LUU2018-SAT'}]}],
    'ranked_recipes': [{'title': 'Bean Salad', 'criteria': [{'label': 'Saturated fat', 'value': 2.1}]}],
}


def test_graph_is_acyclic_and_cites_only_known_evidence():
    graph = load_graph()
    assert sum(len(v) for v in graph.values()) == 14
    assert set(NODE_TYPES) >= {e['target'] for edges in graph.values() for e in edges}


def test_every_priority_acts_on_a_lever_with_a_path_to_liver_fat():
    for priority, lever in PRIORITY_LEVER.items():
        assert actionable(lever)
        assert chain(priority)['path'].endswith('liver fat')
    for measurement in ('CAP', 'ALT', 'liver_stiffness'):
        assert not actionable(measurement)
    assert paths('liver_fat', 'CAP') and not paths('CAP', 'liver_fat')


def test_weight_chain_carries_the_strong_weight_loss_evidence():
    links = chain('weight_management')['links']
    assert [l['from'] for l in links] == ['energy intake', 'body weight']
    assert links[1]['evidence_strength'] == 'STRONG' and 'VG2015-WL5' in links[1]['evidence_ids']


def test_recipe_explanation_traces_each_criterion():
    item = {'recipe_id': '1', 'title': 'Bean Salad', 'rank': 1, 'scores': {'cuisine': 80.0, 'feasibility': 70.0},
            'criteria': [{'priority_id': 'limit_saturated_fat', 'priority': 'high', 'label': 'Saturated fat',
                          'value': 2.1, 'reference': '<= 10', 'score': 100, 'evidence': ['LUU2018-SAT [MODERATE]']}]}
    recipe = {'total_time': 20, 'ingredients': ['beans', 'onion'], 'instructions': ['Mix.', 'Serve.']}
    out = explain_recipe(item, recipe, {'category': 'ingredients_only', 'typicality': 60.0})
    reason = out['reasons'][0]
    assert reason['meets_reference'] and reason['causal_path'] == 'saturated fat intake -> liver fat'
    assert 'LUU2018-SAT' in out['text'] and 'judged on its ingredients' in out['text']
    assert out['context']['techniques'] == [] and out['context']['steps'] == 2


def test_template_summary_is_faithful():
    result = summarise(TWIN)
    assert result['written_by'] == 'template' and result['check']['faithful']
    assert '314' in result['text'] and '247' in result['text'] and 'LUU2018-SAT' in result['text']


def test_generator_text_with_an_invented_number_or_citation_falls_back_to_the_template():
    made_up = summarise(TWIN, generator=lambda f: 'Your CAP is 290 dB/m (see ABC2020-XYZ).')
    assert made_up['written_by'] == 'template' and 'faithfulness' in made_up['fallback_reason']
    assert made_up['rejected_check']['unsupported_numbers'] == ['290']
    assert made_up['rejected_check']['unsupported_citations'] == ['ABC2020-XYZ']
    good = summarise(TWIN, generator=lambda f: f'Your CAP is about {f["predicted_cap"]} dB/m (LUU2018-SAT).')
    assert good['written_by'] == 'generator' and good['check']['faithful']
    broken = summarise(TWIN, generator=lambda f: 1 / 0)
    assert broken['written_by'] == 'template' and 'generator failed' in broken['fallback_reason']


def test_fibroscan_and_young_note_reach_the_summary_and_pass_the_check():
    scan = dict(TWIN, clinical=dict(TWIN['clinical'], fibrosis={'fib4': None, 'liver_stiffness_kpa': 6.2,
                                                                'category': 'low', 'note': None}))
    text = summarise(scan)
    assert 'FibroScan liver stiffness is 6.2 kPa' in text['text'] and text['check']['faithful']
    young = dict(TWIN, clinical=dict(TWIN['clinical'], fibrosis={'fib4': 0.4, 'category': 'low',
                                                                 'note': 'FIB-4 is less reliable under age 35.'}))
    text = summarise(young)
    assert 'under age 35' in text['text'] and text['check']['faithful']


def test_withheld_recipes_are_not_summarised():
    blocked = dict(TWIN, clinical=dict(TWIN['clinical'], guardrails={
        'notices': [{'message': 'High fibrosis risk (FIB-4 above 2.67). Please seek a clinical assessment.'}],
        'block_recipes': True}))
    f = facts(blocked)
    assert f['recipes'] == [] and 'Bean Salad' not in template_summary(f)
    assert check(template_summary(f), f)['faithful']


def test_cap_explanation_adds_up_and_interval_covers_prediction():
    from src.explain_patient import explain_cap
    from src.predict_cap import predict_patients
    from src.uncertainty import cap_interval
    data = pd.read_csv('data/processed/nhanes_combined.csv', nrows=50)
    row = data.iloc[0].to_dict()
    predicted = float(predict_patients(pd.DataFrame([row])).iloc[0])
    explanation = explain_cap(row, top=500)
    total = explanation['average_cap'] + sum(r['contribution_db_m'] for r in explanation['raising'] + explanation['lowering'])
    assert explanation['predicted_cap'] == pytest.approx(predicted, abs=0.1)
    assert total == pytest.approx(predicted, abs=0.5)
    low, high = cap_interval(predicted)
    assert low <= predicted <= high and high - low <= 2 * 67

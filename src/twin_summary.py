"""Grounded plain-language summary of the twin's output (Proposal section 17.7).

The summary is written only from a set of facts the twin computed (facts()), with every number already
rounded as it will be shown. A template writes the summary now; a language model can be plugged in
later through `generator`, a function that takes the facts and returns text. Whatever writes it, an
automatic check confirms that every number and every evidence citation in the text appears in the
facts; if the check fails, the template summary is shown instead and the reason is recorded.

The summary is written for the patient, so it gives the chances of liver fat and a plain verdict
taken from the twin's own yes/no decisions (patient_state), not the CAP estimate: at R^2 0.56 a
single CAP number and its 90% range (about +/- 67 dB/m) would suggest more precision than the
model has. The CAP estimate and its range stay in the twin's output for the clinician view.

Written for people without medical training (changes after the team's review, 8 November 2026): evidence codes
stay in the clinician view and the patient reads how many studies back each priority; priorities and liver
scarring are explained in plain words; the published cut-offs are shown with their source (CAP 248 and 280
dB/m, Karlas et al. 2017; FIB-4 1.3 and 2.67, EASL-EASD-EASO 2024; stiffness 8 and 12 kPa), and a score
within 10% of a cut-off is called close to that boundary. Notices about changing diet (under 18,
glucose-lowering medicine, pregnancy) appear only when a diet change is suggested, and the FIB-4 under-35
note only when liver fat is likely: guidelines look for scarring in people with fatty liver.
"""
import re

CITATION = re.compile(r'\b[A-Z]{2,5}\d{4}-[A-Z0-9][A-Z0-9-]*\b')
# Digits joined to a name by a hyphen (FIB-4, VG2015-WL7-10) are part of the name, not a value.
NUMBER = re.compile(r'(?<![\w.-])\d+(?:\.\d+)?')
VERDICT = {'a_lot': 'Your results suggest you likely have a lot of liver fat.',
           'some': 'Your results suggest you likely have some liver fat.',
           'little': 'Your results show little sign of liver fat.'}
LABEL_WORDS = {'MASLD': 'metabolic dysfunction-associated steatotic liver disease (MASLD)',
               'MetALD': 'MASLD with increased alcohol intake (MetALD)',
               'ALD': 'liver fat that is likely related to alcohol',
               'cryptogenic_SLD': 'liver fat without a clear metabolic cause',
               'no_steatosis_indicated': 'no sign of fatty liver', 'unknown': 'an unclear liver status'}


CAP_CUTOFFS = {'some': 248, 'a_lot': 280, 'source': 'Karlas et al. 2017'}
FIB4_CUTOFFS = {'low': 1.3, 'high': 2.67, 'source': 'EASL-EASD-EASO 2024'}
STIFFNESS_CUTOFFS = {'low': 8, 'high': 12, 'source': 'EASL-EASD-EASO 2024'}
BORDER = 0.10                       # within 10% of a cut-off = close to that boundary
DIET_NOTICES = {'under_18', 'glucose_lowering_medicine', 'pregnant'}
PLAIN_PRIORITY = [('weight', 'losing some weight'), ('dietary quality', 'eating a healthier diet overall'),
                  ('added sugar', 'less added sugar'),
                  ('saturated fat', 'less saturated fat, the kind in butter, fatty meat and cheese'),
                  ('processed', 'food made from fresh or less processed ingredients'),
                  ('glycaemic load', 'meals that raise blood sugar less, called a lower glycaemic load')]


def plain_priority(title):
    low = title.lower()
    return next((words for key, words in PLAIN_PRIORITY if key in low), low)


def pct_text(p):
    """A chance as shown: never 0% or 100%, which would claim certainty the model does not have."""
    value = round(100 * p)
    return 'more than 99%' if value >= 100 else 'less than 1%' if value <= 0 else f'{value}%'


def near_boundary(score, cutoffs):
    """The cut-off a score lies within BORDER of (relative), or None."""
    if score is None:
        return None
    return next((c for c in (cutoffs['low'], cutoffs['high']) if abs(score - c) <= BORDER * c), None)


def facts(twin, top=3):
    """The numbers and citations a summary may use, rounded as they will be shown."""
    state = twin['current_state']
    out = {'p_cap248_pct': round(100 * state['p_cap248']), 'p_cap280_pct': round(100 * state['p_cap280']),
           'p_cap248_text': pct_text(state['p_cap248']), 'p_cap280_text': pct_text(state['p_cap280']),
           'verdict': verdict(twin.get('patient_state') or {}), 'cap_cutoffs': CAP_CUTOFFS}
    clinical = twin.get('clinical') or {}
    if clinical.get('liver_label'):
        out['label'] = clinical['liver_label']['label']
    fat_likely = out['verdict'] in ('some', 'a_lot')
    if clinical.get('fibrosis'):
        fib = clinical['fibrosis']
        out['fib4'] = fib.get('fib4')
        out['liver_stiffness_kpa'] = fib.get('liver_stiffness_kpa')
        out['fibrosis_category'] = fib['category']
        note = fib.get('note') or ''
        if not fat_likely and 'under age 35' in note:     # no scarring check is needed without fatty liver
            note = ' '.join(x for x in re.split(r'(?<=\.)\s+', note) if 'under age 35' not in x and 'FibroScan' not in x)
        out['fibrosis_note'] = note or None
        scan = out['liver_stiffness_kpa'] is not None
        cutoffs = STIFFNESS_CUTOFFS if scan else FIB4_CUTOFFS
        out['fibrosis_cutoffs'] = cutoffs
        out['fibrosis_near'] = near_boundary(out['liver_stiffness_kpa'] if scan else out['fib4'], cutoffs)
    out['blocked'] = bool((clinical.get('guardrails') or {}).get('block_recipes'))
    out['priorities'] = [{'title': p['title'], 'words': plain_priority(p['title']), 'level': p['priority'],
                          'evidence_ids': [e['evidence_id'] for e in p.get('supporting_evidence', [])],
                          'studies': len(p.get('supporting_evidence', []))}
                         for p in twin.get('nutritional_priorities', [])]
    out['recipes'] = [] if out['blocked'] else [
        {'title': r['title'], 'values': {c['label']: c['value'] for c in r.get('criteria', [])}}
        for r in twin.get('ranked_recipes', [])[:top]]
    diet_change = bool(out['priorities'] or out['recipes'])
    out['notices'] = [n['message'] for n in (clinical.get('guardrails') or {}).get('notices', [])
                      if diet_change or n.get('id') not in DIET_NOTICES]
    return out


def verdict(state):
    """Plain verdict from the twin's decisions on the two calibrated probabilities (None if unknown)."""
    if state.get('steatosis_moderate_severe_likely') == 1:
        return 'a_lot'
    if state.get('steatosis_likely') == 1:
        return 'some'
    if state.get('steatosis_likely') == 0:
        return 'little'
    return None


def article(word):
    """'a low', 'an indeterminate'."""
    return f'{"an" if str(word)[:1].lower() in "aeiou" else "a"} {word}'


def template_summary(f):
    lines = [VERDICT[f['verdict']]] if f.get('verdict') else []
    cap = f['cap_cutoffs']
    lines.append(f'The chance that you have at least some liver fat is {f["p_cap248_text"]}, and a lot of liver fat '
                 f'{f["p_cap280_text"]}. On a FibroScan, doctors call a reading of {cap["some"]} dB/m or more "some fat" '
                 f'and {cap["a_lot"]} dB/m or more "a lot" ({cap["source"]}).')
    if 'label' in f:
        lines.append(f'Taking the chance of liver fat into account, this pattern fits '
                     f'{LABEL_WORDS.get(f["label"], f["label"])}. This is a research label, not a diagnosis.')
    if f.get('liver_stiffness_kpa') is not None or f.get('fib4') is not None:
        c = f['fibrosis_cutoffs']
        lines.append('Liver scarring (fibrosis) is long-term damage that makes the liver stiffer.')
        if f.get('liver_stiffness_kpa') is not None:
            lines.append(f'Your FibroScan liver stiffness is {f["liver_stiffness_kpa"]} kPa, {article(f["fibrosis_category"])} '
                         f'scarring-risk result: below {c["low"]} kPa is low, {c["low"]} to {c["high"]} kPa indeterminate and '
                         f'{c["high"]} kPa or more high ({c["source"]}).')
        else:
            lines.append(f'Your scarring score (FIB-4) is {f["fib4"]}, {article(f["fibrosis_category"])} result: below '
                         f'{c["low"]} is low, {c["low"]} to {c["high"]} indeterminate and above {c["high"]} high ({c["source"]}).')
        if f.get('fibrosis_near') is not None:
            c = f['fibrosis_cutoffs']
            sides = 'low and indeterminate' if f['fibrosis_near'] == c['low'] else 'indeterminate and high'
            lines.append(f'Your result is close to the boundary between {sides} ({f["fibrosis_near"]}).')
    if f.get('fibrosis_note'):
        lines.append(f['fibrosis_note'])
    lines.extend(f['notices'])
    if f['priorities']:
        parts = [f'{p["words"]} ({p["level"]} priority' + (f', backed by {p["studies"]} '
                 f'{"study" if p["studies"] == 1 else "studies"})' if p['studies'] else ')') for p in f['priorities']]
        lines.append('What matters most for you: ' + '; '.join(parts) + '.')
    if f['recipes']:
        lines.append('Recipes that fit these priorities: ' + ', '.join(r['title'] for r in f['recipes']) + '.')
    return ' '.join(lines)


def _allowed(value, numbers, citations):
    if isinstance(value, dict):
        for v in value.values():
            _allowed(v, numbers, citations)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _allowed(v, numbers, citations)
    elif isinstance(value, bool) or value is None:
        return
    elif isinstance(value, (int, float)):
        numbers.add(f'{value:g}' if isinstance(value, float) else str(value))
        numbers.add(str(value))
    elif isinstance(value, str):
        numbers.update(NUMBER.findall(value))
        citations.update(CITATION.findall(value))


def check(text, f):
    """Numbers and citations in `text` that do not appear in the facts (empty lists = faithful)."""
    numbers, citations = set(), set()
    _allowed(f, numbers, citations)
    stray_numbers = [n for n in NUMBER.findall(text) if n not in numbers and f'{float(n):g}' not in numbers]
    stray_citations = [c for c in CITATION.findall(text) if c not in citations]
    return {'faithful': not stray_numbers and not stray_citations,
            'unsupported_numbers': stray_numbers, 'unsupported_citations': stray_citations}


def summarise(twin, generator=None):
    """Summary text, who wrote it, and the result of the faithfulness check."""
    f = facts(twin)
    template = template_summary(f)
    if generator is None:
        return {'text': template, 'written_by': 'template', 'check': check(template, f)}
    try:
        text = generator(f)
    except Exception as error:      # a failing model must never block the summary
        return {'text': template, 'written_by': 'template', 'check': check(template, f),
                'fallback_reason': f'generator failed: {error}'}
    result = check(text, f)
    if result['faithful']:
        return {'text': text, 'written_by': 'generator', 'check': result}
    return {'text': template, 'written_by': 'template', 'check': check(template, f),
            'fallback_reason': 'generator text failed the faithfulness check', 'rejected_check': result}

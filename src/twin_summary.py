"""Grounded plain-language summary of the twin's output (Proposal section 17.7).

The summary is written only from a set of facts the twin computed (facts()), with every number already
rounded as it will be shown. A template writes the summary now; a language model can be plugged in
later through `generator`, a function that takes the facts and returns text. Whatever writes it, an
automatic check confirms that every number and every evidence citation in the text appears in the
facts; if the check fails, the template summary is shown instead and the reason is recorded.
"""
import re

CITATION = re.compile(r'\b[A-Z]{2,5}\d{4}-[A-Z0-9][A-Z0-9-]*\b')
# Digits joined to a name by a hyphen (FIB-4, VG2015-WL7-10) are part of the name, not a value.
NUMBER = re.compile(r'(?<![\w.-])\d+(?:\.\d+)?')
GRADE_WORDS = {'S0': 'no clear sign of liver fat from this value alone', 'S1': 'mild liver fat', 'S2': 'moderate liver fat',
               'S3': 'a lot of liver fat'}
LABEL_WORDS = {'MASLD': 'metabolic dysfunction-associated steatotic liver disease (MASLD)',
               'MetALD': 'MASLD with increased alcohol intake (MetALD)',
               'ALD': 'liver fat that is likely related to alcohol',
               'cryptogenic_SLD': 'liver fat without a clear metabolic cause',
               'no_steatosis_indicated': 'no sign of fatty liver', 'unknown': 'an unclear liver status'}


def facts(twin, top=3):
    """The numbers and citations a summary may use, rounded as they will be shown."""
    state = twin['current_state']
    out = {'predicted_cap': round(state['predicted_cap']), 'grade': state['predicted_grade'],
           'p_cap248_pct': round(100 * state['p_cap248']), 'p_cap280_pct': round(100 * state['p_cap280'])}
    uncertainty = twin.get('uncertainty')
    if uncertainty:
        out['cap_low'], out['cap_high'] = (round(v) for v in uncertainty['cap_interval_90'])
    clinical = twin.get('clinical') or {}
    if clinical.get('liver_label'):
        out['label'] = clinical['liver_label']['label']
    if clinical.get('fibrosis'):
        out['fib4'] = clinical['fibrosis']['fib4']
        out['liver_stiffness_kpa'] = clinical['fibrosis'].get('liver_stiffness_kpa')
        out['fibrosis_category'] = clinical['fibrosis']['category']
        out['fibrosis_note'] = clinical['fibrosis'].get('note')
    out['notices'] = [n['message'] for n in (clinical.get('guardrails') or {}).get('notices', [])]
    out['blocked'] = bool((clinical.get('guardrails') or {}).get('block_recipes'))
    out['priorities'] = [{'title': p['title'], 'level': p['priority'],
                          'evidence_ids': [e['evidence_id'] for e in p.get('supporting_evidence', [])]}
                         for p in twin.get('nutritional_priorities', [])]
    out['recipes'] = [] if out['blocked'] else [
        {'title': r['title'], 'values': {c['label']: c['value'] for c in r.get('criteria', [])}}
        for r in twin.get('ranked_recipes', [])[:top]]
    return out


def template_summary(f):
    lines = [f'Your liver estimate: CAP about {f["predicted_cap"]} dB/m, which suggests {GRADE_WORDS.get(f["grade"], f["grade"])}.']
    if 'cap_low' in f:
        lines.append(f'Because no estimate is exact, your CAP is likely between {f["cap_low"]} and {f["cap_high"]} dB/m.')
    lines.append(f'The chance of at least mild liver fat is {f["p_cap248_pct"]}%, and of a lot of liver fat {f["p_cap280_pct"]}%.')
    if 'label' in f:
        lines.append(f'Taking the chance of liver fat into account, this pattern fits '
                     f'{LABEL_WORDS.get(f["label"], f["label"])}. This is a research label, not a diagnosis.')
    if f.get('liver_stiffness_kpa') is not None:
        lines.append(f'Your FibroScan liver stiffness is {f["liver_stiffness_kpa"]} kPa, a {f["fibrosis_category"]} '
                     'fibrosis-risk result.')
    elif f.get('fib4') is not None:
        lines.append(f'Your liver-scarring score (FIB-4) is {f["fib4"]}, a {f["fibrosis_category"]} result.')
    if f.get('fibrosis_note'):
        lines.append(f['fibrosis_note'])
    lines.extend(f['notices'])
    if f['priorities']:
        parts = [f'{p["title"].lower()} ({p["level"]} priority; evidence {", ".join(p["evidence_ids"])})'
                 if p['evidence_ids'] else f'{p["title"].lower()} ({p["level"]} priority)' for p in f['priorities']]
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

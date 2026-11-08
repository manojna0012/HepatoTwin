"""Evaluation of the plain-language summaries (Proposal sections 17.7 and 12.4).

For 300 held-out test patients inside the model's scope (random, fixed seed), the twin runs end to end and
its summary is scored on:

1. Faithfulness: the share of summaries whose every number and evidence citation appears in the twin's
   facts (twin_summary.check).
2. The check itself, by planting errors in each summary as a language model might make them: a number
   changed, a number added, a citation invented (each should be caught), and a harmless rewording with no
   number or citation changed (should pass). A wrong word with no number in it ("low" changed to "high")
   is planted too: the check does not read words, so this measures its blind spot.
3. Readability: Flesch Reading Ease and Flesch-Kincaid grade (syllables counted by vowel groups), words
   and sentences.

A random sample of 10 summaries is written out for the team to read (summaries_for_review.md).

    .venv/Scripts/python.exe -m src.summary_evaluation
"""
import json
import re

import numpy as np
import pandas as pd

from src.train_combined_model import in_scope
from src.train_final_model import ROOT
from src.evaluate_ranking import with_diet
from src.twin_pipeline import load_recipes, run_twin
from src.twin_summary import LABEL_WORDS, NUMBER, check, facts

DATA = ROOT / 'data' / 'processed' / 'nhanes_combined.csv'
PREDICTIONS = ROOT / 'data' / 'results' / 'combined' / 'test_predictions.csv'
OUTPUT = ROOT / 'data' / 'results' / 'summaries'
PATIENTS = 300
REVIEW = 10
SEED = 7


def syllables(word):
    word = word.lower().strip(".,;:()'\"")
    groups = re.findall(r'[aeiouy]+', word)
    count = len(groups) - (1 if word.endswith('e') and len(groups) > 1 and not word.endswith('le') else 0)
    return max(1, count)


def readability(text):
    sentences = max(1, len(re.findall(r'[.!?](?:\s|$)', text)))
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    n = max(1, len(words))
    syl = sum(syllables(w) for w in words)
    return {'words': len(words), 'sentences': sentences,
            'flesch_reading_ease': 206.835 - 1.015 * n / sentences - 84.6 * syl / n,
            'flesch_kincaid_grade': 0.39 * n / sentences + 11.8 * syl / n - 15.59}


def planted(text):
    """Versions of a summary with one planted error each (None when the summary offers no place for it)."""
    numbers = list(NUMBER.finditer(text))
    out = {}
    if numbers:
        m = numbers[0]
        value = m.group()
        changed = str(int(float(value)) + 7) if '.' not in value else f'{float(value) + 0.7:.2f}'
        out['number changed'] = text[:m.start()] + changed + text[m.end():]
    out['number added'] = text + ' This lowers your risk by 37%.'
    out['citation invented'] = text + ' (evidence ZZ2099-FAKE1)'
    out['harmless rewording'] = text.replace('suggest', 'indicate').replace('Recipes that fit', 'Recipes matching')
    if re.search(r'\b(low|high)\b', text):
        out['wrong word, no number'] = re.sub(r'\b(low|high)\b', lambda w: 'high' if w.group() == 'low' else 'low', text, count=1)
    return out


def main():
    data = pd.read_csv(DATA)
    test = data[data.SEQN.isin(pd.read_csv(PREDICTIONS).SEQN)]
    test = test[in_scope(test)].sample(PATIENTS, random_state=SEED)
    test = with_diet(test.reset_index(drop=True))            # the 24-hour recall, for personal fit
    recipes = load_recipes()
    rows, planted_rows, twins = [], [], {}
    for _, patient in test.iterrows():
        twin = run_twin(patient.to_dict(), recipes)
        twins[int(patient.SEQN)] = twin
        summary, f = twin['summary'], facts(twin)
        rows.append({'SEQN': int(patient.SEQN), 'text': summary['text'], 'faithful': summary['check']['faithful'],
                     'written_by': summary['written_by'], **readability(summary['text'])})
        for kind, text in planted(summary['text']).items():
            planted_rows.append({'SEQN': int(patient.SEQN), 'error': kind, 'caught': not check(text, f)['faithful']})
    frame, plants = pd.DataFrame(rows), pd.DataFrame(planted_rows)
    caught = plants.groupby('error').caught.agg(['mean', 'size'])
    summary = {'patients': len(frame), 'faithful_share': round(float(frame.faithful.mean()), 3),
               'planted_errors': {k: {'caught_share': round(float(v['mean']), 3), 'n': int(v['size'])} for k, v in caught.iterrows()},
               'readability': {k: {'median': round(float(frame[k].median()), 1), 'iqr': [round(float(frame[k].quantile(q)), 1) for q in (.25, .75)]}
                               for k in ('words', 'sentences', 'flesch_reading_ease', 'flesch_kincaid_grade')}}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / 'summaries.csv', index=False)
    plants.to_csv(OUTPUT / 'planted_errors.csv', index=False)
    (OUTPUT / 'summary_evaluation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    sample = frame.sample(REVIEW, random_state=SEED)
    write_review(sample, {int(seqn): twins[int(seqn)] for seqn in sample.SEQN})
    print(json.dumps(summary, indent=1))


def expected_words(twin):
    """The words the summary should use, worked out again from the twin's numbers and the stated rules."""
    p248, p280 = twin['current_state']['p_cap248'], twin['current_state']['p_cap280']
    t = twin['layer2']['thresholds']
    verdict = ('a lot of liver fat' if p280 >= t['cap280'] else 'some liver fat' if p248 >= t['cap248']
               else 'little sign of liver fat')
    lines = [f'- Chance of at least mild fat (CAP >= 248): **{100 * p248:.0f}%**; of a lot of fat (CAP >= 280): **{100 * p280:.0f}%**.',
             f'- Rule: "a lot" if the second chance is at least {100 * t["cap280"]:.1f}%; otherwise "some" if the first is at '
             f'least {100 * t["cap248"]:.1f}%; otherwise "little". Expected words: **{verdict}**.']
    fib = (twin.get('clinical') or {}).get('fibrosis') or {}
    if fib.get('liver_stiffness_kpa') is not None:
        lines.append(f'- FibroScan stiffness **{fib["liver_stiffness_kpa"]} kPa** (measured): expected **{fib["category"]}** result.')
    elif fib.get('fib4') is not None:
        score = fib['fib4']
        word = 'low' if score < 1.3 else 'high' if score > 2.67 else 'indeterminate'
        lines.append(f'- FIB-4 **{score}**. Rule: low below 1.3, indeterminate 1.3 to 2.67, high above 2.67. '
                     f'Expected word: **{word}**.')
    label = ((twin.get('clinical') or {}).get('liver_label') or {}).get('label')
    if label:
        lines.append(f'- Liver label from the clinical rules: {label}, which the summary should call '
                     f'**"{LABEL_WORDS.get(label, label)}"**.')
    return lines


def write_review(sample, twins):
    lines = ['# Summaries for team review', '',
             "For each summary: compare its words with the box under it (worked out again from the twin's numbers and",
             'the rules), then tick. Write a short note under any summary that is wrong, unclear or alarming.', '']
    for _, r in sample.iterrows():
        lines += [f'## Patient {r.SEQN}', '', r.text, '', '**What to check it against**', '', *expected_words(twins[int(r.SEQN)]),
                  '', '- [ ] True  - [ ] Clear  - [ ] Nothing alarming or missing', '', 'Notes:', '']
    (OUTPUT / 'summaries_for_review.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()

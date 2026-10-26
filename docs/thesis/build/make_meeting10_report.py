"""Build the Meeting 10 progress report in the layout of the user's Meeting 9 report."""
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

OUT = sys.argv[1]
FONT = 'Times New Roman'
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
for side in ('left_margin', 'right_margin'):
    setattr(sec, side, Cm(2.54))
sec.top_margin, sec.bottom_margin = Cm(2.54), Cm(2.54)

normal = doc.styles['Normal']
normal.font.name = FONT
normal.font.size = Pt(11)
normal.element.rPr.rFonts.set(qn('w:eastAsia'), FONT)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.15
for name, size in (('Heading 1', 18), ('Heading 2', 13)):
    st = doc.styles[name]
    st.font.name, st.font.size, st.font.bold = FONT, Pt(size), True
    st.font.color.rgb = RGBColor(0, 0, 0)
    fonts = st.element.rPr.rFonts
    for attr in ('w:asciiTheme', 'w:hAnsiTheme', 'w:eastAsiaTheme', 'w:cstheme'):
        fonts.attrib.pop(qn(attr), None)
    for attr in ('w:ascii', 'w:hAnsi', 'w:eastAsia', 'w:cs'):
        fonts.set(qn(attr), FONT)
    st.paragraph_format.space_before = Pt(14 if name == 'Heading 1' else 10)
    st.paragraph_format.space_after = Pt(6)


def para(text='', bold_lead=None, align=None, size=None, bold=False, italic=False, after=None):
    p = doc.add_paragraph()
    if bold_lead:
        r = p.add_run(bold_lead)
        r.bold = True
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if size:
        for run in p.runs:
            run.font.size = Pt(size)
    if align == 'center':
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == 'justify':
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def body(text):
    return para(text, align='justify')


def bullets(items):
    for item in items:
        p = doc.add_paragraph(style='List Bullet')
        if isinstance(item, tuple):
            r = p.add_run(item[0])
            r.bold = True
            p.add_run(item[1])
        else:
            p.add_run(item)
        p.paragraph_format.space_after = Pt(2)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


def numbered(items):
    for n, item in enumerate(items, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.95)
        p.paragraph_format.first_line_indent = Cm(-0.6)
        p.add_run(f'{n}.	')
        p.paragraph_format.tab_stops.add_tab_stop(Cm(0.95))
        r = p.add_run(item[0])
        r.bold = True
        p.add_run(item[1])
        p.paragraph_format.space_after = Pt(2)


def heading(text, level):
    doc.add_heading(text, level=level)


def sub(text):
    p = para(text, bold=True, after=3)
    p.paragraph_format.space_before = Pt(6)


def rule():
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    border = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    for k, v in (('w:val', 'single'), ('w:sz', '6'), ('w:space', '1'), ('w:color', '999999')):
        bottom.set(qn(k), v)
    border.append(bottom)
    pPr.append(border)


def table(caption, header, rows, widths=None, numeric=()):
    cap = para(caption, bold=True, after=4)
    cap.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(header))
    t.style = 'Table Grid'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ''
        run = cell.paragraphs[0].add_run(h)
        run.bold = True
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ''
            cells[i].paragraphs[0].add_run(str(v))
            if i in numeric:
                cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for r in t.rows:
        for c in r.cells:
            for p in c.paragraphs:
                p.paragraph_format.space_after = Pt(2)
                for run in p.runs:
                    run.font.size = Pt(10)
    for i, r in enumerate(t.rows):
        trPr = r._tr.get_or_add_trPr()
        trPr.append(OxmlElement('w:cantSplit'))
        if i == 0:
            trPr.append(OxmlElement('w:tblHeader'))
        else:
            for c in r.cells:
                c.paragraphs[0].paragraph_format.keep_with_next = i < 2
    for c in t.rows[0].cells:
        c.paragraphs[0].paragraph_format.keep_with_next = True
    if widths:
        for r in t.rows:
            for c, w in zip(r.cells, widths):
                c.width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


# ---------------------------------------------------------------- title block
para('HepatoTwin', align='center', size=24, bold=True, after=4)
para('Clinical Safety, Explanations, In-Silico Evaluation and Thesis Writing', align='center', size=11.5, after=10)
para('Progress Report — Meeting 10', align='center', bold=True, after=0)
para('Date: 30th October 2026', align='center', bold=True, after=0)
rule()

# ---------------------------------------------------------------- results read from the repository
import json as _json
from pathlib import Path as _Path

import pandas as _pd

REPO = _Path(r'C:/Users/subas/OneDrive/Desktop/HepatoTwin')
SCRATCH = REPO / 'docs' / 'thesis' / 'build' / 'layer4_comparison'
fib = _pd.read_csv(REPO / 'data/results/fibrosis/fib4_validation.csv')
fib_summary = _json.loads((REPO / 'data/results/fibrosis/fib4_summary.json').read_text())
labels = _pd.read_csv(REPO / 'data/results/clinical/liver_label_summary.csv')
notices = _pd.read_csv(REPO / 'data/results/clinical/liver_labels.csv')
conformal = _json.loads((REPO / 'data/results/combined/uncertainty/conformal.json').read_text())
n3 = _json.loads((REPO / 'data/results/nhanes3/summary.json').read_text())
before = _pd.read_csv(SCRATCH / 'layer4_meeting9/method_comparison.csv', index_col=0)
blend = _pd.read_csv(SCRATCH / 'layer4_blend50/method_comparison.csv', index_col=0)
gap_avg = _pd.read_csv(SCRATCH / 'layer4_gapfill_avg/method_comparison.csv', index_col=0)
after = _pd.read_csv(REPO / 'data/results/layer4/method_comparison.csv', index_col=0)
DEFAULT = 'Multi-criteria, health-gated (proposed)'


def pct(x):
    return f'{100 * x:.1f}%'


def f3(x):
    return f'{x:.3f}'


def share(adult, label):
    row = labels[(labels.adult == adult) & (labels.label == label)]
    return pct(float(row.share.iloc[0])) if len(row) else '0%'


# ---------------------------------------------------------------- 1
heading('1. Project Objective', 1)
body('Meeting 10 covered the two tasks of the weekly action plan, the first part of the in-silico evaluation, and '
     'thesis writing:')
numbered([
    ('Task 1, twin state and evidence: ', 'fibrosis risk and a referral pathway, disease subtype and safety '
     'guardrails, per-patient explanations and uncertainty, a causal knowledge graph, NHANES III data for external '
     'validation, and the virtual-patient simulator (in-silico evaluation, part 1).'),
    ('Task 2, recipe ranking and explanations: ', 'recipe explanations traced through the knowledge graph, '
     'grounded plain-language summaries with an automatic faithfulness check, and improved cuisine and '
     'feasibility scores.'),
    ('Thesis writing: ', 'the Introduction and Literature Review chapters.'),
])
body('All evaluations use held-out patients aged 12–59, the scope of the Layer 2 model; fibrosis scoring and the '
     'NHANES III data cover adults only.')

# ---------------------------------------------------------------- 2
heading('2. Work Completed', 1)
heading('2.1 Task 1: Twin State and Evidence', 2)
bullets([
    'Implemented FIB-4 fibrosis risk with the EASL-EASD-EASO 2024 cut-offs and validated it against liver stiffness '
    'measured in the same FibroScan examination; a high score withholds recipes until a clinical assessment.',
    'Implemented MASLD, MetALD and alcohol-related liver disease labels from the 2023 nomenclature, and safety '
    'notices for patients under 18, with higher alcohol intake, taking glucose-lowering medicine or pregnant.',
    'Added per-patient SHAP explanations of the predicted CAP, a 90% conformal CAP interval and bootstrap bands '
    'for the calibrated probabilities.',
    'Built a causal knowledge graph of 14 links that restricts dietary priorities to causes of liver fat.',
    'Downloaded the NHANES III examination, laboratory and hepatic ultrasound files and mapped them onto the '
    'model’s inputs.',
    'Built the virtual-patient simulator with cited evidence tables (in-silico evaluation, part 1).',
])
heading('2.2 Task 2: Recipe Ranking and Explanations', 2)
bullets([
    'Explained each recommended recipe through the knowledge graph, from each criterion to liver fat with its '
    'evidence rows, plus the cuisine and practicality context.',
    'Generated patient summaries from a template using only the twin’s own facts, with an automatic check of every '
    'number and citation and a slot for a language model later.',
    'Added ingredient co-occurrence to the cuisine score and cooking techniques and equipment to the feasibility '
    'score, and re-evaluated the ranking on 300 held-out patients.',
])
heading('2.3 Thesis Writing', 2)
bullets([
    'Drafted the Introduction (Chapter 1) and Literature Review (Chapter 2) in LaTeX, with preview PDFs.',
    'Covered every paper in the “papers read” tracker; about 60 papers were read in full and the rest from their '
    'abstracts.',
    'Checked published R² values against the full papers and corrected six figures in earlier reports.',
])
para('All code is maintained in the project repository, and 129 automated tests pass '
     '(84 from Meeting 9 and 45 new).', bold_lead='Implementation status: ', align='justify')

# ---------------------------------------------------------------- 3
heading('3. Technical Implementation and Results', 1)
heading('3.1 Fibrosis Risk and Referral Pathway', 2)
body('FIB-4 = age × AST / (platelets × √ALT) (Sterling et al., 2006) is computed for patients aged 18–59: below 1.3 '
     'is low risk, 1.3–2.67 indeterminate (a second test such as elastography is advised) and above 2.67 high risk. '
     'Patients under 18 receive no score, because FIB-4 was developed in adults. It was validated against liver '
     f'stiffness in {fib_summary["n"]:,} adults aged 18–59 with a reliable reading (IQR/median ≤ 30%).')
rows = []
for cutoff in (8.0, 12.0):
    part = fib[(fib.group == 'ages 18-59, reliable stiffness') & (fib.stiffness_cutoff_kpa == cutoff)]
    low, high = part[part.rule == 'fib4_ge_1.3'].iloc[0], part[part.rule == 'fib4_gt_2.67'].iloc[0]
    rows.append([f'≥ {cutoff:g} kPa', pct(low.prevalence), f3(low.auc), f3(low.sensitivity), f3(low.npv),
                 f3(high.specificity), f3(high.ppv)])
table('Table 1. FIB-4 against measured liver stiffness, adults aged 18–59.',
      ['Stiffness', 'Prevalence', 'AUC', 'Sensitivity (FIB-4 ≥ 1.3)', 'NPV (FIB-4 < 1.3)',
       'Specificity (> 2.67)', 'PPV (> 2.67)'], rows, widths=[2.2, 2.1, 1.6, 2.6, 2.6, 2.6, 2.3],
      numeric=(1, 2, 3, 4, 5, 6))
shares = fib_summary['category_shares']
body(f'{pct(shares["low"])} of adults were low risk, {pct(shares["indeterminate"])} indeterminate and '
     f'{pct(shares["high"])} high risk. FIB-4 rules out raised stiffness well (negative predictive value '
     f'{rows[1][4]} for ≥ 12 kPa) but detects it poorly (AUC {rows[0][2]} and {rows[1][2]}), '
     'which matches its known weakness in young, general populations. It is therefore used as a rule-out test.')
body('Two refinements follow from this. Because FIB-4 is less reliable under age 35 (AASLD 2023 Practice Guidance), '
     'a low score at ages 18–34 now carries a note that it does not fully rule out scarring and that a FibroScan '
     'would help; this applies to 35.6% of adults. And when a patient has a reliable FibroScan reading, for example '
     'after referral, the measured stiffness replaces FIB-4: below 8 kPa low risk, 8–12 kPa indeterminate, and 12 kPa '
     'or more high risk, with recipes withheld until a clinical assessment. NHANES participants are evaluated without '
     'their stiffness, as a real user without a scan would be.')

heading('3.2 Disease Subtype and Safety Guardrails', 2)
body('Steatosis with at least one of five cardiometabolic criteria (body size, glycaemia, blood pressure, '
     'triglycerides, HDL) is MASLD; MASLD with 140–350 g alcohol a week (women) or 210–420 g (men) is MetALD; above '
     'that range the label is alcohol-related liver disease (ALD). Alcohol is estimated from the NHANES frequency '
     'and quantity questions at 14 g per US standard drink.')
table('Table 2. Labels in the in-scope cohort, using measured CAP ≥ 248 dB/m as steatosis.',
      ['Label', 'Adults (18–59)', 'Adolescents (12–17)'],
      [[name, share(True, key), share(False, key)] for name, key in
       [('MASLD', 'MASLD'), ('MetALD', 'MetALD'), ('ALD', 'ALD'), ('Steatosis without a cardiometabolic factor',
        'cryptogenic_SLD'), ('No steatosis', 'no_steatosis_indicated')]], widths=[8, 4, 4], numeric=(1, 2))
cols = {c: pct(notices[c].mean()) for c in notices if c.startswith('notice_')}
body(f'Safety notices: {cols.get("notice_under_18", "0%")} of in-scope participants are under 18, '
     f'{cols.get("notice_alcohol", "0%")} drink in the MetALD range or above, and '
     f'{cols.get("notice_glucose_lowering_medicine", "0%")} take glucose-lowering medicine. No pregnant participant '
     'is in the cohort, because NHANES does not perform FibroScan in pregnancy; the notice remains for real users. A '
     'high FIB-4 adds a referral notice and withholds recipes.')

heading('3.3 Patient-Level Explanations and Uncertainty', 2)
body('Per-patient SHAP values split each predicted CAP into the average prediction plus one contribution per input, '
     'in dB/m. For participant SEQN 109349 (predicted CAP 313.7 dB/m against an average of 258.0), waist '
     'circumference added 15.6 dB/m and the derived Fatty Liver Index 10.5 dB/m.')
half = conformal['half_width']
cov = conformal['test_coverage']
table('Table 3. Conformal CAP intervals: held-out test coverage within the scope.',
      ['Target coverage', 'Interval', 'Observed coverage', 'Ages 12–17', 'Ages 18–39', 'Ages 40–59'],
      [[pct(float(level)), f'± {half[level]:.1f} dB/m', pct(cov[level]['scope']), pct(cov[level]['scope ages 12-17']),
        pct(cov[level]['scope ages 18-39']), pct(cov[level]['scope ages 40-59'])] for level in ('0.8', '0.9', '0.95')],
      widths=[2.8, 3, 2.8, 2.4, 2.4, 2.4], numeric=(1, 2, 3, 4, 5))
widths = conformal['mean_probability_band_width_scope']
body(f'The 90% interval is wide (± {half["0.9"]:.1f} dB/m), the direct consequence of the model’s R², which is why '
     'the twin leads with calibrated probabilities. Bootstrap bands for those probabilities have a mean width of '
     f'{100 * widths["cap248"]:.1f} percentage points (CAP ≥ 248 dB/m) and {100 * widths["cap280"]:.1f} '
     '(CAP ≥ 280 dB/m); they cover the calibration step only.')

heading('3.4 Causal Knowledge Graph', 2)
body('Fourteen hand-written links connect six levers (energy, saturated fat, added sugar, diet quality, physical '
     'activity, alcohol) through body weight and insulin resistance to liver fat, liver injury and fibrosis; CAP, '
     'ALT and liver stiffness are measurements only. Each link cites rows of the evidence table or the paper it '
     'rests on. A dietary priority may act only on a lever with a causal path to liver fat, so no recipe criterion '
     'targets a measurement such as ALT or CAP.')
table('Table 4. Causal chains behind the four dietary priorities.',
      ['Priority', 'Chain to liver fat', 'Evidence (strongest link)'],
      [['Weight management', 'energy intake → body weight → liver fat', 'VG2015-WL5 (STRONG)'],
       ['Limit saturated fat', 'saturated fat intake → liver fat', 'LUU2018-SAT (MODERATE)'],
       ['Limit excess sugar', 'added sugar intake → liver fat', 'CHI2014-HYPER (MODERATE)'],
       ['Diet quality', 'Mediterranean-style diet quality → liver fat', 'STE2026-MEDDIET (MODERATE)']],
      widths=[3.6, 7, 5.4])

heading('3.5 NHANES III External Validation Data', 2)
body('The NHANES III (1988–1994) examination and laboratory files and the 2009–2010 re-reading of the archived '
     'gallbladder ultrasound for hepatic steatosis were downloaded and mapped onto the model’s inputs, converting '
     'units where they differ (for example calcium from mmol/L and CRP from mg/dL).')
bullets([
    f'{n3["participants_20_59_gradable"]:,} adults aged 20–59 with a gradable image; '
    f'{pct(n3["moderate_severe_share"])} have moderate or severe steatosis.',
    f'{n3["model_inputs_mapped"]} of the {n3["model_inputs_total"]} model inputs are mapped; the 24-hour recall, '
    'questionnaire items and some blood counts are not, and the model treats them as missing.',
    'Mapped values agree with the modern cohort except where the population has changed (median BMI 26.1 against '
    '28.4 kg/m²); assays were not recalibrated, and NHANES III CRP is not a high-sensitivity assay.',
    'The public-use Linked Mortality File links 10,672 of them to deaths up to 31 December 2019: 2,174 deaths over '
    'a median 27.3 years. Its cause-of-death groups have no liver category, so it supports all-cause, cardiovascular '
    'and cancer mortality but not liver-related deaths.',
])

heading('3.6 Cuisine and Feasibility Scores', 2)
body('In RecipeDB, 7,134 of the 10,108 candidate meals (71%) are labelled “Other” and previously scored 0 for every '
     'cuisine. Ingredient co-occurrence now measures how typical a recipe’s ingredient pairs are of a cuisine: recipes '
     'labelled with a cuisine score a median of 55–66 for it, against 11–20 for recipes labelled with other cuisines. '
     'Feasibility now also reads the instructions: each demanding technique (deep-frying, kneading dough, overnight '
     'marinating, thermometer work, flambéing) lowers the score by 25% and each piece of special equipment by 10%.')
rows = []
for name, frame in (('Meeting 9: label only; time, ingredients, steps', before),
                    ('Label and ingredients blended; techniques averaged in', blend),
                    ('Ingredients fill label gaps; techniques averaged in', gap_avg),
                    ('Adopted: ingredients fill label gaps; techniques as a penalty', after)):
    r = frame.loc[DEFAULT]
    rows.append([name, f3(r.health), f3(r.cuisine), f3(r.feasibility), f3(r.all_three),
                 '–' if 'cuisine_typicality' not in r else f'{r.cuisine_typicality:.0f}',
                 '–' if 'demanding_technique' not in r else pct(r.demanding_technique)])
table('Table 5. Default ranking on 300 held-out patients: share of top-10 recipes meeting each criterion.',
      ['Cuisine scoring', 'Health', 'Cuisine label', 'Feasible', 'All three', 'Ingredient typicality',
       'Demanding technique'], rows, widths=[5.6, 1.5, 1.7, 1.6, 1.6, 2.1, 2.1], numeric=(1, 2, 3, 4, 5, 6))
body('Four designs were compared. Blending the label with the ingredients for every recipe narrowed the gap between a '
     'correctly labelled recipe and an unlabelled one from 100 to about 10 points, so practicality outweighed cuisine '
     'and fewer top-10 recipes were in the patient’s cuisine. Scoring only unlabelled recipes on ingredients (up to 50) '
     'restored the cuisine match. Averaging the technique check in as a fourth feasibility component diluted time, '
     'ingredients and steps and lowered the practical share; applying it as a penalty leaves recipes without '
     'demanding techniques unchanged. The adopted design combines the two.')

heading('3.7 Recipe Explanations and Plain-Language Summaries', 2)
body('Each of the top three recipes is explained criterion by criterion: the value, the reference it meets, the causal '
     'chain to liver fat with evidence rows and strength, and the cuisine and practicality context. For SEQN 109349 '
     'the first recipe, Chipotle Pinto Beans, has 0.8% of energy from saturated fat against a reference of 10%, '
     'traced as “saturated fat intake → liver fat (MODERATE, LUU2018-SAT, EASL2024-DIET)”.')
body('The patient summary is written by a template from the twin’s own facts: the CAP estimate and interval, the '
     'probabilities, the label, FIB-4, safety notices, priorities with evidence identifiers, and recipe names. An '
     'automatic check confirms that every number and citation in the text appears in those facts. A language model '
     'can be plugged in later; if its text fails the check, the template is shown instead. Tests confirm that an '
     'invented number or citation is caught and that recipes are left out when they are withheld.')

heading('3.8 In-Silico Evaluation Design', 2)
body('The evaluation has two separate parts. The simulator plays the patient: it generates each virtual patient’s '
     'true monthly weight, blood tests, liver fat and CAP from published trials, and the measurements a real patient '
     'would report. The twin (Layers 2–5) sees only those measurements and must estimate, predict and adjust each '
     'month. Comparing what the twin estimated with what the simulator knew shows how well it tracks, detects '
     'change, projects and learns each patient’s response.')
bullets([
    ('Independence: ', 'the simulator never uses the Layer 2 model, so the twin is not graded against its own output. '
     'An automated test enforces this.'),
    ('Purpose: ', 'synthetic data are used to test the twin, never to train it.'),
    ('Terminology: ', 'the design is an in-silico evaluation with virtual patients (a plasmode simulation, or a '
     '“twin experiment” in data assimilation).'),
    ('What it can show: ', 'that the twin updates, tracks and adapts correctly when the truth is known; it does not '
     'show that the twin predicts real patients.'),
])

heading('3.9 Evidence for the Simulator', 2)
body('Every number the simulator uses is stored in a cited table, with its source, the location in the paper, and '
     'whether it was quoted or derived (with the calculation written out).')
table('Table 6. Simulator parameter table by level of access.',
      ['Access', 'Rows', 'Meaning'],
      [['Full text', '33', 'Read in the full paper'],
       ['Abstract', '15', 'Only the abstract was available'],
       ['Database', '9', 'EFLM Biological Variation Database; CDC growth charts'],
       ['Design', '6', 'Project assumptions, stated as such'],
       ['Secondary', '1', 'From a review; primary trial still to be confirmed']], widths=[3, 2, 11], numeric=(1,))
table('Table 7. Calibration targets the virtual patients must reproduce.',
      ['Trial', 'Targets', 'Use'],
      [['Look AHEAD, year 1', 'Mean weight change −8.6% (intensive) vs −0.6%; ≥ 5% lost: 68.0% vs 13.3%', 'Tune'],
       ['Look AHEAD, year 8', 'Mean −4.7%; ≥ 5% lost: 50.3%; ≥ 10% lost: 26.9%', 'Held back'],
       ['Diabetes Prevention Program', '50% reached ≥ 7% loss at 24 weeks; 38% kept it', 'Tune; held back'],
       ['Vilar-Gomez et al. (2015)', '30% lost ≥ 5% at 52 weeks with ordinary lifestyle advice', 'Tune'],
       ['Sangprasert et al. (2025)', 'CAP 319 → 292 dB/m over 6 months', 'Held back']],
      widths=[4, 9.5, 2.5])
body('Held-back targets are not looked at until calibration is complete, so they remain an honest check.')

heading('3.10 Virtual-Patient Simulator', 2)
body('Month 0 is the 1,236 real in-scope test patients. Each follows the usual diet, scenario A (−500 kcal/day) or '
     'scenario B (−250 kcal/day) for 12 months. A person’s adherence pattern (steady, slipping, regaining or dropping '
     'out) and how strongly they respond are drawn once per person, so scenarios for the same person differ only in '
     'diet.')
bullets([
    ('Weight: ', 'Hall’s energy-balance rule (1 kg per 100 kJ/day eventually), with each patient’s speed and '
     'eventual loss set by their body fat (time constant 5–24 months).'),
    ('Growth: ', 'patients aged 12–17 (209) grow along their own CDC weight and height centiles.'),
    ('Blood markers: ', 'glucose, HbA1c, insulin, lipids and blood pressure change by a share of the patient’s own '
     'baseline, with separate evidence for patients with diabetes (Look AHEAD) and without (34 pooled trials); CRP, '
     'uric acid and white cells change per kg.'),
    ('Liver: ', 'liver fat falls with weight lost and further while eating less; CAP follows liver fat '
     '(5 dB/m per PDFF point); ALT, AST and GGT follow the liver-fat change.'),
])
table('Table 8. True trajectories of the virtual cohort (calibration not yet applied).',
      ['Scenario', 'Month', 'Mean weight change', 'Lost ≥ 5%', 'Lost ≥ 10%', 'Mean CAP change (dB/m)'],
      [['A (−500 kcal/day)', '6', '−5.1%', '44%', '6%', '−9.5'],
       ['A (−500 kcal/day)', '12', '−5.9%', '50%', '17%', '−11.3'],
       ['B (−250 kcal/day)', '6', '−2.3%', '5%', '0%', '−4.8'],
       ['B (−250 kcal/day)', '12', '−2.5%', '17%', '1%', '−5.8'],
       ['Usual diet', '12', '+1.0%*', '0%', '0%', '0.0']], widths=[3.6, 1.6, 2.8, 2.2, 2.2, 3.6], numeric=(1, 2, 3, 4, 5))
para('*Growth of the adolescents; adults on the usual diet do not change.', size=9.5, italic=True)

table('Table 9. Weaknesses found in the first run and how they were fixed.',
      ['Problem', 'Fix', 'Evidence'],
      [['CAP conversion five times too large', 'Use the published CAP cut-offs (275 dB/m ≈ 10% PDFF): 5 dB/m per point',
        'Wang et al. (2020)'],
       ['ALT fell too much in people with healthy livers', 'Liver enzymes follow the change in liver fat',
        'Koutoukidis et al. (2019); Wong et al. (2013)'],
       ['Blood-sugar effects came from people with diabetes', 'Separate evidence for patients without diabetes',
        'Appuhamy et al. (2014)'],
       ['Liver fat reacted too slowly', 'Extra fall while eating less, fitted to the first week of Lim et al.',
        'Lim et al. (2011); Patel et al. (2015)'],
       ['One weight-loss speed for everyone', 'Personal speed and eventual loss from body fat',
        'Hall (2011) web appendix; Deurenberg et al. (1991)'],
       ['Adolescents did not grow', 'Growth along CDC centiles', 'CDC 2000 growth charts'],
       ['Insulin, hip, arm, CRP fixed', 'Each now changes with weight',
        'Appuhamy (2014); Lim (2011); Selvin et al. (2007)'],
       ['Uric acid and white cells fixed', 'Each now changes per kg lost',
        'Yokose et al. (2020); Dixon & O’Brien (2006)']], widths=[5, 6.5, 4.5])
body('A workaround for markers without trial evidence was tested and rejected: estimating a weight-loss effect from '
     'how the marker differs between heavier and lighter people in NHANES. For blood tests with known trial values '
     'it was off by a factor of 0.6 to 5.5, and in the wrong direction for HDL. It is used only for arm '
     'circumference, where the same method reproduced the trial values for hip (0.63 vs 0.65 cm/kg) and waist '
     '(0.77 vs 0.86 cm/kg). Red-cell measures, platelets and ferritin are held constant: no trial shows a change, and '
     'they barely differ with body size.')

heading('3.11 Thesis Chapters', 2)
body('The Introduction (about 1,750 words) covers the background, digital twins, computational gastronomy, the '
     'problem and research gap, the research question and objectives, the approach with a seven-layer overview '
     'figure, contributions, scope and limitations, and the thesis outline.')
body('The Literature Review (about 4,900 words, 84 references) covers MASLD and its dietary treatment, non-invasive '
     'measurement of liver fat, machine-learning prediction and its limits, dietary assessment, digital twins, '
     'computational gastronomy and food recommendation, and how a twin can be evaluated without longitudinal data. '
     'It ends with a table of the research gap.')
body('The full-text check corrected six figures in the R² explanation document, Experiment Record and Progress '
     'Report: the published range (0.34–0.56 → 0.34–0.54), FLI+ (0.56 for one fold → 0.54, the four-fold mean), FLI '
     'alone (0.28–0.35 → 0.28–0.36), UK Biobank DXA (0.63–0.65 → 0.63–0.64), Haas (“not reliable” → r 0.58) and Zhu '
     'et al. (AUC 0.892, a precision–recall value → macro AUC 0.874 across three grades).')

body('About 60 papers were read in full. About 22 could only be read as abstracts because they are behind paywalls; '
     'these are cited only for what their abstracts state. Published R² values appear in the Literature Review; the '
     'project’s own R² experiments will go in the planned Experiments chapter.')


# ---------------------------------------------------------------- 4
heading('4. Current Challenges and Limitations', 1)
bullets([
    ('FIB-4 detects fibrosis poorly in this population: ', 'it is used to rule out raised stiffness, not to find it.'),
    ('Adolescent MASLD criteria: ', 'checked against the nomenclature’s figure. Two alternatives are not used: '
     'waist above the 95th percentile and, at age 12, blood pressure above the 95th percentile, because they need '
     'reference tables the project does not have. They can only miss a criterion, never add one.'),
    ('NHANES limits for the labels: ', 'blood-pressure and cholesterol treatment are not available, so a diagnosis '
     'stands in for treatment; triglycerides are non-fasting; other causes of steatosis are not assessed.'),
    ('Wide CAP intervals: ', 'the 90% interval of about ± 67 dB/m reflects the model’s accuracy; probabilities remain '
     'the twin’s main state.'),
    ('Cuisine labels: ', 'only 29% of candidate meals carry a specific cuisine label; ingredient co-occurrence fills '
     'gaps but is an indirect signal.'),
    ('NHANES III: ', 'assays and the population differ from 2017–2023, the reference is an ultrasound grade rather '
     'than CAP, and the public mortality file cannot identify liver-related deaths.'),
    ('Summaries: ', 'the template is safe but plain and lists evidence identifiers; a language model has not yet been '
     'connected.'),
    ('Simulator not yet calibrated: ', 'adherence patterns and response spread are starting guesses until they are '
     'tuned to the trial targets, and scenarios A and B are fixed calorie reductions.'),
    ('User-retrieved values: ', 'the EFLM biological-variation values and the DIRECT uric-acid coefficient were read '
     'from the source by the user; screenshots should be kept for the appendix.'),
])

# ---------------------------------------------------------------- 5
heading('5. Decisions Taken During Meeting 10', 1)
sub('5.1 Use FIB-4 as a rule-out test with a referral before recipes')
body('FIB-4 rules out raised stiffness reliably but detects it poorly. High-risk patients are asked to seek a '
     'clinical assessment, and recipes are withheld until then.')
body('A measured FibroScan stiffness, when available, replaces FIB-4, and a low FIB-4 under age 35 carries a note '
     'recommending one.')
sub('5.2 Restrict dietary priorities to causes of liver fat')
body('The causal knowledge graph allows a recipe criterion only on a lever with a causal path to liver fat, so the '
     'twin never optimises a measurement such as ALT or CAP directly.')
sub('5.3 Let ingredients fill cuisine gaps without overriding labels')
body('Blending label and ingredients for every recipe lowered the share of top-10 recipes in the patient’s cuisine. '
     'Labelled recipes therefore keep their label score, and only unlabelled recipes are scored on ingredients.')
sub('5.4 Use template summaries until a language model is chosen')
body('Summaries are written from a template now; a language model can be added later through the same automatic '
     'faithfulness check, with the template as the fallback.')
sub('5.5 Keep the simulator independent and use it only for testing')
body('The simulator is built only from published evidence and never uses the Layer 2 model. Synthetic data test the '
     'twin and are never used to train it.')
sub('5.6 Fix weaknesses with evidence rather than list them as limitations')
body('Eight weaknesses in the simulator’s first run were fixed using new published sources. A shortcut for markers '
     'without evidence was tested first and rejected because it failed on markers with known answers.')

# ---------------------------------------------------------------- 6
heading('6. Overall Project Status', 1)
body('All tasks planned for Meeting 10 are complete. The main outcomes are:')
bullets([
    'A second dimension of the twin state: fibrosis risk, validated against measured stiffness, with a referral '
    'pathway.',
    'Research labels for MASLD, MetALD and ALD, and safety notices for groups for whom unsupervised diet change may '
    'not be appropriate.',
    'Per-patient explanations, a 90% CAP interval with verified coverage, and probability bands.',
    'A causal knowledge graph that both constrains the priorities and explains each recipe.',
    'Template summaries with an automatic faithfulness check and a slot for a language model.',
    'Cuisine and feasibility scores that use ingredient co-occurrence and cooking techniques.',
    'NHANES III data mapped for external validation and linked to 27 years of mortality follow-up.',
    'A working virtual-patient simulator and the first two thesis chapters.',
    '129 passing automated tests.',
])

# ---------------------------------------------------------------- 7
heading('7. Immediate Next Steps — Meeting 11', 1)
body('From the weekly action plan:')
numbered([
    ('Backend: ', 'a FastAPI service wrapping Layers 2–5 with an SQLite database.'),
    ('Re-assessment timeline: ', 'dated measurements and food logs with the predict–observe–compare–adjust loop.'),
    ('3D liver view: ', 'grades S0–S3 with a timeline slider, coloured by the calibrated probabilities and their '
     'uncertainty.'),
    ('In-silico evaluation, part 2: ', 'the observation layer (laboratory variation, CAP noise, under-reported food '
     'logs, quarterly blood tests, missed visits) and calibration of the simulator to the trial targets.'),
    ('Dashboard and new criteria: ', 'the React dashboard, ultra-processing (FoodProX) and glycaemic load.'),
])
body('Also outstanding: continue recruiting 5–10 pilot evaluators.')

doc.save(OUT)
print('saved', OUT)

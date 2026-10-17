"""Exploratory analysis of every dataset the final pipeline uses.

Writes figures and a summary (summary.json) to data/results/eda/:
- NHANES FibroScan cohort, 2017-March 2020 and 2021-2023: cohort, CAP and
  steatosis prevalence by group, examination quality, missing inputs, and
  the measurements most associated with CAP;
- day-1 24-hour recall with FPED: reliability and intake against the
  reference values used for recipes, HEI-2020;
- RecipeDB: screening from 20,000 recipes to candidate meals and to those
  that meet every reference value, cuisines, composition and ratings.

Run from the project root after build_combined_dataset.py and
build_habitual_diet.py:

    .venv/Scripts/python.exe -m src.eda_report
"""
import json
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.build_combined_dataset import OUTPUT as COHORT_FILE
from src.build_habitual_diet import OUTPUT as DIET_FILE
from src.combined_features import COMBINED_FEATURES
from src.recipe_scoring.multi_criteria import is_candidate, score_components
from src.recipe_scoring.recipe_criteria import CRITERIA, criteria_for, derived_metrics
from src.train_final_model import ROOT
from src.twin_pipeline import load_recipes

OUT = ROOT / 'data' / 'results' / 'eda'
ETHNICITY = {1: 'Mexican American', 2: 'Other Hispanic', 3: 'Non-Hispanic White', 4: 'Non-Hispanic Black',
             6: 'Non-Hispanic Asian', 7: 'Other or multiracial'}
AGE_BINS, AGE_LABELS = [11, 17, 39, 59, 120], ['12–17', '18–39', '40–59', '60+']
# Inputs that NHANES collects only for a subsample or after a screening question.
BY_DESIGN = {'LBXIN': 'fasting subsample', 'LBXGLU': 'fasting subsample', 'DIQ050': 'asked if diabetic',
             'DIQ070': 'asked if diabetic', 'SMQ040': 'asked if ever smoked', 'ALQ130': 'asked if drinks',
             'ALQ151': 'adults only', 'ALQ121': 'adults only', 'ALQ111': 'adults only',
             'DMDEDUC2': 'adults 20+', 'LBXFER': 'subsample'}
LABELS = {'BMXWAIST': 'Waist circumference', 'BMXBMI': 'BMI', 'BMXWT': 'Weight', 'LBXSTR': 'Triglycerides',
          'LBXSATSI': 'ALT', 'LBXSGTSI': 'GGT', 'LBXGH': 'HbA1c', 'LBDHDD': 'HDL cholesterol (inverse)',
          'LBXIN': 'Fasting insulin', 'LBXSUA': 'Uric acid', 'BMXHIP': 'Hip circumference',
          'BMXARMC': 'Arm circumference', 'LBXSGL': 'Glucose (serum)', 'LBXGLU': 'Fasting glucose',
          'RIDAGEYR': 'Age', 'LBXHSCRP': 'C-reactive protein', 'LBXSASSI': 'AST', 'BPXOSY1': 'Systolic BP',
          'LBXTC': 'Total cholesterol', 'LBXWBCSI': 'White cell count', 'BPXODI1': 'Diastolic BP',
          'LBXFER': 'Ferritin', 'BMXHT': 'Height', 'LBXSCR': 'Creatinine', 'LBXPLTSI': 'Platelets',
          'ALQ130': 'Drinks per drinking day', 'ALQ151': 'Ever 4-5+ drinks almost daily',
          'ALQ121': 'Drinking frequency, past year', 'ALQ111': 'Ever had alcohol', 'LBXSLDSI': 'LDH',
          'DMDEDUC2': 'Education', 'DIQ050': 'Taking insulin', 'DIQ070': 'Taking diabetes pills',
          'SMQ040': 'Smokes now', 'WHD050': 'Weight one year ago', 'BMXARML': 'Arm length',
          'BMXLEG': 'Leg length', 'INDFMPIR': 'Income-to-poverty ratio', 'LBXHGB': 'Haemoglobin',
          'LBXHCT': 'Haematocrit', 'LBXSBU': 'Blood urea nitrogen', 'LBXSAL': 'Albumin',
          'LBXSIR': 'Iron', 'LBXSAPSI': 'Alkaline phosphatase', 'LBXSTP': 'Total protein',
          'WHQ070': 'Tried to lose weight', 'DIQ160': 'Told prediabetes', 'DR1TTFAT': 'Total fat (recall)',
          'DR1TKCAL': 'Energy (recall)', 'DR1TPROT': 'Protein (recall)', 'DR1TCARB': 'Carbohydrate (recall)'}


def style():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 10, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                         'axes.spines.top': False, 'axes.spines.right': False})


def share(mask):
    return float(np.mean(mask))


def cohort(summary):
    data = pd.read_csv(COHORT_FILE)
    cap = data.LUXCAPM
    data['age_group'] = pd.cut(data.RIDAGEYR, AGE_BINS, labels=AGE_LABELS)
    data['ethnicity'] = data.RIDRETH3.map(ETHNICITY)
    data['sex'] = data.RIAGENDR.map({1: 'Male', 2: 'Female'})
    summary['cohort'] = {
        'participants': len(data), 'by_cycle': data.cycle.value_counts().to_dict(),
        'age': data.RIDAGEYR.describe().round(1).to_dict(), 'female': share(data.RIAGENDR.eq(2)),
        'bmi_median': float(data.BMXBMI.median()), 'obese': share(data.BMXBMI >= 30),
        'cap': cap.describe().round(1).to_dict(), 'cap248': share(cap >= 248), 'cap280': share(cap >= 280),
        'cap302': share(cap >= 302), 'cap_at_400': share(cap >= 400),
        'cap_iqr_reliable': share(data.LUXCPIQR < 40),
        'by_cycle_cap248': data.groupby('cycle').LUXCAPM.apply(lambda c: share(c >= 248)).to_dict(),
        'by_group': {name: data.groupby(col, observed=True).LUXCAPM.agg(
            n='size', mean='mean', cap248=lambda c: share(c >= 248)).round(3).reset_index()
            .rename(columns={col: 'group'}).to_dict('records')
            for name, col in [('sex', 'sex'), ('age', 'age_group'), ('ethnicity', 'ethnicity')]},
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'wspace': 0.3})
    bins = np.arange(100, 405, 10)
    for cycle, colour in (('2017-2020', 'tab:blue'), ('2021-2023', 'tab:orange')):
        axes[0].hist(cap[data.cycle.eq(cycle)], bins=bins, alpha=0.55, color=colour,
                     label=f'{cycle} (n = {data.cycle.eq(cycle).sum():,})')
    for cut, name, dx in ((248, 'S1', -14), (268, 'S2', -14), (280, 'S3', 2)):
        axes[0].axvline(cut, color='black', ls=':', lw=1)
        axes[0].text(cut + dx, axes[0].get_ylim()[1] * 0.97, name, fontsize=8)
    axes[0].set_xlabel('CAP (dB/m)'); axes[0].set_ylabel('Participants')
    axes[0].set_title('(a) CAP distribution by cycle', loc='left', fontsize=11)
    axes[0].legend(frameon=False, fontsize=8, loc='upper right')
    groups = data.groupby('age_group', observed=True).LUXCAPM
    order = AGE_LABELS
    axes[1].boxplot([groups.get_group(g) for g in order], labels=order, showfliers=False)
    axes[1].axhline(248, color='black', ls=':', lw=1)
    axes[1].set_xlabel('Age group (years)'); axes[1].set_ylabel('CAP (dB/m)')
    axes[1].set_title('(b) CAP by age group', loc='left', fontsize=11)
    fig.savefig(OUT / 'fig_cap_distribution.png'); plt.close(fig)

    # Spearman correlation of each numeric input with CAP.
    numeric = [c for c in COMBINED_FEATURES if c not in ('RIAGENDR', 'RIDRETH3') and data[c].nunique() > 10]
    corr = data[numeric].corrwith(cap, method='spearman').dropna()
    top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(15)
    summary['correlations'] = {c: round(float(v), 3) for c, v in top.items()}
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    names = [LABELS.get(c, c).replace(' (inverse)', '') for c in top.index][::-1]
    ax.barh(names, top.values[::-1], color=['tab:blue' if v > 0 else 'tab:red' for v in top.values[::-1]])
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Spearman correlation with CAP')
    ax.set_title('Measurements most associated with CAP', loc='left', fontsize=11)
    fig.savefig(OUT / 'fig_correlations.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'wspace': 0.3})
    for ax, col, label in ((axes[0], 'BMXBMI', 'BMI (kg/m²)'), (axes[1], 'BMXWAIST', 'Waist (cm)')):
        ok = data[col].notna()
        hb = ax.hexbin(data.loc[ok, col], cap[ok], gridsize=40, cmap='Blues', mincnt=1)
        ax.axhline(248, color='black', ls=':', lw=1)
        ax.set_xlabel(label); ax.set_ylabel('CAP (dB/m)')
        rho = data[col].corr(cap, method='spearman')
        ax.set_title(f'CAP against {label.split(" ")[0]} (Spearman ρ = {rho:.2f})', loc='left', fontsize=11)
        fig.colorbar(hb, ax=ax, fraction=0.046, label='Participants')
    fig.savefig(OUT / 'fig_cap_vs_body_size.png'); plt.close(fig)

    missing = data[COMBINED_FEATURES].isna().mean().sort_values(ascending=False)
    summary['missing'] = {
        'inputs': len(COMBINED_FEATURES), 'complete_inputs': int((missing == 0).sum()),
        'inputs_over_20pct': int((missing > 0.2).sum()), 'median_missing': float(missing.median()),
        'top': [dict(input=c, label=LABELS.get(c, c), share=round(float(v), 3), reason=BY_DESIGN.get(c, ''))
                for c, v in missing.head(12).items()]}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    top = missing.head(15)[::-1]
    ax.barh([f'{LABELS.get(c, c)}{" *" if c in BY_DESIGN else ""}' for c in top.index], top.values * 100,
            color=['grey' if c in BY_DESIGN else 'tab:blue' for c in top.index])
    ax.set_xlabel('Missing (% of participants)')
    ax.set_title('Inputs with the most missing values (* by design)', loc='left', fontsize=11)
    fig.savefig(OUT / 'fig_missingness.png'); plt.close(fig)
    return data


def diet(summary):
    data = pd.read_csv(DIET_FILE)
    reliable = data[data.recall_reliable]
    summary['diet'] = {
        'participants': len(data), 'with_recall': int(data.DR1TKCAL.notna().sum()),
        'reliable': len(reliable),
        'excluded': {'no usable recall': int((data.DR1DRSTZ != 1).sum()),
                     'not a usual day': int((data.DR1DRSTZ.eq(1) & data.DR1_300.ne(2)).sum()),
                     'implausible energy': int((data.DR1DRSTZ.eq(1) & data.DR1_300.eq(2)
                                                & ~data.recall_reliable).sum())},
        'fped_source': data.fped_source.value_counts().to_dict(),
        'medians': reliable.groupby('cycle')[['DR1TKCAL', 'saturated_fat_pct_energy', 'added_sugar_pct_energy',
                                              'sugar_pct_energy', 'fiber_per_1000kcal', 'hei2020']]
        .median().round(1).to_dict('index'),
        'miss_reference': {'saturated fat > 10% of energy': share(reliable.saturated_fat_pct_energy > 10),
                           'added sugars > 10% of energy': share(reliable.added_sugar_pct_energy.dropna() > 10),
                           'fibre < 14 g per 1,000 kcal': share(reliable.fiber_per_1000kcal < 14)},
    }
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), gridspec_kw={'wspace': 0.35})
    panels = [('saturated_fat_pct_energy', 'Saturated fat (% energy)', 10, (0, 30)),
              ('added_sugar_pct_energy', 'Added sugars (% energy)', 10, (0, 50)),
              ('fiber_per_1000kcal', 'Fibre (g / 1,000 kcal)', 14, (0, 35)),
              ('hei2020', 'HEI-2020 (0–100)', None, (0, 100))]
    for ax, (col, label, ref, lim) in zip(axes, panels):
        values = reliable[col].dropna()
        ax.hist(values.clip(*lim), bins=40, color='tab:blue', alpha=0.75)
        if ref:
            ax.axvline(ref, color='tab:red', lw=1.2)
            ax.text(ref, ax.get_ylim()[1] * 0.92, ' reference', color='tab:red', fontsize=8)
        ax.set_xlabel(label); ax.set_xlim(*lim)
    axes[0].set_ylabel('Participants')
    fig.suptitle('Intake on the recall day (participants with a reliable recall)', x=0.08, ha='left', fontsize=11)
    fig.savefig(OUT / 'fig_diet.png'); plt.close(fig)


def recipes(summary):
    corpus = load_recipes()
    objectives, _ = criteria_for([{'id': k, 'priority': 'medium'} for k in CRITERIA])
    candidates = [r for r in corpus if is_candidate(r)]
    status = Counter()
    meets = []
    for recipe in candidates:
        item = score_components(recipe, objectives)
        status[item['status']] += 1
        if item.get('meets_all_references'):
            meets.append(recipe)
    cuisines = Counter(r.get('cuisine') or 'Missing' for r in corpus)
    metrics = pd.DataFrame([{**derived_metrics(r['nutrition']), 'calories': r['nutrition'].get('calories')}
                            for r in candidates]).apply(pd.to_numeric, errors='coerce')
    ratings = pd.DataFrame({'rating': [r.get('ratings') for r in corpus],
                            'count': [r.get('ratings_count') for r in corpus]}).apply(pd.to_numeric, errors='coerce')
    summary['recipes'] = {
        'corpus': len(corpus), 'candidates': len(candidates), 'not_candidates': len(corpus) - len(candidates),
        'status': dict(status), 'meet_every_reference': len(meets),
        'meet_by_cuisine': dict(Counter(r.get('cuisine') for r in meets).most_common(8)),
        'cuisines': dict(cuisines.most_common(12)), 'indian': cuisines.get('Indian', 0),
        'other_share': cuisines.get('Other', 0) / len(corpus),
        'candidate_medians': metrics.median().round(1).to_dict(),
        'candidate_meets': {'energy <= 600 kcal': share(metrics.calories <= 600),
                            'saturated fat <= 10% of energy': share(metrics.saturated_fat_pct_energy <= 10),
                            'sugars <= 10% of energy': share(metrics.sugar_pct_energy <= 10),
                            'fibre >= 14 g per 1,000 kcal': share(metrics.fiber_per_1000kcal >= 14)},
        'ratings': {'median': float(ratings.rating.median()), 'p25': float(ratings.rating.quantile(0.25)),
                    'below_4_5': share(ratings.rating < 4.5), 'count_median': float(ratings['count'].median()),
                    'count_p90': float(ratings['count'].quantile(0.9))},
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'wspace': 0.45, 'width_ratios': [1, 1.1]})
    stages = ['RecipeDB corpus', 'Main meals (candidates)', 'Scored', 'Meet every reference']
    counts = [len(corpus), len(candidates), status['ranked'], len(meets)]
    axes[0].barh(stages[::-1], counts[::-1], color='tab:blue')
    for y, v in enumerate(counts[::-1]):
        axes[0].text(v, y, f' {v:,}', va='center', fontsize=9)
    axes[0].set_xlim(0, len(corpus) * 1.25)
    axes[0].set_title('(a) From corpus to qualifying meals', loc='left', fontsize=11)
    top = cuisines.most_common(10)[::-1]
    axes[1].barh([c for c, _ in top], [n for _, n in top], color='tab:orange')
    axes[1].set_xscale('log'); axes[1].set_xlabel('Recipes (log scale)')
    axes[1].set_title('(b) Cuisine labels in RecipeDB', loc='left', fontsize=11)
    fig.savefig(OUT / 'fig_recipe_funnel.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), gridspec_kw={'wspace': 0.35})
    panels = [('calories', 'Energy per serving (kcal)', 600, (0, 1500)),
              ('saturated_fat_pct_energy', 'Saturated fat (% energy)', 10, (0, 40)),
              ('sugar_pct_energy', 'Total sugars (% energy)', 10, (0, 60)),
              ('fiber_per_1000kcal', 'Fibre (g / 1,000 kcal)', 14, (0, 40))]
    for ax, (col, label, ref, lim) in zip(axes, panels):
        ax.hist(metrics[col].dropna().clip(*lim), bins=40, color='tab:green', alpha=0.75)
        ax.axvline(ref, color='tab:red', lw=1.2)
        ax.set_xlabel(label); ax.set_xlim(*lim)
    axes[0].set_ylabel('Candidate meals')
    fig.suptitle('Composition of candidate meals against the reference values (red)', x=0.08, ha='left',
                 fontsize=11)
    fig.savefig(OUT / 'fig_recipe_composition.png'); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    summary = {}
    cohort(summary)
    diet(summary)
    recipes(summary)
    with open(OUT / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, default=float, ensure_ascii=False)
    print(json.dumps(summary, indent=1, default=float, ensure_ascii=False)[:6000])
    print(f'Saved figures and summary to {OUT}')


if __name__ == '__main__':
    main()

"""Evaluate the Layer 4 ranking engine (proposal section 12.2).

For held-out NHANES participants inside the model's scope (ages 12-59) with
at least one Layer 3 priority, the top
10 recipes from each method are checked against the three criteria:

- health: the recipe meets the reference value of every evidence-linked
  criterion for that patient (recipe_criteria.py);
- cuisine: exact or configured related match with the requested cuisine;
- feasibility: total time <= 60 min, <= 15 ingredients and <= 12 steps (the
  practical_feasibility.py screening limits).

Methods: the proposed multi-criteria ranking (default weights including
acceptability, health-gated as in rank_recipes); the same without
acceptability; diversity re-ranking; the same weights without the health gate; health score
only; nutrition only, a non-personalised score giving every recipe criterion
equal weight whatever the patient's priorities; and random selection from
the same candidate pool. A weight-sensitivity analysis reruns the proposed
ranking over a grid of weights and compares each top 10 with the default.

NHANES records no cuisine preference, so each participant is assigned one at
random (seeded) from the cuisines with at least 100 candidate recipes.

Priorities use each participant's 24-hour recall (build_habitual_diet.py,
proposal section 17.1); --no-diet reproduces the evaluation without it and
writes to data/results/layer4_before_diet. Personalisation (section 12.4) is
the mean top-10 overlap between pairs of participants given the same
cuisine: lower means recommendations differ more between patients.
priority_response.csv checks that they differ in the right direction: the
mean composition of each top 10 by the level of that priority. Run from
the project root after train_combined_model.py and build_habitual_diet.py:

    .venv/Scripts/python.exe -m src.evaluate_ranking
    .venv/Scripts/python.exe -m src.evaluate_ranking --no-diet
"""
import argparse
import itertools
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.train_final_model import ROOT
from src.combined_features import COMBINED_FEATURES
from src.build_combined_dataset import OUTPUT as DATA_FILE
from src.build_habitual_diet import OUTPUT as DIET_FILE
from src.predict_cap import MODEL_FILE, SCOPE_AGES
from src.recipe_scoring.evidence_rules import EVIDENCE_RULES
from src.recipe_scoring.meal_plan import (MAX_INGREDIENTS, MAX_STEPS, MAX_TIME, cuisine_profiles, diversify,
                                          intra_list_diversity, practical,
                                          protein_family, similar_cuisines, weekly_plan)
from src.recipe_scoring.cuisine_typicality import default_typicality
from src.recipe_scoring.practical_feasibility import techniques_and_equipment
from src.recipe_scoring.multi_criteria import (DEFAULT_WEIGHTS, gated_health, is_candidate, rank_recipes,
                                               score_components)
from src.recipe_scoring.patient_objective_generator import generate_patient_objectives
from src.recipe_scoring.patient_state import derive_patient_state
from src.recipe_scoring.recipe_criteria import CRITERIA, criteria_for, derived_metrics
from src.twin_pipeline import CLASSIFIERS, _classifier, load_recipes

OUTPUT_DIR = ROOT / 'data' / 'results' / 'layer4'
BEFORE_DIET_DIR = ROOT / 'data' / 'results' / 'layer4_before_diet'
SEED = 42
TOP_K = 10
MIN_CUISINE_RECIPES = 100
RANDOM_REPEATS = 20
MMR_POOL, PLAN_POOL = 50, 150
WITHOUT_ACCEPTABILITY = {k: w for k, w in DEFAULT_WEIGHTS.items() if k != 'acceptability'}
feasible = practical          # 60 minutes, 15 ingredients, 12 steps (meal_plan.py)


def patient_priorities(test):
    """Layer 2 probabilities and Layer 3 priorities for every test participant."""
    folder = MODEL_FILE.parent
    probabilities, thresholds = {}, {}
    for name in CLASSIFIERS:
        classifier = _classifier(folder / f'steatosis_{name}.joblib')
        probabilities[name] = classifier.predict_proba(test[COMBINED_FEATURES])
        thresholds[name] = float(classifier.threshold)
    rows = []
    for i, (_, patient) in enumerate(test.iterrows()):
        state = derive_patient_state(patient.to_dict(), {k: float(v[i]) for k, v in probabilities.items()},
                                     thresholds)
        rows.append(generate_patient_objectives(state, EVIDENCE_RULES)['priorities'])
    return rows


def signature(priorities):
    return tuple(sorted((p['id'], p['priority']) for p in priorities if p['id'] in CRITERIA))


def components_table(candidates, objectives, cuisine):
    """Arrays of criterion scores for every candidate, scored by the engine itself."""
    health, cuisine_score, feasibility, meets_health, ranked, margin, accept = [], [], [], [], [], [], []
    for recipe in candidates:
        item = score_components(recipe, objectives, requested_cuisine=cuisine)
        if item['status'] != 'ranked':
            ranked.append(False)
            health.append(np.nan), cuisine_score.append(np.nan), feasibility.append(np.nan)
            meets_health.append(False), margin.append(np.nan), accept.append(np.nan)
            continue
        ranked.append(True)
        health.append(item['scores']['health'])
        cuisine_score.append(item['scores'].get('cuisine', np.nan))
        feasibility.append(item['scores'].get('feasibility', np.nan))
        meets_health.append(item['meets_all_references'])
        margin.append(np.nan if item['health_margin'] is None else item['health_margin'])
        accept.append(item['scores'].get('acceptability', np.nan))
    return {k: np.array(v) for k, v in dict(health=health, cuisine=cuisine_score, feasibility=feasibility,
                                             meets_health=meets_health, ranked=ranked,
                                             margin=margin, acceptability=accept).items()}


def weighted(table, weights):
    total = np.zeros(len(table['health']))
    norm = np.zeros(len(table['health']))
    for name, weight in weights.items():
        values = table[name]
        present = ~np.isnan(values)
        total[present] += weight * values[present]
        norm[present] += weight
    with np.errstate(invalid='ignore', divide='ignore'):
        score = np.round(total / norm, 2)  # as overall_score
    score[~table['ranked']] = np.nan
    return score


def gated(table, weights):
    """Weighted score with recipes meeting every health reference ranked first (rank_recipes default).

    Within that group part of the health component is the margin past the references.
    """
    margin = table['margin']
    with_margin = dict(table, health=np.where(np.isnan(margin), table['health'], gated_health(margin)))
    return weighted(with_margin, weights) + 1000 * table['meets_health']


def top_k(score, health, ids, k=TOP_K):
    """Indices of the top k by score, ties broken by health then recipe ID (as rank_recipes)."""
    valid = np.flatnonzero(~np.isnan(score))
    order = np.lexsort((ids[valid], -health[valid], -score[valid]))
    return valid[order[:k]]


def ranked_items(table, ids, k):
    """The first k items of the default ranking, in rank_recipes' item format."""
    score = gated(table, DEFAULT_WEIGHTS)
    picks = top_k(score, table['health'], ids, k)
    return picks, [dict(recipe_id=ids[j], overall=float(score[j] - 1000 * table['meets_health'][j]),
                        meets_all_references=bool(table['meets_health'][j]),
                        health_margin=None if np.isnan(table['margin'][j]) else float(table['margin'][j]))
                   for j in picks]


def with_diet(test):
    """Add the recall columns that the cohort file does not already hold."""
    diet = pd.read_csv(DIET_FILE)
    diet = diet[['SEQN'] + [c for c in diet.columns if c not in test.columns]]
    return test.merge(diet, on='SEQN', how='left', validate='one_to_one')


def priority_response(assignments, top_lists, candidates):
    """Mean composition of each participant's top 10, by that criterion's priority level."""
    index = {str(r['recipe_id']): derived_metrics(r['nutrition']) for r in candidates}
    rows = []
    for name, criterion in CRITERIA.items():
        if not criterion.get('margin'):
            continue
        for a, top in zip(assignments, top_lists):
            level = dict(p.split(':') for p in a['priorities'].split(';')).get(name)
            if level:
                values = [index[i][criterion['nutrient']] for i in top]
                rows.append(dict(priority=name, level=level, measure=criterion['nutrient'],
                                 value=np.nanmean(np.array(values, dtype=float))))
    return (pd.DataFrame(rows).groupby(['priority', 'measure', 'level']).value
            .agg(participants='count', mean_top10='mean').reset_index())


def personalisation(top_lists, cuisines_by_patient):
    """Mean top-10 overlap between pairs of participants sharing a cuisine."""
    overlaps = []
    for cuisine in set(cuisines_by_patient):
        lists = [set(t) for t, c in zip(top_lists, cuisines_by_patient) if c == cuisine]
        overlaps += [len(a & b) / TOP_K for a, b in itertools.combinations(lists, 2)]
    return float(np.mean(overlaps))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patients', type=int, default=300, help='Test participants to evaluate')
    parser.add_argument('--no-diet', action='store_true', help='Ignore the 24-hour recall (before run)')
    args = parser.parse_args()
    global OUTPUT_DIR
    if args.no_diet:
        OUTPUT_DIR = BEFORE_DIET_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                         'font.size': 10, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

    data = pd.read_csv(DATA_FILE)
    manifest = pd.read_csv(MODEL_FILE.parent / 'split_manifest.csv')
    test = data[manifest.partition.eq('test').to_numpy()].reset_index(drop=True)
    test = test[test.RIDAGEYR.between(*SCOPE_AGES)].reset_index(drop=True)   # model scope
    if not args.no_diet:
        test = with_diet(test)
    priorities = patient_priorities(test)
    with_priority = [i for i, p in enumerate(priorities) if signature(p)]
    chosen = sorted(rng.choice(with_priority, size=min(args.patients, len(with_priority)), replace=False))
    print(f'{len(with_priority)} of {len(test)} test participants have a recipe criterion; '
          f'evaluating {len(chosen)}', flush=True)

    candidates = [r for r in load_recipes() if is_candidate(r)]
    ids = np.array([str(r['recipe_id']) for r in candidates])
    is_feasible = np.array([feasible(r) for r in candidates])
    # Section 7.1 additions: ingredient typicality for the requested cuisine (co-occurrence) and
    # whether the instructions need a demanding technique.
    demanding = np.array([bool(techniques_and_equipment(r.get('instructions'))[0]) for r in candidates])
    typicality_model, typicality = default_typicality(), {}
    cuisine_counts = pd.Series([r['cuisine'] for r in candidates]).value_counts()
    cuisines = sorted(c for c, n in cuisine_counts.items() if n >= MIN_CUISINE_RECIPES and c != 'Other')
    recipe_cuisine = np.array([str(r['cuisine']).casefold() for r in candidates])
    print(f'{len(candidates)} candidate meals; simulated cuisines: {cuisines}', flush=True)

    all_criteria = [{'id': name, 'priority': 'medium', 'supporting_evidence': []} for name in CRITERIA]
    generic_objectives, _ = criteria_for(all_criteria)
    tables, generic_tables = {}, {}
    grid = [dict(health=h / 10, cuisine=c / 10, feasibility=(10 - h - c) / 10)
            for h, c in itertools.product(range(1, 9), repeat=2) if h + c <= 9]
    accept = DEFAULT_WEIGHTS.get('acceptability', 0)   # held fixed across the grid
    profiles = cuisine_profiles(load_recipes())
    similar = {c: similar_cuisines(c, profiles) for c in cuisines}
    method_rows, sensitivity_rows, assignments, top_lists, plan_rows = [], [], [], [], []
    position = {rid: j for j, rid in enumerate(ids)}
    mmr_cache, plan_cache = {}, {}

    for n, i in enumerate(chosen, 1):
        cuisine = cuisines[rng.integers(len(cuisines))]
        objectives, _ = criteria_for(priorities[i])
        key = (signature(priorities[i]), cuisine)
        if key not in tables:
            tables[key] = components_table(candidates, objectives, cuisine)
        if cuisine not in generic_tables:
            generic_tables[cuisine] = components_table(candidates, generic_objectives, cuisine)
        table, generic = tables[key], generic_tables[cuisine]
        meets_cuisine = recipe_cuisine == cuisine.casefold()
        if cuisine not in typicality:
            typicality[cuisine] = np.array([np.nan if (s := typicality_model.score(r, cuisine)) is None else s
                                            for r in candidates])
        assignments.append(dict(SEQN=int(test.SEQN[i]), cuisine=cuisine, priorities=';'.join(
            f'{a}:{b}' for a, b in signature(priorities[i]))))

        def record(method, picks):
            method_rows.append(dict(SEQN=int(test.SEQN[i]), method=method, n=len(picks),
                                    health=table['meets_health'][picks].mean(),
                                    cuisine=meets_cuisine[picks].mean(),
                                    feasibility=is_feasible[picks].mean(),
                                    all_three=(table['meets_health'][picks] & meets_cuisine[picks]
                                               & is_feasible[picks]).mean(),
                                    mean_health_score=np.nanmean(table['health'][picks]),
                                    diversity=intra_list_diversity([candidates[j] for j in picks]),
                                    acceptability=np.nanmean(table['acceptability'][picks]),
                                    cuisine_typicality=np.nanmean(typicality[cuisine][picks]),
                                    demanding_technique=demanding[picks].mean()))

        base = top_k(gated(table, DEFAULT_WEIGHTS), table['health'], ids)
        record('Multi-criteria, health-gated (proposed)', base)
        top_lists.append(list(ids[base]))
        assignments[-1]['top10'] = ';'.join(ids[base])
        if key not in mmr_cache:
            # Every recipe that meets the references can enter a plan (about 565).
            picks, items = ranked_items(table, ids, 1000)
            pool_recipes = [candidates[j] for j in picks]
            mmr = [position[str(item['recipe_id'])] for item in
                   diversify(items, pool_recipes, k=TOP_K, pool=MMR_POOL)]
            mmr_cache[key] = np.array(mmr)
            plan_cache[key] = weekly_plan(items, pool_recipes, objectives, cuisines=[cuisine],
                                          similar=similar[cuisine], pool=PLAN_POOL)
        record('Multi-criteria + diversity re-ranking (MMR)', mmr_cache[key])
        record('Multi-criteria, without acceptability',
               top_k(gated(table, WITHOUT_ACCEPTABILITY), table['health'], ids))
        plan, checks, notes = plan_cache[key]
        meals = [m for day in plan for m in day]
        step = notes['step']
        plan_rows.append(dict(SEQN=int(test.SEQN[i]), cuisine=cuisine, **checks,
                              all_constraints=all(checks.values()),
                              complete_step1=step == 1, complete_by_step2=step in (1, 2),
                              complete_by_step3=step in (1, 2, 3),
                              repeated_recipes=len(notes['repeated']), borrowed_meals=len(notes['borrowed']),
                              borrowed_from=';'.join(sorted(set(notes['borrowed'].values()))),
                              meals=len(meals), protein_families=len({protein_family(m) for m in meals}),
                              feasible_meals=np.mean([feasible(m) for m in meals]) if meals else np.nan))
        record('Multi-criteria, weighted only',
               top_k(weighted(table, DEFAULT_WEIGHTS), table['health'], ids))
        record('Health score only', top_k(weighted(table, {'health': 1}), table['health'], ids))
        record('Nutrition only (non-personalised)',
               top_k(weighted(generic, {'health': 1}), generic['health'], ids))
        pool = np.flatnonzero(table['ranked'])
        for _ in range(RANDOM_REPEATS):
            record('Random', rng.choice(pool, size=TOP_K, replace=False))

        base_set = set(base)
        for weights in grid:
            picks = top_k(gated(table, {**weights, 'acceptability': accept}), table['health'], ids)
            sensitivity_rows.append(dict(SEQN=int(test.SEQN[i]), **weights,
                                         overlap_with_default=len(base_set & set(picks)) / TOP_K,
                                         all_three=(table['meets_health'][picks] & meets_cuisine[picks]
                                                    & is_feasible[picks]).mean()))
        if n % 50 == 0:
            print(f'  {n} participants', flush=True)

    # The fast path must reproduce the engine's own ranking.
    i = chosen[0]
    objectives, _ = criteria_for(priorities[i])
    cuisine = assignments[0]['cuisine']
    engine, _ = rank_recipes(candidates, objectives, requested_cuisine=cuisine)
    fast = top_k(gated(tables[(signature(priorities[i]), cuisine)], DEFAULT_WEIGHTS),
                 tables[(signature(priorities[i]), cuisine)]['health'], ids)
    if [str(item['recipe_id']) for item in engine[:TOP_K]] != list(ids[fast]):
        raise AssertionError('Evaluation ranking differs from rank_recipes.')

    methods = pd.DataFrame(method_rows)
    summary = methods.groupby('method', sort=False)[['health', 'cuisine', 'feasibility', 'all_three',
                                                     'mean_health_score', 'diversity', 'acceptability',
                                                     'cuisine_typicality', 'demanding_technique']].mean()
    summary.insert(0, 'participants', len(chosen))
    sensitivity = pd.DataFrame(sensitivity_rows)
    grid_summary = sensitivity.groupby(['health', 'cuisine', 'feasibility'])[
        ['overlap_with_default', 'all_three']].mean().reset_index()
    methods.to_csv(OUTPUT_DIR / 'method_comparison_by_patient.csv', index=False)
    summary.to_csv(OUTPUT_DIR / 'method_comparison.csv')
    grid_summary.to_csv(OUTPUT_DIR / 'weight_sensitivity.csv', index=False)
    pd.DataFrame(assignments).to_csv(OUTPUT_DIR / 'simulated_preferences.csv', index=False)
    profiles = pd.Series([a['priorities'] for a in assignments])
    personal = dict(mean_pairwise_top10_overlap_same_cuisine=personalisation(
                        top_lists, [a['cuisine'] for a in assignments]),
                    distinct_priority_profiles=int(profiles.nunique()),
                    share_with_most_common_profile=float(profiles.value_counts(normalize=True).iloc[0]),
                    distinct_top10_lists=len({tuple(sorted(t)) for t in top_lists}),
                    uses_24h_recall=not args.no_diet)
    profiles.value_counts().rename_axis('priorities').rename('participants').to_csv(
        OUTPUT_DIR / 'priority_profiles.csv')
    plans = pd.DataFrame(plan_rows)
    plans.to_csv(OUTPUT_DIR / 'meal_plans_by_patient.csv', index=False)
    plan_summary = plans.drop(columns=['SEQN', 'cuisine', 'borrowed_from']).mean().rename(
        'mean over participants')
    plans.groupby('cuisine')[['complete_step1', 'complete_by_step2', 'complete_by_step3',
                              'borrowed_meals']].mean().to_csv(OUTPUT_DIR / 'meal_plan_by_cuisine.csv')
    plan_summary.to_csv(OUTPUT_DIR / 'meal_plan_summary.csv')
    response = priority_response(assignments, top_lists, candidates)
    response.to_csv(OUTPUT_DIR / 'priority_response.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={'wspace': 0.35})
    for ax, column, title in [(axes[0], 'overlap_with_default', '(a) Top-10 overlap with default weights'),
                              (axes[1], 'all_three', '(b) Top 10 meeting all three criteria')]:
        pivot = grid_summary.pivot(index='health', columns='cuisine', values=column)
        image = ax.imshow(pivot.to_numpy(), origin='lower', cmap='viridis', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(pivot.columns)), [f'{c:.1f}' for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [f'{h:.1f}' for h in pivot.index])
        for (r, c), v in np.ndenumerate(pivot.to_numpy()):
            if not np.isnan(v):
                ax.text(c, r, f'{v:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if v < 0.6 else 'black')
        ax.set_xlabel('Cuisine weight (feasibility = 1 − health − cuisine)')
        ax.set_ylabel('Health weight')
        ax.set_title(title, fontsize=11, loc='left')
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.savefig(OUTPUT_DIR / 'fig_weight_sensitivity.png')
    plt.close(fig)

    with open(OUTPUT_DIR / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(dict(similar_cuisines=similar, participants=len(chosen), eligible_participants=len(with_priority),
                       test_participants=len(test), scope_ages=list(SCOPE_AGES),
                       candidate_recipes=len(candidates), top_k=TOP_K,
                       default_weights=DEFAULT_WEIGHTS, simulated_cuisines=cuisines,
                       random_repeats=RANDOM_REPEATS, seed=SEED,
                       feasibility_limits=dict(minutes=MAX_TIME, ingredients=MAX_INGREDIENTS, steps=MAX_STEPS),
                       personalisation=personal,
                       note='Cuisine preferences are simulated; NHANES does not record them.'), f, indent=2)
    print('\nTop-10 criteria satisfaction (mean over participants):')
    print(summary.to_string(float_format=lambda v: f'{v:.3f}'))
    default = grid_summary.query('health == 0.6 and cuisine == 0.2')
    print('\nWeight sensitivity: mean top-10 overlap with default over the grid '
          f'{grid_summary.overlap_with_default.mean():.2f}; default all-three '
          f'{default.all_three.iloc[0]:.3f}')
    print('\nPersonalisation:', json.dumps(personal, indent=2))
    print('Seven-day plans (share meeting each constraint):')
    print(plan_summary.to_string(float_format=lambda v: f'{v:.3f}'))
    print('Top-10 composition by priority level:')
    print(response.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    print(f'Saved to {OUTPUT_DIR}')


if __name__ == '__main__':
    main()

"""Degree of food processing (FPro) for every USDA SR Legacy food (Weekly Plan, meeting 11).

FoodProX (Menichetti et al., Nature Communications 2023) is a random forest that predicts the NOVA
class of a food (1 unprocessed or minimally processed, 2 processed culinary ingredients, 3 processed,
4 ultra-processed) from its nutrient profile. FPro projects the four class probabilities onto the line
from minimally processed to ultra-processed: FPro = (1 - p1 + p4) / 2, from 0 to 1.

The published scores cover only the SR foods that were in the authors' training table, so the model is
re-trained here on their public data (data/raw_foodprox/FNDDS_SR_combined_58_nutrients.csv from
github.com/menicgiulia/FoodProX-FPro-Score-Generation: FNDDS and SR foods with manual NOVA labels,
58 nutrients as the natural log of grams per 100 g, -20 when absent) and applied to every SR Legacy
food (FoodData Central, April 2018 release). As in the paper, foods are classified with five-fold
stratified cross-validation, so every labelled food gets an out-of-fold FPro; unlabelled foods get the
mean of the five fold models. The re-implementation (scikit-learn, written here) is checked against the
published FPro for the foods in both tables.

    .venv/Scripts/python.exe -m src.recipe_scoring.foodprox
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
FOODPROX_DIR = ROOT / 'data' / 'raw_foodprox'
TRAINING_FILE = FOODPROX_DIR / 'FNDDS_SR_combined_58_nutrients.csv'
PUBLISHED_FILE = FOODPROX_DIR / 'FoodProX_58_full_food_profiles.csv'
SR_DIR = ROOT / 'data' / 'raw_usda' / 'FoodData_Central_sr_legacy_food_csv_2018-04'
OUTPUT_DIR = ROOT / 'data' / 'results' / 'foodprox'
SR_FPRO_FILE = OUTPUT_DIR / 'sr_legacy_fpro.csv'

ABSENT = -20.0
SEED = 42
FOLDS = 5
TREES = 300

# FoodProX column -> SR Legacy nutrient id.
NUTRIENT_IDS = {
    'Protein': 1003, 'Total Fat': 1004, 'Carbohydrate': 1005, 'Alcohol': 1018, 'Water': 1051,
    'Caffeine': 1057, 'Theobromine': 1058, 'Sugars, total': 2000, 'Fiber, total dietary': 1079,
    'Calcium': 1087, 'Iron': 1089, 'Magnesium': 1090, 'Phosphorus': 1091, 'Potassium': 1092,
    'Sodium': 1093, 'Zinc': 1095, 'Copper': 1098, 'Selenium': 1103, 'Retinol': 1105,
    'Carotene, beta': 1107, 'Carotene, alpha': 1108, 'Vitamin E (alpha-tocopherol)': 1109,
    'Cryptoxanthin, beta': 1120, 'Lycopene': 1122, 'Lutein + zeaxanthin': 1123, 'Vitamin C': 1162,
    'Thiamin': 1165, 'Riboflavin': 1166, 'Niacin': 1167, 'Vitamin B-6': 1175, 'Folate, total': 1177,
    'Vitamin B-12': 1178, 'Vitamin K (phylloquinone)': 1185, 'Folic acid': 1186, 'Folate, food': 1187,
    'Cholesterol': 1253, 'Fatty acids, total saturated': 1258, '4:00': 1259, '6:00': 1260, '8:00': 1261,
    '10:00': 1262, '12:00': 1263, '14:00': 1264, '16:00': 1265, '18:00': 1266, '18:01': 1268,
    '18:02': 1269, '18:03': 1270, '20:04': 1271, '22:6 n-3': 1272, '16:01': 1275, '18:04': 1276,
    '20:01': 1277, '20:5 n-3': 1278, '22:01': 1279, '22:5 n-3': 1280,
    'Fatty acids, total monounsaturated': 1292, 'Fatty acids, total polyunsaturated': 1293,
}
FEATURES = list(NUTRIENT_IDS)
TO_GRAMS = {'G': 1.0, 'MG': 1e-3, 'UG': 1e-6}


def fpro(probabilities):
    """FPro from the four NOVA class probabilities (columns p1..p4)."""
    p = np.asarray(probabilities)
    return (1 - p[:, 0] + p[:, 3]) / 2


def log_profile(grams_per_100g):
    """FoodProX features: natural log of grams per 100 g, ABSENT for zero or missing amounts."""
    x = np.asarray(grams_per_100g, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.log(x)
    return np.where(np.isfinite(out), out, ABSENT)


def sr_legacy_profiles(sr_dir=SR_DIR):
    """SR Legacy foods as FoodProX feature rows (index fdc_id) and their descriptions."""
    food = pd.read_csv(sr_dir / 'food.csv', usecols=['fdc_id', 'description']).set_index('fdc_id')
    units = pd.read_csv(sr_dir / 'nutrient.csv', usecols=['id', 'unit_name']).set_index('id').unit_name
    amounts = pd.read_csv(sr_dir / 'food_nutrient.csv', usecols=['fdc_id', 'nutrient_id', 'amount'])
    amounts = amounts[amounts.nutrient_id.isin(NUTRIENT_IDS.values())]
    amounts['grams'] = amounts.amount * amounts.nutrient_id.map(units).map(TO_GRAMS)
    wide = amounts.pivot_table(index='fdc_id', columns='nutrient_id', values='grams', aggfunc='first')
    wide = wide.reindex(index=food.index, columns=list(NUTRIENT_IDS.values()))
    features = pd.DataFrame(log_profile(wide.to_numpy()), index=food.index, columns=FEATURES)
    return features, food.description


def training_table(path=TRAINING_FILE):
    """Labelled foods (NOVA 1-4) with one row per distinct nutrient profile and label."""
    data = pd.read_csv(path)
    data = data[data.novaclass.between(1, 4)]
    return data.drop_duplicates(subset=FEATURES + ['novaclass']).reset_index(drop=True)


def fit_folds(data, seed=SEED, folds=FOLDS, trees=TREES):
    """Five-fold FoodProX: the fold models and out-of-fold class probabilities."""
    x, y = data[FEATURES].to_numpy(), data.novaclass.to_numpy() - 1
    oof = np.zeros((len(data), 4))
    models = []
    for train, test in StratifiedKFold(folds, shuffle=True, random_state=seed).split(x, y):
        model = RandomForestClassifier(n_estimators=trees, class_weight='balanced', n_jobs=-1, random_state=seed)
        model.fit(x[train], y[train])
        oof[test] = model.predict_proba(x[test])
        models.append(model)
    return models, oof


def class_auc(y, probabilities):
    """One-versus-rest ROC AUC for each NOVA class."""
    return {f'NOVA {k + 1}': round(roc_auc_score(y == k, probabilities[:, k]), 3) for k in range(4)}


def score_sr_legacy(models, data, oof, features):
    """FPro for every SR Legacy food: out-of-fold where the food's profile was labelled, else the fold mean."""
    mean = np.mean([m.predict_proba(features.to_numpy()) for m in models], axis=0)
    labelled = {}
    for row, p in zip(data[FEATURES].to_numpy(), oof):
        labelled.setdefault(tuple(np.round(row, 6)), p)
    keys = [tuple(np.round(r, 6)) for r in features.to_numpy()]
    probs = np.array([labelled.get(k, m) for k, m in zip(keys, mean)])
    out = pd.DataFrame(probs, index=features.index, columns=['p1', 'p2', 'p3', 'p4'])
    out['fpro'] = fpro(probs)
    out['out_of_fold'] = [k in labelled for k in keys]
    return out


def main():
    data = training_table()
    print(f'Training foods: {len(data)} (NOVA 1-4: {data.novaclass.value_counts().sort_index().tolist()})')
    models, oof = fit_folds(data)
    auc = class_auc(data.novaclass.to_numpy() - 1, oof)
    print('Out-of-fold AUC:', auc)

    features, descriptions = sr_legacy_profiles()
    scores = score_sr_legacy(models, data, oof, features)
    scores.insert(0, 'description', descriptions)

    published = pd.read_csv(PUBLISHED_FILE, usecols=['Main food description', 'DB', 'FPro'])
    published = published[published.DB == 'SR'].drop_duplicates('Main food description')
    both = scores.merge(published, left_on='description', right_on='Main food description')
    rho = spearmanr(both.fpro, both.FPro).statistic
    mae = float(np.abs(both.fpro - both.FPro).mean())
    print(f'Against published FPro ({len(both)} SR foods in both): Spearman {rho:.3f}, mean absolute difference {mae:.3f}')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores.round(4).to_csv(SR_FPRO_FILE)
    json.dump({'training_foods': len(data), 'folds': FOLDS, 'trees': TREES, 'seed': SEED,
               'out_of_fold_auc': auc, 'sr_legacy_foods': len(scores),
               'sr_foods_out_of_fold': int(scores.out_of_fold.sum()),
               'published_comparison': {'foods': len(both), 'spearman': round(float(rho), 3),
                                        'mean_absolute_difference': round(mae, 3)}},
              open(OUTPUT_DIR / 'foodprox_run.json', 'w'), indent=2)
    print('Wrote', SR_FPRO_FILE)


if __name__ == '__main__':
    main()

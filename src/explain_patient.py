"""Per-patient explanation of the predicted CAP (Proposal section 17.4).

SHAP values from XGBoost's built-in TreeSHAP (pred_contribs; shap's TreeExplainer cannot read
XGBoost 3 models) split one patient's predicted CAP into the average prediction plus one contribution
per model input, in dB/m. Positive contributions raise the patient's CAP above the average; the
contributions sum exactly to the prediction. They describe what drives the model's estimate, not
causes that diet can change.
"""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import xgboost

from src.feature_labels import label
from src.predict_cap import MODEL_FILE, _validated_inputs


@lru_cache(maxsize=None)
def _model(path):
    return joblib.load(path)


def explain_cap(patient, model_file=MODEL_FILE, top=5):
    """Average CAP, the patient's prediction, and the inputs that raise and lower it most."""
    model = _model(str(model_file))
    X = _validated_inputs(pd.DataFrame([patient]), model)
    transformed = model[:-1].transform(X)
    names = list(model[:-1].get_feature_names_out())
    transformed = pd.DataFrame(np.asarray(transformed, dtype=float), columns=names)
    contributions = model[-1].get_booster().predict(xgboost.DMatrix(transformed, missing=np.nan),
                                                     pred_contribs=True)[0]
    base, values = float(contributions[-1]), contributions[:-1]
    rows = [{'feature': name, 'label': label(name), 'value': None if np.isnan(v) else round(float(v), 3),
             'contribution_db_m': round(float(c), 2)}
            for name, v, c in zip(names, transformed.iloc[0], values) if c != 0]
    raising = sorted((r for r in rows if r['contribution_db_m'] > 0), key=lambda r: -r['contribution_db_m'])
    lowering = sorted((r for r in rows if r['contribution_db_m'] < 0), key=lambda r: r['contribution_db_m'])
    return {'average_cap': round(base, 1), 'predicted_cap': round(base + float(values.sum()), 1),
            'raising': raising[:top], 'lowering': lowering[:top],
            'note': 'Contributions explain the model estimate; they are not causes that diet can change.'}

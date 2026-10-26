"""Readable names for model inputs, shared by the thesis figures and patient explanations."""

LABELS = {
    'BMXWAIST': 'Waist circumference', 'BMXBMI': 'BMI', 'BMXWT': 'Body weight', 'BMXHT': 'Height',
    'BMXHIP': 'Hip circumference', 'BMXARMC': 'Arm circumference', 'BMXARML': 'Upper arm length',
    'BMXLEG': 'Upper leg length', 'WHR': 'Waist-to-hip ratio', 'WHtR': 'Waist-to-height ratio',
    'LBXGH': 'HbA1c', 'LBXSGL': 'Glucose (serum)', 'LBXGLU': 'Fasting glucose', 'LBXIN': 'Fasting insulin',
    'HOMA_IR': 'HOMA-IR', 'TyG': 'Triglyceride-glucose index', 'LBXSTR': 'Triglycerides',
    'LBDHDD': 'HDL cholesterol', 'LBXTC': 'Total cholesterol', 'LBXSATSI': 'ALT', 'LBXSASSI': 'AST',
    'LBXSGTSI': 'GGT', 'LBXSUA': 'Uric acid', 'LBXSCR': 'Creatinine', 'RIDAGEYR': 'Age',
    'RIAGENDR': 'Sex', 'RIDRETH3': 'Race/ethnicity', 'LBXHSCRP': 'hs-CRP', 'LBXFER': 'Ferritin',
    'TG_HDL': 'Triglyceride/HDL ratio', 'AST_ALT': 'AST/ALT ratio', 'FLI_proxy': 'FLI (derived)', 'HSI_partial': 'HSI (derived, partial)',
    'LBXWBCSI': 'White cell count', 'LBXPLTSI': 'Platelets', 'LBXSAL': 'Albumin', 'LBXSPH': 'Phosphorus',
    'mean_SBP': 'Systolic blood pressure', 'mean_DBP': 'Diastolic blood pressure',
    'DIQ010': 'Diagnosed diabetes', 'LBXSKSI': 'Potassium', 'LBXSCK': 'Creatine kinase',
    'LBXSBU': 'Blood urea nitrogen', 'LBXSIR': 'Iron', 'LBXSTP': 'Total protein', 'LBXSGB': 'Globulin',
    'DR1TKCAL': 'Energy intake (recall)', 'DR1TPROT': 'Protein intake (recall)', 'DR1TCARB': 'Carbohydrate intake (recall)',
    'DR1TTFAT': 'Fat intake (recall)', 'energy_fraction_protein': 'Share of energy from protein',
    'energy_fraction_carb': 'Share of energy from carbohydrate', 'energy_fraction_fat': 'Share of energy from fat',
    'kcal_per_kg': 'Energy intake per kg', 'LBXSAPSI': 'Alkaline phosphatase', 'LBXSTB': 'Bilirubin',
    'LBXSCA': 'Calcium', 'LBXSNASI': 'Sodium', 'LBXSCLSI': 'Chloride', 'LBXSC3SI': 'Bicarbonate',
    'LBXSLDSI': 'Lactate dehydrogenase', 'TC_HDL': 'Total/HDL cholesterol ratio', 'INDFMPIR': 'Family income-to-poverty ratio',
    'DMDEDUC2': 'Education', 'DMDBORN4': 'Country of birth', 'LBXLYPCT': 'Lymphocytes (%)',
    'LBXNEPCT': 'Neutrophils (%)', 'LBXMOPCT': 'Monocytes (%)', 'LBXHGB': 'Haemoglobin', 'LBXHCT': 'Haematocrit',
    'LBXMCVSI': 'Mean cell volume', 'LBXRDW': 'Red cell distribution width', 'LBXMPSI': 'Mean platelet volume',
    'BPXOSY1': 'Systolic blood pressure (1st)', 'BPXOSY2': 'Systolic blood pressure (2nd)',
    'BPXOSY3': 'Systolic blood pressure (3rd)', 'BPXODI1': 'Diastolic blood pressure (1st)',
    'BPXODI2': 'Diastolic blood pressure (2nd)', 'BPXODI3': 'Diastolic blood pressure (3rd)', 'BPXOPLS1': 'Pulse',
    'DIQ160': 'Told prediabetes', 'DIQ050': 'Taking insulin', 'DIQ070': 'Taking diabetes pills',
    'BPQ020': 'Told high blood pressure', 'BPQ080': 'Told high cholesterol', 'WHQ070': 'Tried to lose weight',
    'WHD050': 'Weight a year ago', 'HUQ010': 'Self-rated health', 'SLD012': 'Sleep hours', 'SMQ020': 'Ever smoked',
    'SMQ040': 'Smokes now', 'ALQ111': 'Ever drank alcohol', 'ALQ121': 'Alcohol frequency', 'ALQ130': 'Drinks per drinking day',
    'ALQ151': 'Ever heavy drinking', 'PAD680': 'Sedentary minutes', 'weight_change_1y': 'Weight change in the past year',
}


def label(name):
    base = name.split('__')[-1]
    for code, text in LABELS.items():
        if base == code:
            return text
        if base.startswith(code + '_'):
            return f'{text} = {base[len(code) + 1:]}'
    return base

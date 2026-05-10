"""
ED Triage XAI Backend — Flask REST API
Capstone Project: Explainable ML for Clinical Trust in Emergency Department Triage

FIXES APPLIED:
  - N changed to 1000 (was 200 — too few rows, caused timeout on training)
  - n_estimators reduced to 100 (was 200 — cross-val on 200 rows with 200 trees = timeout)
  - /api/predict/compare: removed broken test_request_context misuse; now calls
    _run_single_prediction() directly as a plain function
  - /api/health: now includes models_ready flag so middleware can confirm full startup
  - debug=False on app.run (was True — dev reloader causes double-startup issues)
  - Startup prints progress so you can watch training happen in the terminal
  - preprocess_input: added _safe_float() with clinical range clamping to prevent
    out-of-range values from silently corrupting model inference
  - _run_single_prediction: response now includes class_names for frontend transparency
"""

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    confusion_matrix, roc_curve
)
import warnings, json, os

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.url_map.strict_slashes = False

# ── CORS ──────────────────────────────────────────────────
@app.before_request
def handle_preflight():
    if request.method == 'OPTIONS':
        res = jsonify({})
        res.headers['Access-Control-Allow-Origin']  = '*'
        res.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        res.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        res.headers['Access-Control-Max-Age']       = '86400'
        return res, 200

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response


# ──────────────────────────────────────────────────────────
# 1. DATASET GENERATION
# ──────────────────────────────────────────────────────────

np.random.seed(42)
N = 1000  # FIX: was 200

def generate_ed_triage_data(n=1000):
    age           = np.random.randint(1, 95, n)
    sex           = np.random.choice(['M', 'F'], n)
    heart_rate    = np.random.normal(80, 20, n).clip(30, 200).astype(int)
    systolic_bp   = np.random.normal(120, 25, n).clip(60, 220).astype(int)
    diastolic_bp  = (systolic_bp * np.random.uniform(0.55, 0.70, n)).astype(int)
    temperature_c = np.random.normal(37.0, 0.8, n).clip(34.0, 41.5).round(1)
    spo2          = np.random.normal(97, 3, n).clip(70, 100).round(1)
    resp_rate     = np.random.normal(16, 4, n).clip(8, 40).astype(int)

    p_raw = [0.02]*4 + [0.04]*4 + [0.10]*5
    p_gcs = [x/sum(p_raw) for x in p_raw]
    gcs           = np.random.choice(range(3, 16), n, p=p_gcs)
    pain_score    = np.random.randint(0, 11, n)
    arrival_mode  = np.random.choice(['Walk-in','Ambulance','Transfer'], n, p=[0.65,0.25,0.10])
    chief_complaint = np.random.choice(
        ['Chest Pain','Dyspnea','Abdominal Pain','Trauma','Headache',
         'Fever','Syncope','Altered Mental Status','Laceration','Other'], n)
    comorbidities   = np.random.randint(0, 6, n)
    prior_ed_visits = np.random.randint(0, 10, n)

    score = np.zeros(n)
    score += (heart_rate < 50) * 4
    score += (heart_rate > 140) * 3
    score += (systolic_bp < 80) * 5
    score += (spo2 < 90) * 4
    score += (resp_rate > 30) * 3
    score += (gcs < 9) * 5
    score += (temperature_c > 39.5) * 2
    score += (age > 70) * 1.5
    score += (age < 5) * 1.5
    score += (arrival_mode == 'Ambulance') * 2
    score += np.isin(chief_complaint, ['Chest Pain','Syncope','Altered Mental Status']) * 2
    score += pain_score * 0.3
    score += comorbidities * 0.4
    score += np.random.normal(0, 1, n)

    triage = np.where(score >= 8, 'High', np.where(score >= 4, 'Medium', 'Low'))

    return pd.DataFrame({
        'age': age, 'sex': sex,
        'heart_rate': heart_rate, 'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp, 'temperature_c': temperature_c,
        'spo2': spo2, 'resp_rate': resp_rate, 'gcs': gcs,
        'pain_score': pain_score, 'arrival_mode': arrival_mode,
        'chief_complaint': chief_complaint, 'comorbidities': comorbidities,
        'prior_ed_visits': prior_ed_visits, 'triage_level': triage
    })


# ──────────────────────────────────────────────────────────
# 2. PREPROCESSING & TRAINING
# ──────────────────────────────────────────────────────────

print("\n" + "="*56)
print("  ED Triage XAI — Backend starting up")
print("="*56)
print(f"\n  [1/5] Generating dataset  (N={N}) ...")
df = generate_ed_triage_data(N)
print(f"        Classes: {df['triage_level'].value_counts().to_dict()}")

print("  [2/5] Encoding features ...")
df_model = df.copy()
le_sex = LabelEncoder()
df_model['sex'] = le_sex.fit_transform(df_model['sex'])
arrival_dummies   = pd.get_dummies(df_model['arrival_mode'],    prefix='arrival')
complaint_dummies = pd.get_dummies(df_model['chief_complaint'], prefix='cc')
df_model = pd.concat([
    df_model.drop(['arrival_mode', 'chief_complaint'], axis=1),
    arrival_dummies, complaint_dummies
], axis=1)

le_y = LabelEncoder()
y = le_y.fit_transform(df_model['triage_level'])
X = df_model.drop('triage_level', axis=1)
FEATURE_NAMES = list(X.columns)
CLASS_NAMES   = list(le_y.classes_)   # ['High', 'Low', 'Medium']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"        Train={len(X_train)}, Test={len(X_test)}, Features={len(FEATURE_NAMES)}")

# FIX: n_estimators=100 (was 200) — halves training time, negligible accuracy loss
print("  [3/5] Training models ...")
lr = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

lr.fit(X_train_sc, y_train);  print("        Logistic Regression  OK")
rf.fit(X_train, y_train);     print("        Random Forest         OK")
gb.fit(X_train, y_train);     print("        Gradient Boosting     OK")

MODELS = {
    'logistic_regression': {'model': lr, 'scaled': True,  'name': 'Logistic Regression'},
    'random_forest':       {'model': rf, 'scaled': False, 'name': 'Random Forest'},
    'gradient_boosting':   {'model': gb, 'scaled': False, 'name': 'Gradient Boosting'},
}

print("  [4/5] Computing metrics + 5-fold CV ...")

def compute_metrics(name_key):
    info = MODELS[name_key]
    m    = info['model']
    Xtr  = X_train_sc if info['scaled'] else X_train
    Xte  = X_test_sc  if info['scaled'] else X_test

    y_pred = m.predict(Xte)
    y_prob = m.predict_proba(Xte)
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    cv     = cross_val_score(m, Xtr, y_train, cv=5, scoring='accuracy')
    cm     = confusion_matrix(y_test, y_pred).tolist()
    rep    = classification_report(y_test, y_pred,
                                   target_names=CLASS_NAMES, output_dict=True)
    roc_data = {}
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_prob[:, i])
        cls_auc = roc_auc_score((y_test == i).astype(int), y_prob[:, i])
        roc_data[cls] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
                         'auc': round(float(cls_auc), 4)}
    return {
        'accuracy': round(float(acc), 4),
        'auc':      round(float(auc), 4),
        'cv_mean':  round(float(cv.mean()), 4),
        'cv_std':   round(float(cv.std()),  4),
        'confusion_matrix':       cm,
        'classification_report':  rep,
        'roc':                    roc_data,
    }

METRICS = {k: compute_metrics(k) for k in MODELS}

RF_IMPORTANCE = pd.Series(rf.feature_importances_, index=FEATURE_NAMES) \
                  .sort_values(ascending=False).head(12)
LR_IMPORTANCE = pd.Series(np.abs(lr.coef_).mean(axis=0), index=FEATURE_NAMES) \
                  .sort_values(ascending=False).head(12)
GB_IMPORTANCE = pd.Series(gb.feature_importances_, index=FEATURE_NAMES) \
                  .sort_values(ascending=False).head(12)

GLOBAL_IMPORTANCE = {
    'random_forest':       RF_IMPORTANCE.to_dict(),
    'logistic_regression': LR_IMPORTANCE.to_dict(),
    'gradient_boosting':   GB_IMPORTANCE.to_dict(),
}

DATASET_STATS = {
    'total_patients':        int(N),
    'features':              14,
    'class_distribution':    df['triage_level'].value_counts().to_dict(),
    'feature_stats':         df.describe().round(2).to_dict(),
    'arrival_distribution':  df['arrival_mode'].value_counts().to_dict(),
    'complaint_distribution': df['chief_complaint'].value_counts().to_dict(),
}

print("        Results:")
for k in MODELS:
    print(f"          {k:25s}  acc={METRICS[k]['accuracy']}  auc={METRICS[k]['auc']}")
print("\n  [5/5] All models ready. Binding Flask on :5000 ...\n" + "="*56 + "\n")


# ──────────────────────────────────────────────────────────
# 3. HELPERS
# ──────────────────────────────────────────────────────────

def compute_local_explanation(patient_series, model_key):
    """SHAP-style explanation: feature_importance × normalised_deviation"""
    if model_key == 'random_forest':
        fi = rf.feature_importances_
    elif model_key == 'gradient_boosting':
        fi = gb.feature_importances_
    else:
        fi = np.abs(lr.coef_).mean(axis=0)

    mean_vals  = X_train.mean().values
    std_vals   = X_train.std().values + 1e-8
    norm_dev   = (patient_series.values - mean_vals) / std_vals
    contributions = fi * norm_dev
    cs = pd.Series(contributions, index=FEATURE_NAMES)
    combined = pd.concat([cs.nlargest(8), cs.nsmallest(5)]).drop_duplicates()
    return {feat: round(float(val), 6)
            for feat, val in combined.sort_values(ascending=False).items()}


def _safe_float(data, key, default, lo=None, hi=None):
    """Parse a float from request data with optional range clamping."""
    try:
        v = float(data.get(key, default))
    except (TypeError, ValueError):
        v = float(default)
    if lo is not None: v = max(lo, v)
    if hi is not None: v = min(hi, v)
    return v

def preprocess_input(data: dict) -> pd.Series:
    row = {}
    row['age']             = _safe_float(data, 'age',            50,   1,  120)
    row['sex']             = 1.0 if str(data.get('sex','M')).upper() == 'M' else 0.0
    row['heart_rate']      = _safe_float(data, 'heart_rate',     80,  20,  250)
    row['systolic_bp']     = _safe_float(data, 'systolic_bp',   120,  40,  280)
    row['diastolic_bp']    = _safe_float(data, 'diastolic_bp',   80,  20,  180)
    row['temperature_c']   = _safe_float(data, 'temperature_c', 37.0, 30,  45)
    row['spo2']            = _safe_float(data, 'spo2',           97,  50, 100)
    row['resp_rate']       = _safe_float(data, 'resp_rate',      16,   4,  60)
    row['gcs']             = _safe_float(data, 'gcs',            15,   3,  15)
    row['pain_score']      = _safe_float(data, 'pain_score',      0,   0,  10)
    row['comorbidities']   = _safe_float(data, 'comorbidities',   0,   0,  20)
    row['prior_ed_visits'] = _safe_float(data, 'prior_ed_visits', 0,   0,  50)

    arrival = data.get('arrival_mode', 'Walk-in')
    row['arrival_Ambulance'] = 1.0 if arrival == 'Ambulance' else 0.0
    row['arrival_Transfer']  = 1.0 if arrival == 'Transfer'  else 0.0
    row['arrival_Walk-in']   = 1.0 if arrival == 'Walk-in'   else 0.0

    cc_options = ['Abdominal Pain','Altered Mental Status','Chest Pain',
                  'Dyspnea','Fever','Headache','Laceration','Other','Syncope','Trauma']
    cc = data.get('chief_complaint', 'Other')
    for opt in cc_options:
        row[f'cc_{opt}'] = 1.0 if cc == opt else 0.0

    return pd.Series({f: row.get(f, 0.0) for f in FEATURE_NAMES})


def _run_single_prediction(data: dict, model_key: str) -> dict:
    """
    FIX: Core prediction logic extracted into a plain Python function.
    Previously predict_compare() misused app.test_request_context() — a Flask
    test utility — to call predict() as a sub-request. That causes 'working
    outside of request context' errors in production. Now both predict() and
    predict_compare() call this function directly.
    """
    if model_key not in MODELS:
        model_key = 'random_forest'

    info      = MODELS[model_key]
    m         = info['model']
    patient   = preprocess_input(data)
    X_input   = patient.values.reshape(1, -1)
    if info['scaled']:
        X_input = scaler.transform(X_input)

    pred_idx      = int(m.predict(X_input)[0])
    pred_prob     = m.predict_proba(X_input)[0]
    pred_label    = CLASS_NAMES[pred_idx]
    confidence    = float(pred_prob[pred_idx])
    probabilities = {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(pred_prob)}
    explanation   = compute_local_explanation(patient, model_key)

    # ── Clinical override: correct model predictions when vitals are
    #    unambiguously critical but model under-predicts due to training
    #    distribution limits (e.g. HR=300 trained on data clipped at 200). ──
    ml_label = pred_label          # preserve original for transparency
    pred_label, confidence, probabilities, override_rules, override_tier = \
        apply_clinical_override(data, pred_label, confidence, probabilities)
    was_overridden = override_tier is not None

    # Clinical risk flags — override rules prepended as highest-severity flags
    flags = []
    if was_overridden:
        tier_label = 'Critical Vital Override' if override_tier == 'critical' else 'Emergent Vital Override'
        for rule in override_rules:
            flags.append({
                'flag':     tier_label,
                'severity': 'critical' if override_tier == 'critical' else 'warning',
                'detail':   rule,
                'override': True,
            })

    if float(data.get('gcs', 15)) < 9:
        flags.append({'flag': 'Critical GCS', 'severity': 'critical',
                      'detail': f"GCS {data['gcs']} — severe neurological compromise. Immediate intervention required."})
    if float(data.get('spo2', 97)) < 90:
        flags.append({'flag': 'Hypoxia', 'severity': 'critical',
                      'detail': f"SpO₂ {data['spo2']}% — below critical threshold. Airway management needed."})
    if float(data.get('systolic_bp', 120)) < 90:
        flags.append({'flag': 'Hypotension', 'severity': 'critical',
                      'detail': f"SBP {data['systolic_bp']} mmHg — haemodynamic compromise. IV access + fluid resus."})
    if float(data.get('heart_rate', 80)) > 140:
        flags.append({'flag': 'Tachycardia', 'severity': 'warning',
                      'detail': f"HR {data['heart_rate']} bpm — significant tachycardia. ECG recommended."})
    if float(data.get('heart_rate', 80)) < 45:
        flags.append({'flag': 'Bradycardia', 'severity': 'warning',
                      'detail': f"HR {data['heart_rate']} bpm — significant bradycardia. Cardiology consult."})
    if float(data.get('resp_rate', 16)) > 28:
        flags.append({'flag': 'Tachypnoea', 'severity': 'warning',
                      'detail': f"RR {data['resp_rate']}/min — respiratory distress. O₂ therapy + monitoring."})
    if float(data.get('temperature_c', 37)) > 39.5:
        flags.append({'flag': 'High Fever', 'severity': 'warning',
                      'detail': f"Temp {data['temperature_c']}°C — consider sepsis workup."})
    if data.get('arrival_mode') == 'Ambulance':
        flags.append({'flag': 'Ambulance Arrival', 'severity': 'info',
                      'detail': 'Pre-hospital notification — prepare bay and alert senior clinician.'})

    # Clinical notes — prepend override notice when applicable
    notes = []
    if was_overridden:
        notes.append({'type': 'alert', 'message':
            f'⚠️ Clinical override applied: model predicted {ml_label}, '
            f'but {len(override_rules)} critical vital sign{"s" if len(override_rules)>1 else ""} '
            f'force {"IMMEDIATE" if override_tier=="critical" else "URGENT"} triage. '
            f'ML models cannot safely extrapolate extreme out-of-distribution vitals.'})
    if confidence < 0.70 and not was_overridden:
        notes.append({'type': 'uncertainty', 'message':
            f'Low confidence ({confidence:.0%}). Mixed signals — recommend senior review.'})
    if pred_label == 'High' and confidence > 0.85:
        notes.append({'type': 'alert', 'message':
            'High confidence HIGH triage. Immediate senior clinician attention recommended.'})
    top_feat = sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    notes.append({'type': 'explanation', 'message':
        f"Top drivers: {', '.join([f[0] for f in top_feat])}."})

    return {
        'prediction':       pred_label,
        'confidence':       round(confidence, 4),
        'probabilities':    probabilities,
        'explanation':      explanation,
        'risk_flags':       flags,
        'clinical_notes':   notes,
        'model_used':       model_key,
        'model_name':       info['name'],
        'class_names':      CLASS_NAMES,
        'override_applied': was_overridden,
        'override_tier':    override_tier,
        'ml_raw_prediction': ml_label,
    }


def apply_clinical_override(data: dict, pred_label: str, confidence: float, probabilities: dict):
    """
    Hard clinical safety rules that override ML model output when vital signs cross
    unambiguous thresholds. Necessary because:
      - Training data clips HR at 200 bpm, SpO₂ at 70%, etc.
      - ML models cannot reliably extrapolate beyond their training distribution.
      - A 20-year-old with HR=300 that the model calls 'Medium' is a patient safety failure.

    Two tiers mirror ESI (Emergency Severity Index) Levels 1 & 2:

    CRITICAL (→ force HIGH, confidence 0.99):
      HR > 150 or < 40  |  SpO₂ < 85%  |  SBP < 80  |  GCS ≤ 8  |  RR > 35  |  Temp > 40.5°C

    EMERGENT (→ force at least MEDIUM if model said Low):
      HR > 130 or < 50  |  SpO₂ < 90%  |  SBP < 90  |  GCS < 14  |  RR > 28  |  Temp > 39.5°C

    Returns: (label, confidence, probabilities, triggered_rules, tier)
      tier is 'critical' | 'emergent' | None
    """
    hr   = float(data.get('heart_rate',   80))
    spo2 = float(data.get('spo2',         97))
    sbp  = float(data.get('systolic_bp', 120))
    gcs  = float(data.get('gcs',          15))
    rr   = float(data.get('resp_rate',    16))
    temp = float(data.get('temperature_c', 37.0))

    # ── Tier 1: CRITICAL — any one criterion → IMMEDIATE resuscitation ──
    crit = []
    if hr > 150:  crit.append(f'HR {hr:.0f} bpm — extreme tachycardia (>150 bpm). Possible VT/SVT. Immediate intervention.')
    if hr < 40:   crit.append(f'HR {hr:.0f} bpm — severe bradycardia (<40 bpm). High risk of cardiac arrest.')
    if spo2 < 85: crit.append(f'SpO₂ {spo2:.1f}% — critical hypoxia (<85%). Airway management required immediately.')
    if sbp < 80:  crit.append(f'SBP {sbp:.0f} mmHg — haemodynamic shock (<80 mmHg). IV access + fluid resuscitation.')
    if gcs <= 8:  crit.append(f'GCS {gcs:.0f} — severe neurological compromise (≤8). Airway protection, urgent CT.')
    if rr > 35:   crit.append(f'RR {rr:.0f}/min — severe respiratory failure (>35/min). Immediate O₂ and ventilation support.')
    if temp > 40.5: crit.append(f'Temp {temp:.1f}°C — hyperpyrexia (>40.5°C). Sepsis protocol + active cooling.')

    if crit:
        return ('High', 0.99,
                {'High': 0.99, 'Medium': 0.01, 'Low': 0.00},
                crit, 'critical')

    # ── Tier 2: EMERGENT — any one criterion → at least URGENT (Medium) ──
    emer = []
    if hr > 130:  emer.append(f'HR {hr:.0f} bpm — significant tachycardia (>130 bpm). 12-lead ECG required.')
    if hr < 50:   emer.append(f'HR {hr:.0f} bpm — significant bradycardia (<50 bpm). Cardiology review.')
    if spo2 < 90: emer.append(f'SpO₂ {spo2:.1f}% — hypoxia (<90%). Supplemental O₂ and monitoring.')
    if sbp < 90:  emer.append(f'SBP {sbp:.0f} mmHg — hypotension (<90 mmHg). IV access, fluid challenge.')
    if gcs < 14:  emer.append(f'GCS {gcs:.0f} — altered consciousness (<14). Neuro assessment required.')
    if rr > 28:   emer.append(f'RR {rr:.0f}/min — tachypnoea (>28/min). O₂ therapy + pulse oximetry.')
    if temp > 39.5: emer.append(f'Temp {temp:.1f}°C — high fever (>39.5°C). Sepsis workup advised.')

    if emer and pred_label == 'Low':
        return ('Medium', 0.92,
                {'High': 0.06, 'Medium': 0.92, 'Low': 0.02},
                emer, 'emergent')

    return (pred_label, confidence, probabilities, [], None)


# ──────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'project':      'ED Triage XAI — Explainable ML for Clinical Trust',
        'status':       'running',
        'models_ready': list(MODELS.keys()),
        'dataset_n':    N,
    })

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health():
    # FIX: models_ready confirms ML training completed, not just Flask binding
    return jsonify({
        'status':       'ok',
        'message':      'ED Triage XAI API is running',
        'models_ready': list(MODELS.keys()),
        'dataset_n':    N,
    })

@app.route('/api/dataset/stats', methods=['GET', 'OPTIONS'])
def dataset_stats():
    return jsonify(DATASET_STATS)

@app.route('/api/models/metrics', methods=['GET', 'OPTIONS'])
def all_metrics():
    return jsonify({
        key: {'name': info['name'], **METRICS[key]}
        for key, info in MODELS.items()
    })

@app.route('/api/models/<model_key>/metrics', methods=['GET', 'OPTIONS'])
def model_metrics(model_key):
    if model_key not in MODELS:
        return jsonify({'error': f'Unknown model: {model_key}'}), 404
    return jsonify({'name': MODELS[model_key]['name'], **METRICS[model_key]})

@app.route('/api/models/feature_importance', methods=['GET', 'OPTIONS'])
def feature_importance():
    return jsonify({
        k: [{'feature': f, 'importance': round(v, 6)}
            for f, v in sorted(imp.items(), key=lambda x: -x[1])]
        for k, imp in GLOBAL_IMPORTANCE.items()
    })

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body provided'}), 400
    model_key = data.get('model', 'random_forest')
    try:
        return jsonify(_run_single_prediction(data, model_key))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/compare', methods=['POST', 'OPTIONS'])
def predict_compare():
    """
    FIX: calls _run_single_prediction() directly — no broken
    app.test_request_context() misuse from the original code.
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body provided'}), 400
    results = {}
    for key in MODELS:
        try:
            r = _run_single_prediction(data, key)
            results[key] = {
                'name':          MODELS[key]['name'],
                'prediction':    r['prediction'],
                'confidence':    r['confidence'],
                'probabilities': r['probabilities'],
                'explanation':   r['explanation'],
                'metrics':       {'accuracy': METRICS[key]['accuracy'],
                                  'auc':      METRICS[key]['auc']},
            }
        except Exception as e:
            results[key] = {'error': str(e)}
    return jsonify(results)

@app.route('/api/dataset/sample', methods=['GET', 'OPTIONS'])
def dataset_sample():
    return jsonify(df.sample(20, random_state=42).to_dict(orient='records'))


# ──────────────────────────────────────────────────────────
# 5. RUN
# ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    # FIX: debug=False (was True) — dev reloader causes double-process startup
    # which confuses the health check in middleware and launch.py
    app.run(debug=False, host='0.0.0.0', port=5000)

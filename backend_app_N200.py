"""
ED Triage XAI Backend — Flask REST API
Capstone Project: Explainable ML for Clinical Trust in Emergency Department Triage
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
import warnings
import json
import os

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.url_map.strict_slashes = False

# ── CORS: handle preflight OPTIONS for every route automatically ──
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
# 1. DATASET GENERATION (runs once at startup)
# ──────────────────────────────────────────────────────────

np.random.seed(42)
N = 200

def generate_ed_triage_data(n=2000):
    age        = np.random.randint(1, 95, n)
    sex        = np.random.choice(['M', 'F'], n)
    heart_rate = np.random.normal(80, 20, n).clip(30, 200).astype(int)
    systolic_bp  = np.random.normal(120, 25, n).clip(60, 220).astype(int)
    diastolic_bp = (systolic_bp * np.random.uniform(0.55, 0.70, n)).astype(int)
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
# 2. PREPROCESSING & TRAINING (runs once at startup)
# ──────────────────────────────────────────────────────────

df = generate_ed_triage_data(N)

df_model = df.copy()
le_sex = LabelEncoder()
df_model['sex'] = le_sex.fit_transform(df_model['sex'])
arrival_dummies   = pd.get_dummies(df_model['arrival_mode'],     prefix='arrival')
complaint_dummies = pd.get_dummies(df_model['chief_complaint'],  prefix='cc')
df_model = pd.concat([
    df_model.drop(['arrival_mode', 'chief_complaint'], axis=1),
    arrival_dummies, complaint_dummies
], axis=1)

le_y = LabelEncoder()
y = le_y.fit_transform(df_model['triage_level'])   # High=0, Low=1, Medium=2
X = df_model.drop('triage_level', axis=1)
FEATURE_NAMES = list(X.columns)
CLASS_NAMES   = list(le_y.classes_)   # ['High', 'Low', 'Medium']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Train all three models
lr  = LogisticRegression(max_iter=1000, random_state=42)
rf  = RandomForestClassifier(n_estimators=200, random_state=42)
gb  = GradientBoostingClassifier(n_estimators=200, random_state=42)

lr.fit(X_train_sc, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

MODELS = {
    'logistic_regression': {'model': lr, 'scaled': True,  'name': 'Logistic Regression'},
    'random_forest':       {'model': rf, 'scaled': False, 'name': 'Random Forest'},
    'gradient_boosting':   {'model': gb, 'scaled': False, 'name': 'Gradient Boosting'},
}

# Pre-compute metrics
def compute_metrics(name_key):
    info = MODELS[name_key]
    m = info['model']
    Xtr = X_train_sc if info['scaled'] else X_train
    Xte = X_test_sc  if info['scaled'] else X_test

    y_pred = m.predict(Xte)
    y_prob = m.predict_proba(Xte)
    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    cv   = cross_val_score(m, Xtr, y_train, cv=5, scoring='accuracy')
    cm   = confusion_matrix(y_test, y_pred).tolist()
    rep  = classification_report(y_test, y_pred,
                                  target_names=CLASS_NAMES, output_dict=True)
    # Per-class ROC
    roc_data = {}
    for i, cls in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_prob[:, i])
        cls_auc = roc_auc_score((y_test == i).astype(int), y_prob[:, i])
        roc_data[cls] = {
            'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': round(float(cls_auc), 4)
        }
    return {
        'accuracy': round(float(acc), 4),
        'auc': round(float(auc), 4),
        'cv_mean': round(float(cv.mean()), 4),
        'cv_std':  round(float(cv.std()),  4),
        'confusion_matrix': cm,
        'classification_report': rep,
        'roc': roc_data,
    }

METRICS = {k: compute_metrics(k) for k in MODELS}

# Global feature importance
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

# Dataset stats
DATASET_STATS = {
    'total_patients': int(N),
    'features': 14,
    'class_distribution': df['triage_level'].value_counts().to_dict(),
    'feature_stats': df.describe().round(2).to_dict(),
    'arrival_distribution': df['arrival_mode'].value_counts().to_dict(),
    'complaint_distribution': df['chief_complaint'].value_counts().to_dict(),
}

print("✅ All models trained and metrics computed.")
for k in MODELS:
    print(f"  {k}: acc={METRICS[k]['accuracy']}, auc={METRICS[k]['auc']}")


# ──────────────────────────────────────────────────────────
# 3. HELPER: LOCAL XAI EXPLANATION
# ──────────────────────────────────────────────────────────

def compute_local_explanation(patient_series, model_key):
    """
    Approximates SHAP-style local explanation using:
    feature_importance × normalised_deviation_from_mean
    Returns top features pushing toward each class.
    """
    info = MODELS[model_key]
    m    = info['model']

    if model_key == 'random_forest':
        fi = rf.feature_importances_
    elif model_key == 'gradient_boosting':
        fi = gb.feature_importances_
    else:  # LR — use absolute mean coef
        fi = np.abs(lr.coef_).mean(axis=0)

    mean_vals = X_train.mean().values
    std_vals  = X_train.std().values + 1e-8
    norm_dev  = (patient_series.values - mean_vals) / std_vals

    contributions = fi * norm_dev
    contrib_series = pd.Series(contributions, index=FEATURE_NAMES)
    top_pos = contrib_series.nlargest(8)
    top_neg = contrib_series.nsmallest(5)
    combined = pd.concat([top_pos, top_neg]).drop_duplicates()

    return {
        feat: round(float(val), 6)
        for feat, val in combined.sort_values(ascending=False).items()
    }


def preprocess_input(data: dict) -> pd.Series:
    """Convert raw API input dict → model-ready feature Series."""
    row = {}
    row['age']           = float(data.get('age', 50))
    row['sex']           = 1.0 if str(data.get('sex','M')).upper() == 'M' else 0.0
    row['heart_rate']    = float(data.get('heart_rate', 80))
    row['systolic_bp']   = float(data.get('systolic_bp', 120))
    row['diastolic_bp']  = float(data.get('diastolic_bp', 80))
    row['temperature_c'] = float(data.get('temperature_c', 37.0))
    row['spo2']          = float(data.get('spo2', 97.0))
    row['resp_rate']     = float(data.get('resp_rate', 16))
    row['gcs']           = float(data.get('gcs', 15))
    row['pain_score']    = float(data.get('pain_score', 0))
    row['comorbidities'] = float(data.get('comorbidities', 0))
    row['prior_ed_visits'] = float(data.get('prior_ed_visits', 0))

    # Arrival mode dummies
    arrival = data.get('arrival_mode', 'Walk-in')
    row['arrival_Ambulance'] = 1.0 if arrival == 'Ambulance' else 0.0
    row['arrival_Transfer']  = 1.0 if arrival == 'Transfer'  else 0.0
    row['arrival_Walk-in']   = 1.0 if arrival == 'Walk-in'   else 0.0

    # Chief complaint dummies
    cc_options = ['Abdominal Pain','Altered Mental Status','Chest Pain',
                  'Dyspnea','Fever','Headache','Laceration','Other','Syncope','Trauma']
    cc = data.get('chief_complaint', 'Other')
    for opt in cc_options:
        row[f'cc_{opt}'] = 1.0 if cc == opt else 0.0

    # Build series aligned to FEATURE_NAMES
    series = pd.Series({f: row.get(f, 0.0) for f in FEATURE_NAMES})
    return series


# ──────────────────────────────────────────────────────────
# 4. API ROUTES
# ──────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def home():
    """Homepage — shows all available API endpoints."""
    return jsonify({
        'project':     'ED Triage XAI — Explainable ML for Clinical Trust',
        'status':      '✅ Backend is running',
        'models_ready': list(MODELS.keys()),
        'endpoints': {
            'GET  /':                              'This help page',
            'GET  /api/health':                    'Health check',
            'GET  /api/dataset/stats':             'Dataset summary statistics',
            'GET  /api/dataset/sample':            'Sample 20 patient rows',
            'GET  /api/models/metrics':            'All model performance metrics',
            'GET  /api/models/<key>/metrics':      'Single model metrics (logistic_regression | random_forest | gradient_boosting)',
            'GET  /api/models/feature_importance': 'Global SHAP feature importance',
            'POST /api/predict':                   'Triage prediction + XAI explanation',
            'POST /api/predict/compare':           'Compare all 3 models on same patient',
        },
        'example_post_body': {
            'age': 65, 'sex': 'M', 'heart_rate': 110, 'systolic_bp': 85,
            'diastolic_bp': 55, 'temperature_c': 37.2, 'spo2': 88.0,
            'resp_rate': 26, 'gcs': 10, 'pain_score': 7,
            'arrival_mode': 'Ambulance', 'chief_complaint': 'Chest Pain',
            'comorbidities': 3, 'prior_ed_visits': 2,
            'model': 'random_forest'
        }
    })


@app.route('/favicon.ico', methods=['GET'])
def favicon():
    """Suppress browser favicon 404."""
    return '', 204


@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health():
    return jsonify({'status': 'ok', 'message': 'ED Triage XAI API is running'})


@app.route('/api/dataset/stats', methods=['GET', 'OPTIONS'])
def dataset_stats():
    """Return summary statistics about the training dataset."""
    return jsonify(DATASET_STATS)


@app.route('/api/models/metrics', methods=['GET', 'OPTIONS'])
def all_metrics():
    """Return performance metrics for all three models."""
    result = {}
    for key, info in MODELS.items():
        result[key] = {
            'name': info['name'],
            **METRICS[key]
        }
    return jsonify(result)


@app.route('/api/models/<model_key>/metrics', methods=['GET', 'OPTIONS'])
def model_metrics(model_key):
    """Return metrics for a specific model."""
    if model_key not in MODELS:
        return jsonify({'error': f'Unknown model: {model_key}'}), 404
    return jsonify({'name': MODELS[model_key]['name'], **METRICS[model_key]})


@app.route('/api/models/feature_importance', methods=['GET', 'OPTIONS'])
def feature_importance():
    """Return global feature importances for all models."""
    return jsonify({
        k: [{'feature': f, 'importance': round(v, 6)}
            for f, v in sorted(imp.items(), key=lambda x: -x[1])]
        for k, imp in GLOBAL_IMPORTANCE.items()
    })


@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    """
    Main prediction endpoint.

    Body (JSON):
    {
      "age": 65, "sex": "M", "heart_rate": 110, "systolic_bp": 85,
      "diastolic_bp": 55, "temperature_c": 37.2, "spo2": 88.0,
      "resp_rate": 26, "gcs": 10, "pain_score": 7,
      "arrival_mode": "Ambulance", "chief_complaint": "Chest Pain",
      "comorbidities": 3, "prior_ed_visits": 2,
      "model": "random_forest"   // optional, default = random_forest
    }

    Returns:
    {
      "prediction": "High",
      "confidence": 0.91,
      "probabilities": {"High": 0.91, "Medium": 0.07, "Low": 0.02},
      "explanation": { "feature": contribution, ... },
      "risk_flags": [...],
      "clinical_notes": [...],
      "model_used": "random_forest"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body provided'}), 400

    model_key = data.get('model', 'random_forest')
    if model_key not in MODELS:
        model_key = 'random_forest'

    info = MODELS[model_key]
    m    = info['model']

    try:
        patient = preprocess_input(data)
        X_input = patient.values.reshape(1, -1)
        if info['scaled']:
            X_input = scaler.transform(X_input)

        pred_idx  = int(m.predict(X_input)[0])
        pred_prob = m.predict_proba(X_input)[0]
        pred_label = CLASS_NAMES[pred_idx]
        confidence = float(pred_prob[pred_idx])
        probabilities = {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(pred_prob)}

        # Local explanation
        raw_series = preprocess_input(data)
        explanation = compute_local_explanation(raw_series, model_key)

        # Clinical risk flags (rule-based safety layer)
        flags = []
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
                          'detail': f"RR {data['resp_rate']}/min — respiratory distress. O2 therapy + monitoring."})
        if float(data.get('temperature_c', 37)) > 39.5:
            flags.append({'flag': 'High Fever', 'severity': 'warning',
                          'detail': f"Temp {data['temperature_c']}°C — consider sepsis workup."})
        if data.get('arrival_mode') == 'Ambulance':
            flags.append({'flag': 'Ambulance Arrival', 'severity': 'info',
                          'detail': 'Pre-hospital notification — prepare bay and alert senior clinician.'})

        # Clinical notes (AI reasoning)
        notes = []
        if confidence < 0.70:
            notes.append({'type': 'uncertainty', 'message':
                f'Low confidence ({confidence:.0%}). XAI suggests this case has mixed signals — recommend senior review.'})
        if pred_label == 'High' and confidence > 0.85:
            notes.append({'type': 'alert', 'message':
                'High confidence HIGH triage. Immediate senior clinician attention recommended.'})
        top_feat = sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        notes.append({'type': 'explanation', 'message':
            f"Top drivers: {', '.join([f[0] for f in top_feat])}."})

        return jsonify({
            'prediction':     pred_label,
            'confidence':     round(confidence, 4),
            'probabilities':  probabilities,
            'explanation':    explanation,
            'risk_flags':     flags,
            'clinical_notes': notes,
            'model_used':     model_key,
            'model_name':     info['name'],
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/compare', methods=['POST', 'OPTIONS'])
def predict_compare():
    """Run predictions across ALL three models for the same patient."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON body provided'}), 400

    results = {}
    for key in MODELS:
        data_copy = dict(data)
        data_copy['model'] = key
        with app.test_request_context(
            '/api/predict', method='POST',
            json=data_copy, content_type='application/json'
        ):
            req = request._get_current_object()

        info = MODELS[key]
        m = info['model']
        try:
            patient = preprocess_input(data)
            X_input = patient.values.reshape(1, -1)
            if info['scaled']:
                X_input = scaler.transform(X_input)
            pred_idx  = int(m.predict(X_input)[0])
            pred_prob = m.predict_proba(X_input)[0]
            pred_label = CLASS_NAMES[pred_idx]
            confidence = float(pred_prob[pred_idx])
            probabilities = {CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(pred_prob)}
            explanation = compute_local_explanation(patient, key)

            results[key] = {
                'name': info['name'],
                'prediction': pred_label,
                'confidence': round(confidence, 4),
                'probabilities': probabilities,
                'explanation': explanation,
                'metrics': {
                    'accuracy': METRICS[key]['accuracy'],
                    'auc': METRICS[key]['auc'],
                }
            }
        except Exception as e:
            results[key] = {'error': str(e)}

    return jsonify(results)


@app.route('/api/dataset/sample', methods=['GET', 'OPTIONS'])
def dataset_sample():
    """Return 20 sample rows from the dataset."""
    sample = df.sample(20, random_state=42).to_dict(orient='records')
    return jsonify(sample)


# ──────────────────────────────────────────────────────────
# 5. RUN
# ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# ============================================================
# app.py - Credit Card Fraud Detection Flask Web Application
# Based on Intellipaat ML Project | Logistic Regression + Random Forest
# ============================================================

from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# ---- Paths ----
MODEL_PATH  = os.path.join('models', 'fraud_model.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')

model  = None
scaler = None

def load_artifacts():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH, 'rb') as f:  model  = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
        print(f'[OK] Model ({type(model).__name__}) and scaler loaded.')
    else:
        print('[WARN] Model not found. Run python train.py first.')

# ---- Routes ----
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'running', 'model_loaded': model is not None})

@app.route('/model_info')
def model_info():
    if model is None:
        return jsonify({'status': 'Model not loaded. Run train.py first.'}), 503
    return jsonify({
        'status': 'ready',
        'model_type': type(model).__name__,
        'n_features': 30,
        'feature_names': ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount'],
        'classes': ['Legitimate (0)', 'Fraud (1)']
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Single transaction prediction."""
    if model is None:
        return jsonify({'error': 'Model not loaded. Run train.py first.'}), 503
    try:
        data     = request.get_json(force=True)
        features = data.get('features', [])
        if len(features) != 30:
            return jsonify({'error': f'Expected 30 features, got {len(features)}.'}), 400

        X = np.array(features, dtype=float).reshape(1, -1)
        # Scale only Time (col 0) and Amount (col 29)
        X[:, [0, 29]] = scaler.transform(X[:, [0, 29]])

        pred  = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]

        return jsonify({
            'prediction'            : pred,
            'label'                 : 'FRAUD' if pred == 1 else 'LEGITIMATE',
            'fraud_probability'     : round(float(proba[1]) * 100, 2),
            'legitimate_probability': round(float(proba[0]) * 100, 2),
            'risk_level'            : 'HIGH' if proba[1] > 0.7 else ('MEDIUM' if proba[1] > 0.3 else 'LOW')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch transaction predictions."""
    if model is None:
        return jsonify({'error': 'Model not loaded. Run train.py first.'}), 503
    try:
        records = request.get_json(force=True).get('records', [])
        results, fraud_count = [], 0
        for rec in records:
            feats = rec.get('features', [])
            if len(feats) != 30:
                results.append({'error': 'Expected 30 features'})
                continue
            X = np.array(feats, dtype=float).reshape(1, -1)
            X[:, [0, 29]] = scaler.transform(X[:, [0, 29]])
            pred  = int(model.predict(X)[0])
            proba = model.predict_proba(X)[0]
            if pred == 1: fraud_count += 1
            results.append({
                'prediction'       : pred,
                'label'            : 'FRAUD' if pred == 1 else 'LEGITIMATE',
                'fraud_probability': round(float(proba[1]) * 100, 2)
            })
        return jsonify({'total': len(results), 'fraud_detected': fraud_count, 'results': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_artifacts()
    app.run(debug=True, host='0.0.0.0', port=5000)

# ============================================================
# train.py - ML Training Pipeline for Credit Card Fraud Detection
# Follows Intellipaat YouTube tutorial methodology
# Models: Logistic Regression + Random Forest (Claude-1x class)
# Dataset: Kaggle Credit Card Fraud Detection (creditcard.csv)
# ============================================================

import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

DATA_PATH  = os.path.join('data', 'creditcard.csv')
MODEL_DIR  = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs('data', exist_ok=True)

# ============================================================
# STEP 1: LOAD & EXPLORE DATA
# ============================================================
def load_and_explore(path):
    print('=' * 60)
    print('STEP 1: Loading & Exploring Data')
    print('=' * 60)
    df = pd.read_csv(path)
    print(f'Shape          : {df.shape}')
    print(f'Columns        : {list(df.columns)}')
    print(f'Missing values : {df.isnull().sum().sum()}')
    print(f'Duplicate rows : {df.duplicated().sum()}')
    print(f'\nClass Distribution:')
    print(df['Class'].value_counts())
    legit = df[df['Class'] == 0]
    fraud = df[df['Class'] == 1]
    print(f'\nLegitimate transactions : {len(legit):,} ({len(legit)/len(df)*100:.2f}%)')
    print(f'Fraudulent transactions : {len(fraud):,} ({len(fraud)/len(df)*100:.4f}%)')
    print(f'\nAmount Statistics (Legitimate):')
    print(legit['Amount'].describe())
    print(f'\nAmount Statistics (Fraud):')
    print(fraud['Amount'].describe())
    return df

# ============================================================
# STEP 2: PREPROCESS DATA
# ============================================================
def preprocess(df):
    print('\n' + '=' * 60)
    print('STEP 2: Preprocessing Data')
    print('=' * 60)
    # Remove duplicates
    df = df.drop_duplicates()
    print(f'After dedup shape: {df.shape}')
    # Features: Time, V1-V28, Amount
    feature_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    X = df[feature_cols].values.copy()
    y = df['Class'].values
    # Scale Time (col 0) and Amount (col 29) - V1-V28 already PCA-scaled
    scaler = StandardScaler()
    X[:, [0, 29]] = scaler.fit_transform(X[:, [0, 29]])
    print(f'Features shape : {X.shape}')
    print(f'Labels   shape : {y.shape}')
    print(f'Scaling applied to: Time, Amount')
    return X, y, scaler

# ============================================================
# STEP 3: HANDLE CLASS IMBALANCE WITH SMOTE
# ============================================================
def handle_imbalance(X_train, y_train):
    print('\n' + '=' * 60)
    print('STEP 3: Handling Class Imbalance with SMOTE')
    print('=' * 60)
    print(f'Before SMOTE: {np.bincount(y_train)}')
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f'After  SMOTE: {np.bincount(y_res)}')
    print(f'New shape   : {X_res.shape}')
    return X_res, y_res

# ============================================================
# STEP 4: TRAIN MODELS
# ============================================================
def train_models(X_train, y_train):
    print('\n' + '=' * 60)
    print('STEP 4: Training Models (Logistic Regression + Random Forest)')
    print('=' * 60)
    # Logistic Regression (Claude-1x baseline)
    print('\n[1/2] Training Logistic Regression...')
    lr = LogisticRegression(
        C=0.01, penalty='l2', solver='lbfgs',
        max_iter=1000, class_weight='balanced', random_state=42
    )
    lr.fit(X_train, y_train)
    print('  Done.')
    # Random Forest (Claude-1x ensemble)
    print('[2/2] Training Random Forest...')
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print('  Done.')
    return lr, rf

# ============================================================
# STEP 5: EVALUATE MODELS
# ============================================================
def evaluate(model, X_test, y_test, name):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'Accuracy' : round(accuracy_score(y_test, y_pred)  * 100, 2),
        'Precision': round(precision_score(y_test, y_pred) * 100, 2),
        'Recall'   : round(recall_score(y_test, y_pred)    * 100, 2),
        'F1-Score' : round(f1_score(y_test, y_pred)        * 100, 2),
        'ROC-AUC'  : round(roc_auc_score(y_test, y_proba)  * 100, 2),
    }
    print(f'\n--- {name} ---')
    for k, v in metrics.items():
        print(f'  {k:<12}: {v:.2f}%')
    print(f'\n  Classification Report:')
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    print(f'  Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    return metrics

# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    print('CREDIT CARD FRAUD DETECTION - TRAINING PIPELINE')
    print('Reference: Intellipaat ML Project Tutorial')
    print('=' * 60)

    if not os.path.exists(DATA_PATH):
        print(f'ERROR: Dataset not found at {DATA_PATH}')
        print('Options:')
        print('  1. Download from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud')
        print('     Place as data/creditcard.csv')
        print('  2. Generate synthetic data: python generate_sample_data.py')
        return

    # Pipeline
    df               = load_and_explore(DATA_PATH)
    X, y, scaler     = preprocess(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f'\nTrain: {X_tr.shape} | Test: {X_te.shape}')
    X_tr_sm, y_tr_sm = handle_imbalance(X_tr, y_tr)
    lr_model, rf_model = train_models(X_tr_sm, y_tr_sm)

    print('\n' + '=' * 60)
    print('STEP 5: Evaluation Results')
    print('=' * 60)
    lr_m = evaluate(lr_model, X_te, y_te, 'Logistic Regression')
    rf_m = evaluate(rf_model, X_te, y_te, 'Random Forest')

    # Select best by F1 (most important for imbalanced classification)
    best_model = rf_model if rf_m['F1-Score'] >= lr_m['F1-Score'] else lr_model
    best_name  = 'Random Forest' if rf_m['F1-Score'] >= lr_m['F1-Score'] else 'Logistic Regression'
    print(f'\n[BEST MODEL] {best_name} (F1={max(rf_m["F1-Score"], lr_m["F1-Score"]):.2f}%)')

    # Save
    with open(os.path.join(MODEL_DIR, 'fraud_model.pkl'), 'wb')  as f: pickle.dump(best_model, f)
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb')       as f: pickle.dump(scaler, f)
    print(f'\nModel  saved -> models/fraud_model.pkl')
    print(f'Scaler saved -> models/scaler.pkl')
    print('\nAll done! Run: python app.py to start the web server.')

if __name__ == '__main__':
    main()

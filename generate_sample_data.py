# ============================================================
# generate_sample_data.py
# Generate synthetic credit card fraud dataset for testing
# Use this if you don't have the Kaggle creditcard.csv dataset
# ============================================================

import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_dataset(n_legit=10000, n_fraud=492):
    """
    Generates a synthetic dataset mimicking the real Kaggle creditcard.csv.
    - 30 features: Time, V1-V28 (PCA-like), Amount
    - Class: 0 = Legitimate, 1 = Fraud
    - Real dataset ratio: ~0.17% fraud
    """
    n_total = n_legit + n_fraud

    # V1-V28: PCA-transformed features (normally distributed)
    V = np.random.randn(n_total, 28)

    # Inject fraud signatures (as seen in real dataset analysis)
    # Fraud records show extreme values in V1, V3, V10, V11, V12, V14, V17
    V[n_legit:, 0]  -= 4.8   # V1  very negative for fraud
    V[n_legit:, 2]  -= 3.5   # V3  negative
    V[n_legit:, 9]  -= 2.8   # V10 negative
    V[n_legit:, 10] += 4.0   # V11 positive
    V[n_legit:, 11] -= 4.5   # V12 very negative
    V[n_legit:, 13] -= 5.0   # V14 very negative (strongest predictor)
    V[n_legit:, 16] -= 3.2   # V17 negative
    V[n_legit:, 3]  += 2.0   # V4  positive for fraud

    # Time: seconds elapsed (2-day period, 0-172800)
    time_legit = np.random.uniform(0, 172800, n_legit)
    time_fraud = np.random.uniform(0, 172800, n_fraud)

    # Amount: legitimate ~Exp(88), fraud ~Exp(122) but capped
    amount_legit = np.clip(np.random.exponential(88,  n_legit), 0.01, 25691.16)
    amount_fraud = np.clip(np.random.exponential(122, n_fraud), 0.01, 2125.87)

    # Labels
    class_legit = np.zeros(n_legit, dtype=int)
    class_fraud = np.ones(n_fraud,  dtype=int)

    # Combine & build DataFrame
    Time   = np.concatenate([time_legit,   time_fraud])
    Amount = np.concatenate([amount_legit, amount_fraud])
    Class  = np.concatenate([class_legit,  class_fraud])

    df = pd.DataFrame(V, columns=[f'V{i}' for i in range(1, 29)])
    df.insert(0, 'Time', Time)
    df['Amount'] = Amount
    df['Class']  = Class

    # Shuffle
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def main():
    os.makedirs('data', exist_ok=True)
    print('Generating synthetic credit card transaction dataset...')
    df   = generate_dataset(n_legit=10000, n_fraud=492)
    path = os.path.join('data', 'creditcard.csv')
    df.to_csv(path, index=False)

    print(f'Saved to     : {path}')
    print(f'Total rows   : {len(df):,}')
    print(f'Features     : {df.shape[1] - 1} (Time, V1-V28, Amount)')
    print(f'\nClass Distribution:')
    print(df['Class'].value_counts())
    print(f'Fraud rate   : {df["Class"].mean() * 100:.4f}%')
    print('\nDataset ready. Now run: python train.py')

if __name__ == '__main__':
    main()

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_and_preprocess():
    df = pd.read_csv("C:/Users/swaya/OneDrive/Desktop/gnn/fraud_detection_project/data/creditcard.csv")

    # ✅ Add Transaction ID
    df['Transaction_ID'] = range(len(df))

    # Features
    X = df.drop(['Class', 'Transaction_ID'], axis=1)
    y = df['Class']

    # Scale all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Add back labels + IDs
    X_scaled['Class'] = y.values
    X_scaled['Transaction_ID'] = df['Transaction_ID']

    return X_scaled


def split_data(df):
    X = df.drop(['Class', 'Transaction_ID'], axis=1)
    y = df['Class']
    ids = df['Transaction_ID']

    return train_test_split(X, y, ids, test_size=0.2, random_state=42)
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report

def run_isolation_forest(X_train, X_test, y_test):
    # Initialize model
    model = IsolationForest(
        n_estimators=100,
        contamination=0.0017,  # approx fraud ratio
        random_state=42
    )

    # Train model
    model.fit(X_train)

    # Predict
    y_pred = model.predict(X_test)

    # Convert predictions
    # IF: 1 = normal, -1 = anomaly
    # We convert: anomaly → 1 (fraud), normal → 0
    y_pred = [1 if x == -1 else 0 for x in y_pred]

    return y_pred
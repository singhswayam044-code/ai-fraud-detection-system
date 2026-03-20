from sklearn.linear_model import LogisticRegression

def run_logistic_regression(X_train, X_test, y_train):
    model = LogisticRegression(
    max_iter=2000,
    class_weight='balanced',
    solver='liblinear',  
    random_state=42
)

    model.fit(X_train, y_train)

    # Predict probability
    y_prob = model.predict_proba(X_test)[:, 1]

    # Convert to labels (threshold = 0.5)
    y_pred = (y_prob > 0.5).astype(int)

    return y_pred, y_prob
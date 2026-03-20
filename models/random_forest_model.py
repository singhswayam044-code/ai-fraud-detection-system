from sklearn.ensemble import RandomForestClassifier

def run_random_forest(X_train, X_test, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',  # VERY IMPORTANT
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return y_pred
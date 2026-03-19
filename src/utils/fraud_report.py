import pandas as pd


def generate_fraud_report(X_test, ids_test, lr_prob, iso_pred, gnn_probs):
    """
    Generates a fraud report with:
    - Transaction ID
    - Risk Level (Low / Medium / High)
    - Fraud Score
    - Explanation (Reason)

    Parameters:
    X_test     : Feature data
    ids_test   : Transaction IDs
    lr_prob    : Logistic Regression probabilities
    iso_pred   : Isolation Forest predictions
    gnn_probs  : GNN fraud probabilities

    Returns:
    Pandas DataFrame
    """

    report = []

    for i in range(len(X_test)):

        # =========================
        # SCORES
        # =========================
        lr_score = lr_prob[i]
        gnn_score = gnn_probs[i]

        # Combine scores (simple average)
        final_score = (lr_score + gnn_score) / 2

        # =========================
        # RISK CLASSIFICATION
        # =========================
        if final_score > 0.7:
            risk = "High Risk"
        elif final_score > 0.4:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        # =========================
        # EXPLANATION
        # =========================
        reasons = []

        if lr_score > 0.7:
            reasons.append("High fraud probability (ML model)")

        if gnn_score > 0.7:
            reasons.append("Connected to suspicious transaction network")

        if iso_pred[i] == 1:
            reasons.append("Anomalous behavior detected")

        if len(reasons) == 0:
            reasons.append("No strong fraud indicators")

        # =========================
        # ADD TO REPORT
        # =========================
        report.append({
            "Transaction_ID": int(ids_test.iloc[i]),
            "Risk_Level": risk,
            "Fraud_Score": round(float(final_score), 3),
            "LR_Probability": round(float(lr_score), 3),
            "GNN_Score": round(float(gnn_score), 3),
            "Anomaly_Flag": int(iso_pred[i]),
            "Reason": ", ".join(reasons)
        })

    # Convert to DataFrame
    df_report = pd.DataFrame(report)

    # =========================
    # SORT BY RISK (IMPORTANT)
    # =========================
    risk_order = {"High Risk": 3, "Medium Risk": 2, "Low Risk": 1}
    df_report["Risk_Order"] = df_report["Risk_Level"].map(risk_order)

    df_report = df_report.sort_values(by=["Risk_Order", "Fraud_Score"], ascending=False)

    df_report = df_report.drop(columns=["Risk_Order"])

    return df_report
# ==============================
# IMPORTS
# ==============================
import os
import torch

from src.preprocessing import load_and_preprocess, split_data

from src.models.isolation_forest_model import run_isolation_forest
from src.models.random_forest_model import run_random_forest
from src.models.logistic_regression_model import run_logistic_regression
from src.models.gnn_model import create_graph, GNNModel, train_gnn

from src.utils.fraud_report import generate_fraud_report


# ==============================
# MAIN PIPELINE
# ==============================
def run_pipeline():

    print("\n🚀 Running Fraud Detection Pipeline...\n")

    # ==============================
    # LOAD DATA
    # ==============================
    df = load_and_preprocess()

    X_train, X_test, y_train, y_test, ids_train, ids_test = split_data(df)

    # ==============================
    # RUN MODELS
    # ==============================
    print("🔍 Running Isolation Forest...")
    iso_pred = run_isolation_forest(X_train, X_test, y_test)

    print("🌲 Running Random Forest...")
    rf_pred = run_random_forest(X_train, X_test, y_train)

    print("📈 Running Logistic Regression...")
    lr_pred, lr_prob = run_logistic_regression(X_train, X_test, y_train)

    # ==============================
    # GNN (GRAPH MODEL)
    # ==============================
    print("🧠 Running GNN...")

    data = create_graph(X_test, y_test, ids_test)

    gnn_model = GNNModel(input_dim=data.num_features)
    train_gnn(gnn_model, data)

    with torch.no_grad():
        gnn_out = gnn_model(data)
        gnn_probs = torch.exp(gnn_out)[:, 1].numpy()

    # ==============================
    # FINAL HYBRID PREDICTION
    # ==============================
    print("⚖️ Combining model predictions...")

    final_pred = []

    for i in range(len(X_test)):
        votes = iso_pred[i] + rf_pred[i] + lr_pred[i]

        if votes >= 2:
            final_pred.append(1)
        else:
            final_pred.append(0)

    # ==============================
    # GENERATE FRAUD REPORT (🔥 KEY PART)
    # ==============================
    print("📝 Generating fraud report...")

    fraud_df = generate_fraud_report(
        X_test,
        ids_test,
        lr_prob,
        iso_pred,
        gnn_probs
    )

    # ==============================
    # SAVE CSV (WITH EXACT PATH)
    # ==============================
    file_path = os.path.join(os.getcwd(), "fraud_report.csv")
    fraud_df.to_csv(file_path, index=False)

    print("\n✅ Fraud report saved at:", file_path)

    # ==============================
    # RETURN OUTPUT
    # ==============================
    return y_test, final_pred
# ==============================
# IMPORTS
# ==============================
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score
)

from src.preprocessing import load_and_preprocess, split_data
from src.models.isolation_forest_model import run_isolation_forest
from src.models.random_forest_model import run_random_forest
from src.models.logistic_regression_model import run_logistic_regression
from src.models.gnn_model import create_graph, GNNModel, train_gnn
from src.models.lstm_model import create_sequences, LSTMModel, train_lstm
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

    # ==============================
    # CLASS IMBALANCE ANALYSIS
    # ==============================
    print("\n📊 Class Distribution:\n")
    print(df['Class'].value_counts(normalize=True))

    df['Class'].value_counts().plot(kind='bar')
    plt.title("Class Distribution (Fraud vs Normal)")
    plt.xticks([0, 1], ['Normal', 'Fraud'], rotation=0)
    plt.show()

    # ==============================
    # SPLIT DATA
    # ==============================
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
    # LSTM
    # ==============================
    print("⏳ Running LSTM...")

    X_train_np = X_train.values
    y_train_np = y_train.values
    X_test_np = X_test.values

    X_seq, y_seq = create_sequences(X_train_np, y_train_np, seq_length=10)

    input_size = X_seq.shape[2]
    lstm_model = LSTMModel(input_size)

    train_lstm(lstm_model, X_seq, y_seq, epochs=3)

    X_test_seq, _ = create_sequences(X_test_np, y_test.values, seq_length=10)

    with torch.no_grad():
        lstm_out = lstm_model(torch.tensor(X_test_seq, dtype=torch.float32)).squeeze().numpy()

    lstm_pred = (lstm_out > 0.5).astype(int)

    # ==============================
    # GNN (NO DATA LEAKAGE)
    # ==============================
    print("🧠 Running GNN...")

    # Train graph
    train_data = create_graph(X_train, y_train, ids_train)

    gnn_model = GNNModel(input_dim=train_data.num_features)
    train_gnn(gnn_model, train_data)

    # Test graph
    test_data = create_graph(X_test, y_test, ids_test)

    with torch.no_grad():
        gnn_out = gnn_model(test_data)
        gnn_probs = torch.exp(gnn_out)[:, 1].numpy()

    # ==============================
    # ALIGN LENGTHS
    # ==============================
    min_len = min(len(iso_pred), len(rf_pred), len(lr_pred), len(lstm_pred))

    iso_pred = iso_pred[:min_len]
    rf_pred = rf_pred[:min_len]
    lr_pred = lr_pred[:min_len]
    lr_prob = lr_prob[:min_len]
    lstm_pred = lstm_pred[:min_len]
    gnn_probs = gnn_probs[:min_len]

    y_test = y_test[:min_len]
    X_test = X_test.iloc[:min_len]
    ids_test = ids_test[:min_len]

    # ==============================
    # FINAL HYBRID PREDICTION
    # ==============================
    print("⚖️ Combining model predictions...")

    final_pred = []

    for i in range(min_len):
        votes = iso_pred[i] + rf_pred[i] + lr_pred[i] + lstm_pred[i]
        final_pred.append(1 if votes >= 3 else 0)

    # ==============================
    # FRAUD REPORT
    # ==============================
    fraud_df = generate_fraud_report(
        X_test,
        ids_test,
        lr_prob,
        iso_pred,
        gnn_probs
    )

    file_path = os.path.join(os.getcwd(), "fraud_report.csv")
    fraud_df.to_csv(file_path, index=False)

    print("\n✅ Fraud report saved at:", file_path)

    # ==============================
    # MODEL EVALUATION
    # ==============================
    print("\n📊 MODEL EVALUATION\n")

    print(classification_report(y_test, final_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, final_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC-AUC
    roc_score = roc_auc_score(y_test, lr_prob)
    print(f"ROC-AUC Score: {roc_score:.4f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, lr_prob)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, lr_prob)

    plt.figure()
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

    # F1 Score
    f1 = f1_score(y_test, final_pred)
    print(f"F1 Score: {f1:.4f}")

    return y_test, final_pred
Fraud Detection System using Machine Learning, Graph Neural Networks, and Temporal Analysis
Overview

This project implements a hybrid fraud detection system that combines traditional machine learning models, graph-based learning, and temporal sequence modeling. The goal is to detect fraudulent transactions with high accuracy while maintaining interpretability and real-world applicability.

The system is designed to handle multiple fraud patterns:

Individual anomalous transactions

Behavioral fraud over time

Network-based coordinated fraud

Key Features

Hybrid model architecture combining multiple approaches

Risk-based classification (Low, Medium, High)

Graph-based fraud detection using GNN

Temporal pattern detection using LSTM

Model interpretability through probability scores and anomaly detection

End-to-end pipeline from preprocessing to reporting

Interactive dashboard using Streamlit

Automated fraud report generation (CSV)

Model evaluation with standard classification metrics

Architecture

The system follows a multi-stage pipeline:

Data preprocessing and normalization

Train-test split with transaction identifiers

Model execution:

Isolation Forest for anomaly detection

Random Forest for classification

Logistic Regression for probability scoring

LSTM for temporal pattern learning

Graph Neural Network for relational learning

Hybrid decision system using majority voting

Fraud scoring and risk classification

Evaluation and visualization

Report generation and dashboard display

Models Used
Isolation Forest

Detects anomalous transactions based on deviation from normal patterns.
Configured with contamination aligned to dataset fraud ratio (~0.17%).

Random Forest

Supervised classifier that captures nonlinear relationships and feature interactions.

Logistic Regression

Provides interpretable probability scores used in final fraud scoring and evaluation.

LSTM

Captures temporal transaction patterns and detects sequence-based fraud behavior.

Graph Neural Network (GNN)

Builds a transaction graph using nearest neighbors and detects fraud based on relational structure.

Hybrid Decision Logic

Final predictions are generated using majority voting across:

Isolation Forest

Random Forest

Logistic Regression

LSTM

This ensures robustness across different fraud detection perspectives.

Risk Classification

Transactions are categorized into:

Low Risk: Normal behavior

Medium Risk: Suspicious activity

High Risk: Likely fraud

Risk is computed using a combination of model outputs including probability scores, anomaly signals, and graph-based insights.

Handling Class Imbalance

The dataset is highly imbalanced (~0.17% fraud cases).

Approach used:

Class weighting in Random Forest and Logistic Regression

Isolation Forest contamination set to match fraud ratio

Evaluation focused on recall, F1 score, and ROC-AUC instead of accuracy

Model Evaluation

Evaluation is performed on a held-out test set using:

Precision

Recall

F1 Score

ROC-AUC Score

Confusion Matrix

ROC Curve

Precision-Recall Curve

Sample Results

Precision (Fraud): 0.71

Recall (Fraud): 0.76

F1 Score: 0.73

ROC-AUC: 0.9795

These results indicate strong discrimination capability and effective fraud detection performance.

Output

The system generates a file:

fraud_report.csv

This includes:

Transaction ID

Risk Level

Fraud Score

Logistic Regression Probability

GNN Score

Anomaly Flag

Explanation for fraud classification

Dashboard

The project includes a Streamlit-based dashboard with:

Fraud detection execution

Risk distribution summary

Full-width transaction table with filtering

Downloadable fraud report

Graph-based fraud visualization

Performance metrics display

Project Structure
fraud_detection_project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── main_pipeline.py
│   ├── preprocessing.py
│   ├── models/
│   └── utils/
│
├── notebooks/
└── data/
Installation

Clone the repository:

git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system

Install dependencies:

pip install -r requirements.txt
Running the Project

Run the pipeline:

python notebooks/test_pipeline.py

Run the dashboard:

streamlit run app.py
Testing

Basic assertion tests are included to validate:

Prediction length consistency

Output format correctness

Non-empty results

Limitations

LSTM integration is basic and can be improved for real sequential modeling

Graph construction uses nearest neighbors rather than real-world entity relationships

No real-time deployment or streaming support

Interpretability can be enhanced further using SHAP or similar methods

Future Improvements

Threshold tuning to improve recall

SMOTE or advanced resampling techniques

Real-time fraud detection pipeline

SHAP-based explainability

Deployment on cloud infrastructure

Interactive graph visualization in dashboard

Conclusion

This project demonstrates a complete fraud detection system that integrates multiple modeling approaches into a single pipeline. It balances performance, interpretability, and practical usability, making it suitable for real-world applications and data science portfolios.

# Fraud Detection System using Machine Learning, Graph Neural Networks, and Temporal Analysis

## Overview

This project implements a hybrid fraud detection system that combines traditional machine learning models, graph-based learning, and temporal sequence modeling. The objective is to detect fraudulent transactions accurately while maintaining interpretability and real-world applicability.

The system addresses multiple fraud scenarios:
- Individual anomalous transactions  
- Behavioral fraud over time  
- Network-based coordinated fraud  

---

## Key Features

- Hybrid architecture combining multiple models  
- Risk-based classification (Low, Medium, High)  
- Graph-based fraud detection using GNN  
- Temporal pattern detection using LSTM  
- Interpretable outputs using probability and anomaly scores  
- End-to-end pipeline from preprocessing to reporting  
- Interactive dashboard using Streamlit  
- Automated fraud report generation (CSV)  
- Model evaluation using standard metrics  

---

## Architecture

The system follows a multi-stage pipeline:

1. Data preprocessing and normalization  
2. Train-test split with transaction identifiers  
3. Model execution:
   - Isolation Forest for anomaly detection  
   - Random Forest for classification  
   - Logistic Regression for probability scoring  
   - LSTM for temporal modeling  
   - Graph Neural Network for relational learning  
4. Hybrid decision system using majority voting  
5. Fraud scoring and risk classification  
6. Evaluation and visualization  
7. Report generation and dashboard display  

---

## Models Used

### Isolation Forest
Detects anomalous transactions by identifying deviations from normal patterns.  
Configured with contamination based on dataset fraud ratio (~0.17%).

### Random Forest
Supervised model that captures nonlinear relationships and feature interactions.

### Logistic Regression
Provides interpretable probability scores used in fraud scoring and evaluation.

### LSTM
Captures temporal transaction patterns and detects sequence-based fraud behavior.

### Graph Neural Network (GNN)
Constructs a graph using nearest neighbors and detects fraud through relational patterns.

---

## Hybrid Decision Logic

Final predictions are generated using majority voting across:
- Isolation Forest  
- Random Forest  
- Logistic Regression  
- LSTM  

This improves robustness and reduces dependence on a single model.

---

## Risk Classification

Transactions are categorized into:

- **Low Risk** — Normal behavior  
- **Medium Risk** — Suspicious activity  
- **High Risk** — Likely fraud  

Risk levels are determined using a combination of model outputs and fraud scores.

---

## Handling Class Imbalance

The dataset is highly imbalanced (~0.17% fraud cases).

Approach used:
- Class weighting in Random Forest and Logistic Regression  
- Isolation Forest contamination set to match fraud ratio  
- Evaluation focused on recall, F1 score, and ROC-AUC rather than accuracy  

---

## Model Evaluation

Evaluation is performed on a held-out test set using:

- Precision  
- Recall  
- F1 Score  
- ROC-AUC Score  
- Confusion Matrix  
- ROC Curve  
- Precision-Recall Curve  

### Sample Results

- Precision (Fraud): 0.71  
- Recall (Fraud): 0.76  
- F1 Score: 0.73  
- ROC-AUC: 0.9795  

These results indicate strong model performance and effective fraud detection capability.

---

## Output

The system generates a file:

`fraud_report.csv`

This file contains:
- Transaction ID  
- Risk Level  
- Fraud Score  
- Logistic Regression Probability  
- GNN Score  
- Anomaly Flag  
- Explanation for classification  

---

## Dashboard

The project includes a Streamlit-based dashboard with:

- Fraud detection execution  
- Risk distribution summary (with pie chart)  
- Transaction table with filtering  
- Downloadable fraud report  
- Graph-based fraud visualization  
- Performance metrics display  

---

## Project Structure
# Fraud Detection System using Machine Learning, Graph Neural Networks, and Temporal Analysis

## Overview

This project implements a hybrid fraud detection system that combines traditional machine learning models, graph-based learning, and temporal sequence modeling. The objective is to detect fraudulent transactions accurately while maintaining interpretability and real-world applicability.

The system addresses multiple fraud scenarios:
- Individual anomalous transactions  
- Behavioral fraud over time  
- Network-based coordinated fraud  

---

## Key Features

- Hybrid architecture combining multiple models  
- Risk-based classification (Low, Medium, High)  
- Graph-based fraud detection using GNN  
- Temporal pattern detection using LSTM  
- Interpretable outputs using probability and anomaly scores  
- End-to-end pipeline from preprocessing to reporting  
- Interactive dashboard using Streamlit  
- Automated fraud report generation (CSV)  
- Model evaluation using standard metrics  

---

## Architecture

The system follows a multi-stage pipeline:

1. Data preprocessing and normalization  
2. Train-test split with transaction identifiers  
3. Model execution:
   - Isolation Forest for anomaly detection  
   - Random Forest for classification  
   - Logistic Regression for probability scoring  
   - LSTM for temporal modeling  
   - Graph Neural Network for relational learning  
4. Hybrid decision system using majority voting  
5. Fraud scoring and risk classification  
6. Evaluation and visualization  
7. Report generation and dashboard display  

---

## Models Used

### Isolation Forest
Detects anomalous transactions by identifying deviations from normal patterns.  
Configured with contamination based on dataset fraud ratio (~0.17%).

### Random Forest
Supervised model that captures nonlinear relationships and feature interactions.

### Logistic Regression
Provides interpretable probability scores used in fraud scoring and evaluation.

### LSTM
Captures temporal transaction patterns and detects sequence-based fraud behavior.

### Graph Neural Network (GNN)
Constructs a graph using nearest neighbors and detects fraud through relational patterns.

---

## Hybrid Decision Logic

Final predictions are generated using majority voting across:
- Isolation Forest  
- Random Forest  
- Logistic Regression  
- LSTM  

This improves robustness and reduces dependence on a single model.

---

## Risk Classification

Transactions are categorized into:

- **Low Risk** — Normal behavior  
- **Medium Risk** — Suspicious activity  
- **High Risk** — Likely fraud  

Risk levels are determined using a combination of model outputs and fraud scores.

---

## Handling Class Imbalance

The dataset is highly imbalanced (~0.17% fraud cases).

Approach used:
- Class weighting in Random Forest and Logistic Regression  
- Isolation Forest contamination set to match fraud ratio  
- Evaluation focused on recall, F1 score, and ROC-AUC rather than accuracy  

---

## Model Evaluation

Evaluation is performed on a held-out test set using:

- Precision  
- Recall  
- F1 Score  
- ROC-AUC Score  
- Confusion Matrix  
- ROC Curve  
- Precision-Recall Curve  

### Sample Results

- Precision (Fraud): 0.71  
- Recall (Fraud): 0.76  
- F1 Score: 0.73  
- ROC-AUC: 0.9795  

These results indicate strong model performance and effective fraud detection capability.

---

## Output

The system generates a file:

`fraud_report.csv`

This file contains:
- Transaction ID  
- Risk Level  
- Fraud Score  
- Logistic Regression Probability  
- GNN Score  
- Anomaly Flag  
- Explanation for classification  

---

## Dashboard

The project includes a Streamlit-based dashboard with:

- Fraud detection execution  
- Risk distribution summary (with pie chart)  
- Transaction table with filtering  
- Downloadable fraud report  
- Graph-based fraud visualization  
- Performance metrics display  

---

## Project Structure
fraud_detection_project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│ ├── main_pipeline.py
│ ├── preprocessing.py
│ ├── models/
│ └── utils/
│
├── notebooks/
└── data/

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system
pip install -r requirements.txt
python notebooks/test_pipeline.py
streamlit run app.py
Testing

Basic assertion tests are included to validate:

Prediction length consistency

Output format correctness

Non-empty results

Limitations
1.LSTM integration is basic and can be improved
2.Graph construction is based on nearest neighbors rather than real-world entity relationships
3.No real-time streaming or deployment
4.Explainability can be extended using advanced techniques
5.Future Improvements
6.Threshold tuning to improve recall
7.Use of SMOTE or advanced imbalance handling
8.Real-time fraud detection pipeline
9.SHAP-based explainability
10.Cloud deployment
11.Interactive graph visualization

Conclusion

This project demonstrates a complete fraud detection system integrating machine learning, graph analysis, and temporal modeling into a unified pipeline. It balances performance, interpretability, and usability, making it suitable for real-world applications and data science portfolios.

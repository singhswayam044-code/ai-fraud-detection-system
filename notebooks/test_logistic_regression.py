import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_and_preprocess, split_data
from src.models.logistic_regression_model import run_logistic_regression
from sklearn.metrics import classification_report

# Load data
df = load_and_preprocess()

# Split
X_train, X_test, y_train, y_test = split_data(df)

# Run model
y_pred, y_prob = run_logistic_regression(X_train, X_test, y_train)

# Evaluate
print(classification_report(y_test, y_pred))
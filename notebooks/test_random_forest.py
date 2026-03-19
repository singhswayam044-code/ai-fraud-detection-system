import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_and_preprocess, split_data
from src.models.random_forest_model import run_random_forest
from sklearn.metrics import classification_report

# Load data
df = load_and_preprocess()

# Split data
X_train, X_test, y_train, y_test = split_data(df)

# Run model
y_pred = run_random_forest(X_train, X_test, y_train)

# Evaluate
print(classification_report(y_test, y_pred))
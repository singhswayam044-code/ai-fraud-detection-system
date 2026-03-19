import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main_pipeline import run_pipeline
from sklearn.metrics import classification_report

y_test, y_pred = run_pipeline()

print(classification_report(y_test, y_pred))
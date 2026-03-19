import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_and_preprocess
from src.models.gnn_model import create_graph, GNNModel, train_gnn, get_fraud_transactions

# Load data
df = load_and_preprocess()

# Reduce size for stability
df = df.sample(5000, random_state=42)

X = df.drop(['Class', 'Transaction_ID'], axis=1)
y = df['Class']
ids = df['Transaction_ID']

# Create graph
data = create_graph(X, y, ids)

print("Graph created:", data)

# Model
model = GNNModel(input_dim=data.num_features)

# Train
train_gnn(model, data)

# 🔥 Identify fraud
fraud_idx, fraud_ids = get_fraud_transactions(model, data)

print("\n🚨 Detected Fraud Transaction IDs:")
print(fraud_ids[:20])
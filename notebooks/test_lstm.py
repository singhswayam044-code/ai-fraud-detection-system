import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import load_and_preprocess
from src.models.gnn_model import create_graph, GNNModel, train_gnn

# Load data
df = load_and_preprocess()

X = df.drop('Class', axis=1)
y = df['Class']

# Create graph
ids = range(len(X))   # temporary IDs for LSTM
data = create_graph(X, y, ids)

print("Graph:", data)

# Model
model = GNNModel(input_dim=data.num_features)

# Train
train_gnn(model, data, epochs=5)
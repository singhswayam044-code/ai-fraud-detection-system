import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import networkx as nx
import matplotlib.pyplot as plt

from src.preprocessing import load_and_preprocess
from src.models.gnn_model import create_graph

# Load data
df = load_and_preprocess()

# Reduce size for visualization
df = df.sample(1000, random_state=42)

# Extract features, labels, IDs
X = df.drop(['Class', 'Transaction_ID'], axis=1)
y = df['Class']
ids = df['Transaction_ID']  # ✅ IMPORTANT

# Create graph
data = create_graph(X, y, ids)

# Convert to NetworkX
G = nx.Graph()
edge_index = data.edge_index.numpy()

# Add edges
for i in range(edge_index.shape[1]):
    u = int(edge_index[0][i])
    v = int(edge_index[1][i])
    G.add_edge(u, v)

# Color nodes
colors = ['red' if y.iloc[i] == 1 else 'lightblue' for i in range(len(y))]

# Draw graph
plt.figure(figsize=(10, 8))
nx.draw(
    G,
    node_color=colors,
    node_size=20,
    edge_color='gray',
    width=0.3,
    alpha=0.7
)

plt.title("Fraud Network Visualization")
plt.show()
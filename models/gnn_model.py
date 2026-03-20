import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors


# ==============================
# CREATE GRAPH WITH IDS
# ==============================
def create_graph(X, y, ids, k=5):
    X = X.values if hasattr(X, "values") else X

    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(X)
    _, indices = nbrs.kneighbors(X)

    edge_index = []

    for i, neighbors in enumerate(indices):
        for j in neighbors:
            edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.values if hasattr(y, "values") else y, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)

    # ✅ Store transaction IDs
    data.ids = ids.values if hasattr(ids, "values") else ids

    return data


# ==============================
# GNN MODEL
# ==============================
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(GNNModel, self).__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


# ==============================
# TRAIN
# ==============================
def train_gnn(model, data, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    for epoch in range(epochs):
        model.train()

        out = model(data)
        loss = criterion(out, data.y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


# ==============================
# 🔥 FRAUD IDENTIFICATION
# ==============================
def get_fraud_transactions(model, data, threshold=0.5):
    model.eval()

    with torch.no_grad():
        out = model(data)
        probs = torch.exp(out)[:, 1]  # fraud probability

    fraud_indices = (probs > threshold).nonzero(as_tuple=True)[0]

    fraud_ids = data.ids[fraud_indices.numpy()]

    return fraud_indices, fraud_ids
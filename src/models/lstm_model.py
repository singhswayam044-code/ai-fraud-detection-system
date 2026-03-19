import numpy as np
import torch
import torch.nn as nn


# 1. CREATE SEQUENCES

def create_sequences(X, y, seq_length=10):
    sequences = []
    labels = []

    for i in range(len(X) - seq_length):
        seq = X[i:i+seq_length]
        label = y[i+seq_length]

        sequences.append(seq)
        labels.append(label)

    return np.array(sequences), np.array(labels)


# 2. LSTM MODEL

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        out = self.sigmoid(out)
        return out



# 3. TRAIN FUNCTION

def train_lstm(model, X_train, y_train, epochs=5):
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    # Loss + optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    for epoch in range(epochs):
        model.train()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
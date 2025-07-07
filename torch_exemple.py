import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import core.weightusageanalyzer as wua 

torch.manual_seed(42)
rng = np.random.default_rng(seed=42)

# Dataset
X_np = rng.random((1000, 2)).astype(np.float32)
y_np = ((X_np[:, 0] + X_np[:, 1]) > 1).astype(np.float32)

X = torch.from_numpy(X_np)
y = torch.from_numpy(y_np).unsqueeze(1) 

# A "too" big PyTorch model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = MLP()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
model.train()
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X)
    predicted = (outputs > 0.5).float()
    accuracy = (predicted == y).float().mean().item()

print(f"\n✅ Accuracy : {accuracy:.4f}")
wua.print_flops_report(model, nb_epochs=50, dataset=X)

importance_list = wua.compute_weight_importance(model, X)
for importance, weights, name in importance_list:
    report, norm_importance = wua.generate_report(importance, weights)
    print(f"\n📌 Report for the layer : {name}")
    wua.print_report(report)
    wua.plot_importance_histogram(norm_importance)

print("\n================================")
wua.show(model, X)

################################################################################################
# We observe that the model is too complex for this simple dataset, we want to reduce the number of FLOPS and make the model more efficient.

# Simplifying the model
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 7)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(7, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = SimpleMLP()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
model.train()
for epoch in range(60):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X)
    predicted = (outputs > 0.5).float()
    accuracy = (predicted == y).float().mean().item()

print(f"\n✅ Accuracy : {accuracy:.4f}")
wua.print_flops_report(model, nb_epochs=60, dataset=X)

print("\n================================")
wua.show(model, X)

# We observe that the model is now more efficient, with fewer FLOPS and a simpler architecture, while maintaining a good accuracy on the dataset.
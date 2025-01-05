import torch
import torch.nn as nn

# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # First layer: 2 inputs, 4 outputs
        self.activation1 = nn.ReLU()  # Activation function (ReLU)
        self.fc2 = nn.Linear(4, 1)  # Second layer: 4 inputs, 1 output
        self.activation2 = nn.Sigmoid()  # Activation function (Sigmoid)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation1(x)  # Apply ReLU
        x = self.fc2(x)
        x = self.activation2(x)  # Apply Sigmoid
        return x

# Example input
inputs = torch.tensor([[0.5, 0.2], [0.8, 0.1]], dtype=torch.float32)

# Instantiate and forward pass
model = SimpleNN()
outputs = model(inputs)
print("Network Outputs:")
print(outputs)

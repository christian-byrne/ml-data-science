import torch
import torch.nn as nn

# Define a simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(2, 1)  # Single layer

    def forward(self, x):
        return self.fc(x)

# Example data
inputs = torch.tensor([[0.5, 0.2], [0.8, 0.1]], requires_grad=True)
targets = torch.tensor([[1.0], [0.0]])

# Define model, loss, and optimizer
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Forward pass
outputs = model(inputs)
loss = criterion(outputs, targets)
print("Loss:", loss.item())

# Backward pass
loss.backward()

# Update parameters
optimizer.step()

# Display gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name} gradients:")
        print(param.grad)

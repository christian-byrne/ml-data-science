import numpy as np

# Define the forward pass
def forward_pass(inputs, weights, biases, activation):
    """
    Perform a forward pass through a layer.

    Parameters:
        inputs (np.ndarray): Input data.
        weights (np.ndarray): Weights matrix.
        biases (np.ndarray): Bias vector.
        activation (callable): Activation function.
    
    Returns:
        np.ndarray: Layer output after activation.
    """
    z = np.dot(inputs, weights) + biases  # Linear combination
    return activation(z)

# Example inputs and parameters
inputs = np.array([[0.5, 0.2], [0.8, 0.1]])
weights = np.array([[0.1, 0.3], [0.2, 0.4]])
biases = np.array([0.1, 0.2])

# Activation function (ReLU)
relu = lambda x: np.maximum(0, x)

# Forward pass
outputs = forward_pass(inputs, weights, biases, relu)
print("Forward Propagation Outputs:")
print(outputs)

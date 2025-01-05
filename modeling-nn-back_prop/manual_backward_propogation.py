def backward_pass(inputs, weights, biases, activation, activation_derivative, target):
    """
    Perform a backward pass through a single layer.

    Parameters:
        inputs (np.ndarray): Input data.
        weights (np.ndarray): Weights matrix.
        biases (np.ndarray): Bias vector.
        activation (callable): Activation function.
        activation_derivative (callable): Derivative of the activation function.
        target (np.ndarray): Target outputs.

    Returns:
        dict: Gradients for weights and biases.
    """
    # Forward pass
    z = np.dot(inputs, weights) + biases
    outputs = activation(z)

    # Loss (MSE) and its derivative
    loss = np.mean((outputs - target) ** 2)
    loss_derivative = 2 * (outputs - target) / len(target)

    # Backpropagate through activation
    activation_grad = loss_derivative * activation_derivative(z)

    # Gradients for weights and biases
    grad_weights = np.dot(inputs.T, activation_grad)
    grad_biases = np.sum(activation_grad, axis=0)

    return {"loss": loss, "grad_weights": grad_weights, "grad_biases": grad_biases}

# Example data
inputs = np.array([[0.5, 0.2]])
weights = np.array([[0.1, 0.3], [0.2, 0.4]])
biases = np.array([0.1, 0.2])
target = np.array([[0.5, 0.3]])

# Activation function and its derivative
relu = lambda x: np.maximum(0, x)
relu_derivative = lambda x: (x > 0).astype(float)

# Backpropagation
gradients = backward_pass(inputs, weights, biases, relu, relu_derivative, target)
print("Gradients:")
print(gradients)

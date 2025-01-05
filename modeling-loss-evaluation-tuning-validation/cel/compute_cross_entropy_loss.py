from sklearn.metrics import log_loss

# Example data
y_true = np.array([1, 0, 1, 0])
y_pred_probs = np.array([0.9, 0.2, 0.8, 0.4])  # Predicted probabilities

# Compute Log Loss
logloss = log_loss(y_true, y_pred_probs)
print("Log Loss (Binary Cross-Entropy):", logloss)

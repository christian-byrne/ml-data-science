from sklearn.metrics import hinge_loss

# Example data
y_true = np.array([1, -1, 1, -1])  # SVM requires -1, 1 labels
y_pred_scores = np.array([0.8, -0.5, 0.9, -0.2])  # Predicted decision scores

# Compute Hinge Loss
hinge = hinge_loss(y_true, y_pred_scores)
print("Hinge Loss:", hinge)

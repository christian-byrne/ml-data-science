from sklearn.metrics import accuracy_score

# Example data
y_true = np.array([0, 1, 1, 0, 1, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1, 1])

# Compute Accuracy
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

from sklearn.metrics import f1_score

# Example data
y_true = np.array([0, 1, 1, 0, 1, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1, 1])

# Compute F1 Score
f1 = f1_score(y_true, y_pred)
print("F1 Score:", f1)

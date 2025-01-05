from sklearn.metrics import r2_score

# Example data
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

# Compute R²
r2 = r2_score(y_true, y_pred)
print("R-squared (R²):", r2)

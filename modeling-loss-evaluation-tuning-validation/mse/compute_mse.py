from sklearn.metrics import mean_squared_error

# Example data
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

# Compute MSE
mse = mean_squared_error(y_true, y_pred)
print("Mean Squared Error (MSE):", mse)

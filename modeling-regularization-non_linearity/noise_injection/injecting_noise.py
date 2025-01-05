from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Generate synthetic regression data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 3 * X.squeeze() + np.random.randn(100) * 2  # Add noise to the output

# Add Gaussian noise to inputs
X_noisy = X + np.random.normal(0, 0.5, X.shape)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_noisy, y)

# Predict and evaluate
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error with Noise Injection: {mse:.4f}")

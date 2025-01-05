import numpy as np

# Parameters of the normal distribution
mu = 70
sigma = 2

# Create an array of evenly spaced values in the range [20, 120] at increments of 0.01 excluding 120
x = np.arange(20, 120, 0.01)
x_fine = np.linspace(20, 120, 1000)

# Calculate the PDF values at each location x
p = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-0.5 * ((x - mu) / sigma)**2)
p_fine = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-0.5 * ((x_fine - mu) / sigma)**2)

# Calculate the spacing between grid points
delta_x = x[1] - x[0]
delta_x_fine = x_fine[1] - x_fine[0]

# Calculate the Riemann sum approximation of the integral
riemann_sum = np.sum(p * delta_x)
riemann_sum_fine = np.sum(p_fine * delta_x_fine)

riemann_sum, riemann_sum_fine
# >>> (1.0, 1.0)
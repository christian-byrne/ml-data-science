import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d

# --- Define the Custom PDF ---
def pdf(x):
    """
    Custom probability density function (PDF).
    Example: A simple mixture of two Gaussian-like functions.
    """
    return 0.3 * np.exp(-0.5 * ((x - 2) / 0.5)**2) + 0.7 * np.exp(-0.5 * ((x + 1) / 0.8)**2)

# --- Create the CDF ---
def create_cdf(pdf, x_range, num_points=1000):
    """
    Numerically integrate the PDF to create the CDF.
    
    Parameters:
        pdf (function): Probability density function.
        x_range (tuple): Range of x values.
        num_points (int): Number of points to sample.
    
    Returns:
        np.ndarray, np.ndarray: x values and corresponding CDF values.
    """
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    cdf_values = np.zeros_like(x_values)
    for i, x in enumerate(x_values):
        cdf_values[i], _ = quad(pdf, x_range[0], x)  # Integrate PDF from start to x
    cdf_values /= cdf_values[-1]  # Normalize to [0, 1]
    return x_values, cdf_values

# --- Perform Inverse Transform Sampling ---
def inverse_transform_sampling(cdf_x, cdf_y, num_samples):
    """
    Perform inverse transform sampling to generate random samples.
    
    Parameters:
        cdf_x (np.ndarray): x values of the CDF.
        cdf_y (np.ndarray): y values of the CDF.
        num_samples (int): Number of samples to generate.
    
    Returns:
        np.ndarray: Random samples from the target distribution.
    """
    # Interpolate the inverse CDF
    inverse_cdf = interp1d(cdf_y, cdf_x, bounds_error=False, fill_value=(cdf_x[0], cdf_x[-1]))
    
    # Generate uniform random numbers and map through inverse CDF
    uniform_samples = np.random.rand(num_samples)
    return inverse_cdf(uniform_samples)

# --- Visualization ---
def visualize_distributions(pdf, samples, x_range, num_points=1000):
    """
    Visualize the original PDF and the histogram of the generated samples.
    
    Parameters:
        pdf (function): Original PDF function.
        samples (np.ndarray): Generated samples.
        x_range (tuple): Range of x values.
        num_points (int): Number of points for the PDF curve.
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y_pdf = pdf(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_pdf, label="Original PDF", linewidth=2)
    plt.hist(samples, bins=30, density=True, alpha=0.6, label="Sampled Distribution")
    plt.title("Inverse Transform Sampling: Before and After")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.grid()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Define the range and number of samples
    x_range = (-4, 4)
    num_samples = 5000

    # Create the CDF
    cdf_x, cdf_y = create_cdf(pdf, x_range)

    # Perform inverse transform sampling
    samples = inverse_transform_sampling(cdf_x, cdf_y, num_samples)

    # Visualize the results
    visualize_distributions(pdf, samples, x_range)

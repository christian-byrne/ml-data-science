import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Theoretical Background ---
# A Joint Probability Density Function (PDF) defines the likelihood of two continuous random variables X and Y.
# For continuous variables, the joint PDF f(X, Y) satisfies:
# 1. Non-negativity: f(X, Y) >= 0
# 2. Normalization: Integral over all values of X and Y equals 1.

# Marginal PDFs can be computed by integrating the joint PDF:
# f_X(x) = Integral of f(X, Y) over all Y
# f_Y(y) = Integral of f(X, Y) over all X

# If the joint PDF is the product of its marginals, X and Y are independent:
# f(X, Y) = f_X(x) * f_Y(y)

# --- Define the Joint PDF ---
def joint_pdf(x, y):
    """
    Joint PDF for two continuous variables X and Y.
    Example: A bivariate Gaussian distribution with correlation.
    """
    mu_x, mu_y = 0, 0  # Means
    sigma_x, sigma_y = 1, 1  # Standard deviations
    rho = 0.5  # Correlation coefficient
    z = ((x - mu_x)**2 / sigma_x**2) + ((y - mu_y)**2 / sigma_y**2) - (2 * rho * (x - mu_x) * (y - mu_y) / (sigma_x * sigma_y))
    return (1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2))) * np.exp(-z / (2 * (1 - rho**2)))

# --- Create a Grid ---
def create_grid(x_range, y_range, num_points=100):
    """
    Create a grid of x and y values for joint PDF evaluation.
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    return X, Y

# --- Compute Marginal PDFs ---
def compute_marginals(X, Y, Z, axis=0):
    """
    Compute marginal PDFs by integrating the joint PDF.
    
    Parameters:
        X, Y: Grid values for X and Y.
        Z: Joint PDF values on the grid.
        axis: Axis along which to integrate (0 for Y, 1 for X).
    
    Returns:
        Marginal PDF values as a 1D array.
    """
    dx = X[0, 1] - X[0, 0]  # Step size in X
    dy = Y[1, 0] - Y[0, 0]  # Step size in Y
    if axis == 0:  # Marginalize over Y
        return np.trapz(Z, dx=dy, axis=0)
    else:  # Marginalize over X
        return np.trapz(Z, dx=dx, axis=1)

# --- Check Independence ---
def check_independence(Z, marginal_x, marginal_y):
    """
    Check independence by comparing the joint PDF to the product of marginals.
    """
    product_marginals = np.outer(marginal_y, marginal_x)
    independence = np.allclose(Z, product_marginals)
    return independence, product_marginals

# --- Compute Conditional PDF ---
def compute_conditional_pdf(Z, marginal_y):
    """
    Compute conditional PDF P(X | Y).
    
    Parameters:
        Z: Joint PDF values on the grid.
        marginal_y: Marginal PDF for Y.
    
    Returns:
        Conditional PDF values as a 2D array.
    """
    conditional_pdf = Z / marginal_y[:, None]  # Divide each row by the corresponding marginal
    return conditional_pdf

# --- Visualization Utilities ---
def visualize_joint_pdf(X, Y, Z, title="Joint PDF"):
    """
    Visualize the joint PDF using a heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(Z, xticklabels=False, yticklabels=False, cmap="viridis")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="f(X, Y)")
    plt.show()

def visualize_marginals(x, marginal_x, y, marginal_y):
    """
    Visualize marginal PDFs for X and Y.
    """
    plt.figure(figsize=(12, 5))
    
    # Marginal for X
    plt.subplot(1, 2, 1)
    sns.lineplot(x=x, y=marginal_x, label='Marginal PDF of X')
    plt.xlabel('X')
    plt.ylabel('Density')
    plt.title('Marginal PDF of X')
    plt.grid()

    # Marginal for Y
    plt.subplot(1, 2, 2)
    sns.lineplot(x=y, y=marginal_y, label='Marginal PDF of Y')
    plt.xlabel('Y')
    plt.ylabel('Density')
    plt.title('Marginal PDF of Y')
    plt.grid()

    plt.tight_layout()
    plt.show()

def visualize_conditional_pdf(X, Y, Z_conditional):
    """
    Visualize conditional PDF P(X | Y).
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(Z_conditional, xticklabels=False, yticklabels=False, cmap="coolwarm")
    plt.title("Conditional PDF P(X | Y)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="P(X | Y)")
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Define ranges and grid resolution
    x_range = (-3, 3)
    y_range = (-3, 3)
    num_points = 100

    # Create grid and evaluate joint PDF
    X, Y = create_grid(x_range, y_range, num_points)
    Z = joint_pdf(X, Y)

    # Visualize joint PDF
    visualize_joint_pdf(X, Y, Z)

    # Compute marginal PDFs
    marginal_x = compute_marginals(X, Y, Z, axis=1)
    marginal_y = compute_marginals(X, Y, Z, axis=0)

    # Visualize marginals
    visualize_marginals(X[0, :], marginal_x, Y[:, 0], marginal_y)

    # Check independence
    independence, product_marginals = check_independence(Z, marginal_x, marginal_y)
    if independence:
        print("X and Y are independent.")
    else:
        print("X and Y are not independent.")

    # Visualize product of marginals for comparison
    visualize_joint_pdf(X, Y, product_marginals, title="Product of Marginal PDFs")

    # Compute conditional PDF P(X | Y)
    conditional_pdf = compute_conditional_pdf(Z, marginal_y)
    visualize_conditional_pdf(X, Y, conditional_pdf)

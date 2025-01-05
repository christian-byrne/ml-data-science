import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the joint PDF as a continuous function
def joint_pdf(x, y):
    """
    Joint Probability Density Function (PDF).
    Assume a Gaussian-like joint distribution for demonstration purposes.
    """
    mu_x, mu_y = 0, 0  # Means
    sigma_x, sigma_y = 1, 1  # Standard deviations
    rho = 0.5  # Correlation coefficient
    z = ((x - mu_x)**2 / sigma_x**2) + ((y - mu_y)**2 / sigma_y**2) - (2 * rho * (x - mu_x) * (y - mu_y) / (sigma_x * sigma_y))
    return (1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2))) * np.exp(-z / (2 * (1 - rho**2)))

# Create a grid for visualization and calculations
def create_grid(x_range, y_range, num_points=100):
    """
    Create a grid of x and y values for evaluation of the joint PDF.
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    return X, Y

# Compute marginal PDFs by integration
def compute_marginals(X, Y, joint_pdf_values, axis=0):
    """
    Compute marginal PDFs by integrating the joint PDF along one axis.
    """
    marginal_pdf = np.trapz(joint_pdf_values, X if axis == 1 else Y, axis=axis)
    return marginal_pdf

# Check independence by comparing the product of marginals to the joint PDF
def check_independence(joint_pdf_values, marginal_x, marginal_y, X, Y):
    """
    Check independence between variables by comparing the joint PDF with the product of marginals.
    """
    product_of_marginals = np.outer(marginal_x, marginal_y)
    independence_check = np.isclose(joint_pdf_values, product_of_marginals)
    return independence_check, product_of_marginals

# Conditional distribution P(X | Y)
def compute_conditional_pdf(joint_pdf_values, marginal_y):
    """
    Compute conditional PDF P(X | Y).
    """
    conditional_pdf = joint_pdf_values / marginal_y[:, None]
    return conditional_pdf

# Visualization utilities
def visualize_joint_pdf(X, Y, Z, title="Joint PDF"):
    """
    Visualize the joint PDF using a heatmap.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(Z, xticklabels=False, yticklabels=False, cmap="viridis")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def visualize_marginal_pdf(values, range_vals, label="Marginal PDF"):
    """
    Visualize marginal PDFs.
    """
    plt.figure(figsize=(8, 4))
    sns.lineplot(x=range_vals, y=values, label=label)
    plt.title(label)
    plt.xlabel("Variable")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Define the range and resolution for the grid
    x_range = (-3, 3)
    y_range = (-3, 3)
    num_points = 100

    # Create grid and evaluate the joint PDF
    X, Y = create_grid(x_range, y_range, num_points)
    Z = joint_pdf(X, Y)

    # Visualize the joint PDF
    visualize_joint_pdf(X, Y, Z, title="Joint PDF of X and Y")

    # Compute marginal PDFs
    marginal_x = compute_marginals(X, Y, Z, axis=1)
    marginal_y = compute_marginals(X, Y, Z, axis=0)

    # Visualize marginal PDFs
    visualize_marginal_pdf(marginal_x, X[0, :], label="Marginal PDF of X")
    visualize_marginal_pdf(marginal_y, Y[:, 0], label="Marginal PDF of Y")

    # Check for independence
    independence_check, product_of_marginals = check_independence(Z, marginal_x, marginal_y, X, Y)

    # Visualize independence check
    visualize_joint_pdf(X, Y, product_of_marginals, title="Product of Marginal PDFs")
    print("Are X and Y independent?", independence_check.all())

    # Compute and visualize conditional PDF P(X | Y)
    conditional_pdf = compute_conditional_pdf(Z, marginal_y)
    visualize_joint_pdf(X, Y, conditional_pdf, title="Conditional PDF P(X | Y)")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Helper Function: Create a Random Bivariate PMF ---
def generate_bivariate_pmf(values_x, values_y, random_seed=None):
    """
    Generate a random bivariate PMF for two discrete variables X and Y.
    
    Parameters:
        values_x (list): Possible values of X.
        values_y (list): Possible values of Y.
        random_seed (int, optional): Random seed for reproducibility.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['X', 'Y', 'P(X,Y)'].
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate random probabilities for each combination of X and Y
    probabilities = np.random.rand(len(values_x), len(values_y))
    probabilities /= probabilities.sum()  # Normalize to sum to 1

    # Create the PMF as a DataFrame
    pmf_data = []
    for i, x in enumerate(values_x):
        for j, y in enumerate(values_y):
            pmf_data.append([x, y, probabilities[i, j]])
    return pd.DataFrame(pmf_data, columns=['X', 'Y', 'P(X,Y)'])

# --- Function: Compute Marginal PMFs ---
def compute_marginals(pmf):
    """
    Compute marginal PMFs from a bivariate PMF.
    
    Parameters:
        pmf (pd.DataFrame): DataFrame with columns ['X', 'Y', 'P(X,Y)'].
    
    Returns:
        pd.DataFrame, pd.DataFrame: Marginal PMFs for X and Y.
    """
    marginal_x = pmf.groupby('X')['P(X,Y)'].sum().reset_index()
    marginal_x.columns = ['X', 'P(X)']

    marginal_y = pmf.groupby('Y')['P(X,Y)'].sum().reset_index()
    marginal_y.columns = ['Y', 'P(Y)']

    return marginal_x, marginal_y

# --- Function: Visualize PMF ---
def visualize_pmf(pmf):
    """
    Visualize a bivariate PMF as a heatmap.
    
    Parameters:
        pmf (pd.DataFrame): DataFrame with columns ['X', 'Y', 'P(X,Y)'].
    """
    pivot_table = pmf.pivot(index='Y', columns='X', values='P(X,Y)')
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="Blues")
    plt.title('Bivariate PMF (P(X,Y))')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# --- Function: Visualize Marginals ---
def visualize_marginals(marginal_x, marginal_y):
    """
    Visualize marginal PMFs for X and Y.
    
    Parameters:
        marginal_x (pd.DataFrame): DataFrame with columns ['X', 'P(X)'].
        marginal_y (pd.DataFrame): DataFrame with columns ['Y', 'P(Y)'].
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Marginal PMF for X
    sns.barplot(x='X', y='P(X)', data=marginal_x, ax=axes[0], palette="viridis")
    axes[0].set_title('Marginal PMF for X')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('P(X)')

    # Marginal PMF for Y
    sns.barplot(x='Y', y='P(Y)', data=marginal_y, ax=axes[1], palette="plasma")
    axes[1].set_title('Marginal PMF for Y')
    axes[1].set_xlabel('Y')
    axes[1].set_ylabel('P(Y)')

    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Generate a random bivariate PMF
    values_x = [1, 2, 3, 4]
    values_y = ['A', 'B', 'C']
    pmf = generate_bivariate_pmf(values_x, values_y, random_seed=42)

    # Display the PMF
    print("Bivariate PMF:")
    print(pmf)

    # Visualize the bivariate PMF
    visualize_pmf(pmf)

    # Compute marginal distributions
    marginal_x, marginal_y = compute_marginals(pmf)

    # Display the marginal PMFs
    print("\nMarginal PMF for X:")
    print(marginal_x)
    print("\nMarginal PMF for Y:")
    print(marginal_y)

    # Visualize marginal distributions
    visualize_marginals(marginal_x, marginal_y)

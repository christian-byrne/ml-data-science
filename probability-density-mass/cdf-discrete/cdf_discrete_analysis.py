import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Theoretical Background ---
# A Cumulative Distribution Function (CDF) gives the probability that a random variable
# takes a value less than or equal to a specified value. For a discrete random variable,
# the CDF is a step function that sums the probabilities of all values up to the given point.

# For a discrete random variable X with possible values {x1, x2, ..., xn}, the CDF is defined as:
# F(x) = P(X <= x) = Sum_{xi <= x} P(X = xi)

# --- Example Data ---
# Define a discrete probability mass function (PMF) for a random variable
pmf = pd.DataFrame({
    'X': [1, 2, 3, 4, 5],  # Possible values of the random variable
    'P(X)': [0.1, 0.2, 0.4, 0.2, 0.1]  # Corresponding probabilities
})

# --- Step 1: Compute the CDF ---
def compute_cdf(pmf):
    """
    Compute the CDF from a given PMF.
    
    Parameters:
        pmf (pd.DataFrame): DataFrame containing the PMF with columns 'X' and 'P(X)'.
    
    Returns:
        pd.DataFrame: DataFrame with columns 'X', 'P(X)', and 'F(X)' (CDF values).
    """
    pmf = pmf.sort_values('X').reset_index(drop=True)  # Ensure the PMF is sorted
    pmf['F(X)'] = pmf['P(X)'].cumsum()  # CDF is the cumulative sum of the PMF
    return pmf

cdf = compute_cdf(pmf)

# --- Step 2: Visualize the PMF and CDF ---
def visualize_pmf_cdf(pmf, cdf):
    """
    Visualize the PMF and CDF using Matplotlib.
    
    Parameters:
        pmf (pd.DataFrame): DataFrame containing the PMF.
        cdf (pd.DataFrame): DataFrame containing the CDF.
    """
    # PMF visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(pmf['X'], pmf['P(X)'], color='skyblue', alpha=0.7, label='PMF')
    plt.xlabel('X')
    plt.ylabel('P(X)')
    plt.title('Probability Mass Function (PMF)')
    plt.grid(axis='y')
    plt.legend()

    # CDF visualization
    plt.subplot(1, 2, 2)
    plt.step(cdf['X'], cdf['F(X)'], where='post', color='orange', label='CDF')
    plt.xlabel('X')
    plt.ylabel('F(X)')
    plt.title('Cumulative Distribution Function (CDF)')
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()
    plt.show()

visualize_pmf_cdf(pmf, cdf)

# --- Step 3: Compute Quantiles ---
def compute_quantile(cdf, p):
    """
    Compute the quantile for a given probability value p.
    
    Parameters:
        cdf (pd.DataFrame): DataFrame containing the CDF.
        p (float): Probability value (0 <= p <= 1).
    
    Returns:
        int: Quantile (smallest X such that F(X) >= p).
    """
    if not (0 <= p <= 1):
        raise ValueError("Probability must be between 0 and 1.")
    quantile = cdf[cdf['F(X)'] >= p].iloc[0]['X']
    return quantile

# Example: Find the 50th percentile (median)
median = compute_quantile(cdf, 0.5)
print(f"The 50th percentile (median) is: {median}")

# --- Step 4: Theoretical Discussion ---
# - Discrete CDF is a step function: It changes values only at specific points in the support of the random variable.
# - CDF properties:
#   1. F(x) is non-decreasing: If x1 <= x2, then F(x1) <= F(x2).
#   2. F(x) is bounded: 0 <= F(x) <= 1 for all x.
#   3. F(x) approaches 0 as x -> -∞ and 1 as x -> ∞.
# - Applications:
#   - Compute probabilities: P(a <= X <= b) = F(b) - F(a-1).
#   - Compute quantiles (e.g., median, percentiles).
#   - Visualize the distribution of a random variable.

# --- Step 5: Use Case in Machine Learning ---
# In ML, discrete CDFs are useful for:
# - Feature engineering: Encoding ordinal features based on cumulative probabilities.
# - Sampling: Generate synthetic data following a specific distribution.
# - Evaluation: Analyze discrete prediction distributions, such as in classification tasks.

# --- Example Use Case: Encoding Ordinal Features ---
def encode_ordinal_with_cdf(cdf, values):
    """
    Encode ordinal feature values using the CDF.
    
    Parameters:
        cdf (pd.DataFrame): DataFrame containing the CDF.
        values (list): List of values to encode.
    
    Returns:
        list: Encoded values (cumulative probabilities).
    """
    encoded = [cdf[cdf['X'] == val]['F(X)'].values[0] for val in values]
    return encoded

# Example: Encode a list of values
values_to_encode = [1, 3, 4]
encoded_values = encode_ordinal_with_cdf(cdf, values_to_encode)
print(f"Encoded values using CDF: {encoded_values}")

import numpy as np
from scipy.stats import entropy

def kl_divergence(p, q):
    """
    Calculate the Kullback-Leibler (KL) Divergence between two distributions.
    
    Parameters:
        p (np.ndarray): True distribution (must sum to 1).
        q (np.ndarray): Approximation distribution (must sum to 1).
    
    Returns:
        float: KL divergence (non-negative).
    """
    # Ensure distributions are normalized
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= np.sum(p)
    q /= np.sum(q)
    
    # Add a small value to avoid division by zero or log(0)
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    
    return np.sum(p * np.log(p / q))

# Example usage
p = np.array([0.4, 0.35, 0.25])  # True distribution
q = np.array([0.5, 0.3, 0.2])    # Approximation distribution

kl_div = kl_divergence(p, q)
print(f"Kullback-Leibler (KL) Divergence: {kl_div:.4f}")

# Alternatively, using scipy's entropy function
kl_div_scipy = entropy(p, q)
print(f"Kullback-Leibler (KL) Divergence (Scipy): {kl_div_scipy:.4f}")

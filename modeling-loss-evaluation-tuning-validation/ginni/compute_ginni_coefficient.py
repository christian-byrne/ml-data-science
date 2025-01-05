import numpy as np

def gini_coefficient(array):
    """
    Calculate the Gini coefficient for a numpy array.
    
    Parameters:
        array (np.ndarray): Array of values (e.g., income, probabilities).
    
    Returns:
        float: Gini coefficient (0 = perfect equality, 1 = maximal inequality).
    """
    array = np.sort(array)  # Sort the array
    n = len(array)
    cumulative = np.cumsum(array)
    gini = (2 * np.sum((np.arange(1, n + 1) / n) * array) - np.sum(array)) / np.sum(array)
    return 1 - gini

# Example usage
values = np.array([1, 2, 3, 4, 5])
print("Gini Coefficient:", gini_coefficient(values))

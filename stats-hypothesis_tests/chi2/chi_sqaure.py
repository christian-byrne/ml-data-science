import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

def calculate_chi_square(table):
    chi2, p, dof, expected = chi2_contingency(
        table,
        correction=False # Don't apply Yates' correction for continuity, because it is not used in this class (although in real world it should be used)
        )
    contributions = (table - expected) ** 2 / expected
    return chi2, p, dof, expected, contributions

def main():
    print("Chi-Square Test Calculator\n")
    
    # Enter your contingency table as a list of lists
    table = [
        [8, 69, 105, 243],
        [11, 57, 55, 63],
        [46, 161, 101, 82]
    ]
    
    # Convert to a NumPy array
    table = np.array(table)
    
    # Calculate Chi-Square stats
    chi2, p_value, dof, expected, contributions = calculate_chi_square(table)
    
    # Initial Data
    print("Pre-Test Conditions:")
    print(f"Contingency Table:")
    print(pd.DataFrame(table))
    
    # Pre-test conditions
    print("\nHypotheses:")
    print("H_0: The two variables are independent (there is NO relationship between them)")
    print("H_a: The two variables are dependent (there IS a relationship between them)")
    print("Significance Level: 0.05")

    print("\nConditions:")
    print("1. Random Sample: Yes")
    print("2. Independent Observations: Yes")
    print("3. Categorical Data: Yes")
    print("4. Expected Frequencies: All expected frequencies are at least 5")
    print("5. All expected values > 5")
    print(pd.DataFrame(expected))
    print("\nExpected Frequencies > 5:")
    print(pd.DataFrame(expected > 5)) # Create dataframe of bools for expected frequencies >= 5

    # Display the results
    print("\nContributions to Chi-Square Statistic:")
    print(pd.DataFrame(contributions))
    
    print("Results:")
    print(f"Chi-Square Statistic: {chi2:.4f}")
    print(f"P-Value: {p_value:.4f}")
    print(f"Degrees of Freedom: {dof}")

    print
    
if __name__ == "__main__":
    main()

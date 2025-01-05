import numpy as np
import pandas as pd
from scipy.stats import f_oneway

def anova_table(data):
    # Number of groups
    k = len(data)
    
    # Total number of observations
    n_total = sum([group[0] for group in data])
    
    # Overall mean
    all_observations = [value for group in data for value in group[1:]]
    grand_mean = np.mean(all_observations)
    
    # SS_total: Total sum of squares
    ss_total = sum([(value - grand_mean)**2 for group in data for value in group[1:]])
    
    # SS_tr: Between-group (treatment) sum of squares
    ss_tr = sum([group[0] * (np.mean(group[1:]) - grand_mean)**2 for group in data])
    
    # SS_e: Within-group (error) sum of squares
    ss_e = sum([sum([(value - np.mean(group[1:]))**2 for value in group[1:]]) for group in data])
    
    # Degrees of freedom
    df_tr = k - 1
    df_e = n_total - k
    
    # Mean squares
    ms_tr = ss_tr / df_tr
    ms_e = ss_e / df_e
    
    # F-statistic
    f_statistic = ms_tr / ms_e
    
    # p-value
    p_value = f_oneway(*[group[1:] for group in data]).pvalue
    
    # Create the ANOVA table as a pandas DataFrame
    anova_table = pd.DataFrame({
        'Source of Variation': ['Treatment', 'Error', 'Total'],
        'SS': [ss_tr, ss_e, ss_total],
        'df': [df_tr, df_e, df_tr + df_e],
        'MS': [ms_tr, ms_e, ''],
        'F': [f_statistic, '', ''],
        'p-value': [p_value, '', '']
    })
    
    # Print the ANOVA table
    print(anova_table)

# NOTE: The first column should represent the number of observations in each group
data = [
    [67, 214, 3.194, 3.079],          # Group 1
    [37, 137.8, 3.724, 2.942],          # Group 2
    [77, 297.2, 3.859, 2.775],          # Group 3
    [41, 127.4, 3.107, 2.326],          # Group 4
    # [2.9, 3.4, 3.2]           # Group 5
]

anova_table(data)

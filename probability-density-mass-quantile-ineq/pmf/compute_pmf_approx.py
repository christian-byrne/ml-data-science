import numpy as np
import pandas as pd

# Define the joint probability distribution table
joint_pmf = pd.DataFrame({
    'A': [0, 0, 0, 0, 1, 1, 1, 1],
    'B': [0, 0, 1, 1, 0, 0, 1, 1],
    'C': [0, 1, 0, 1, 0, 1, 0, 1],
    'P(A,B,C)': [0.01, 0.07, 0.02, 0.10, 0.02, 0.30, 0.04, 0.44]
})

# 1. Compute the marginal P(A, B)
def compute_marginal_ab(joint_pmf):
    marginal_ab = joint_pmf.groupby(['A', 'B']).agg({'P(A,B,C)': 'sum'}).reset_index()
    marginal_ab.columns = ['A', 'B', 'P(A,B)']
    return marginal_ab

# 2. Compute the marginals P(A) and P(B)
def compute_marginal_a_b(joint_pmf):
    marginal_a = joint_pmf.groupby(['A']).agg({'P(A,B,C)': 'sum'}).reset_index()
    marginal_a.columns = ['A', 'P(A)']

    marginal_b = joint_pmf.groupby(['B']).agg({'P(A,B,C)': 'sum'}).reset_index()
    marginal_b.columns = ['B', 'P(B)']
    
    return marginal_a, marginal_b

# 3. Check if A and B are independent
def check_independence(marginal_ab, marginal_a, marginal_b):
    ab_independence = marginal_ab.copy()
    for idx, row in ab_independence.iterrows():
        a_prob = marginal_a[marginal_a['A'] == row['A']]['P(A)'].values[0]
        b_prob = marginal_b[marginal_b['B'] == row['B']]['P(B)'].values[0]
        ab_independence.loc[idx, 'P(A)*P(B)'] = a_prob * b_prob
    ab_independence['Independent'] = np.isclose(ab_independence['P(A,B)'], ab_independence['P(A)*P(B)'])
    is_independent = ab_independence['Independent'].all()
    return ab_independence, is_independent

# 4. Compute conditional distribution P(A,B | C)
def compute_conditional_ab_given_c(joint_pmf):
    marginal_c = joint_pmf.groupby(['C']).agg({'P(A,B,C)': 'sum'}).reset_index()
    marginal_c.columns = ['C', 'P(C)']

    joint_with_conditional = joint_pmf.merge(marginal_c, on='C')
    joint_with_conditional['P(A,B | C)'] = joint_with_conditional['P(A,B,C)'] / joint_with_conditional['P(C)']
    
    conditional_ab_given_c_0 = joint_with_conditional[joint_with_conditional['C'] == 0][['A', 'B', 'P(A,B | C)']].reset_index(drop=True)
    conditional_ab_given_c_1 = joint_with_conditional[joint_with_conditional['C'] == 1][['A', 'B', 'P(A,B | C)']].reset_index(drop=True)
    
    return conditional_ab_given_c_0, conditional_ab_given_c_1

# Perform calculations
marginal_ab = compute_marginal_ab(joint_pmf)
marginal_a, marginal_b = compute_marginal_a_b(joint_pmf)
independence_table, is_independent = check_independence(marginal_ab, marginal_a, marginal_b)
conditional_ab_given_c_0, conditional_ab_given_c_1 = compute_conditional_ab_given_c(joint_pmf)

# 1. Print Marginal P(A, B)
print("\n1. Marginal P(A, B):")
print(marginal_ab)

# 2. Print Marginal P(A) and P(B)
print("\n2. Marginal P(A):")
print(marginal_a)

print("\nMarginal P(B):")
print(marginal_b)

# 3. Print Independence check
print("\n3. Independence check between A and B:")
if is_independent:
    print("A and B are independent.")
else:
    print("A and B are not independent.")
print(independence_table[['A', 'B', 'P(A,B)', 'P(A)*P(B)', 'Independent']])

# 4. Print Conditional P(A,B | C)
print("\n4. Conditional P(A,B | C=0):")
print(conditional_ab_given_c_0)

print("\nConditional P(A,B | C=1):")
print(conditional_ab_given_c_1)

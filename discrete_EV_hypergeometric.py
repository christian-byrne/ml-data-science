from scipy.stats import hypergeom

# Problem parameters
total_boys = 100
total_girls = 150
total_students = total_boys + total_girls  # Total number of students
sample_size = 70  # Number of students selected

# Hypergeometric distribution parameters
# X is the number of boys selected
# Y is the number of girls selected

# Mean of X (number of boys selected)
mean_boys = hypergeom.mean(M=total_students, n=total_boys, N=sample_size)

# Mean of Y (number of girls selected)
mean_girls = hypergeom.mean(M=total_students, n=total_girls, N=sample_size)

# Calculate E(X - Y)
expected_value_X_minus_Y = mean_boys - mean_girls

# Display all the relevant information and calculations
print("Problem Parameters:")
print(f"Total number of boys (N_boys): {total_boys}")
print(f"Total number of girls (N_girls): {total_girls}")
print(f"Total number of students (N_total): {total_students}")
print(f"Sample size (n): {sample_size}")
print("\nCalculations:")
print(f"Mean number of boys selected (E[X]): {mean_boys}")
print(f"Mean number of girls selected (E[Y]): {mean_girls}")
print(f"Expected value of X - Y (E[X - Y]): {expected_value_X_minus_Y}")

import sympy as sp

# Define the variables
x, y, c = sp.symbols('x y c')

# Define the joint PDF f(x, y) = c * (x + y**2) for 0 <= x <= 1 and 0 <= y <= 1
joint_pdf = c * (x + y**2)

# 1. Calculate the marginal PDF of Y by integrating over x from 0 to 1
marginal_y = sp.integrate(joint_pdf, (x, 0, 1))
marginal_y_simplified = sp.simplify(marginal_y)

# Evaluate the marginal PDF of Y for y = 1/2
y_value = 1/2
marginal_y_at_1_2 = marginal_y_simplified.subs(y, y_value)

# 2. Calculate P(X <= 1/2, Y = 1/2) by integrating the joint PDF over x from 0 to 1/2
joint_pdf_at_y_1_2 = joint_pdf.subs(y, y_value)
joint_prob_x_leq_1_2_y_1_2 = sp.integrate(joint_pdf_at_y_1_2, (x, 0, 1/2))

# 3. Calculate the conditional probability P(X <= 1/2 | Y = 1/2)
conditional_prob = joint_prob_x_leq_1_2_y_1_2 / marginal_y_at_1_2

# Output the results
joint_prob_x_leq_1_2_y_1_2_val = joint_prob_x_leq_1_2_y_1_2.subs(c, 1)  # Assume c=1 for the final answer
marginal_y_at_1_2_val = marginal_y_at_1_2.subs(c, 1)
conditional_prob_val = conditional_prob.subs(c, 1)

print(f"P(X <= 1/2, Y = 1/2) = {joint_prob_x_leq_1_2_y_1_2_val}")
print(f"Marginal P(Y = 1/2) = {marginal_y_at_1_2_val}")
print(f"P(X <= 1/2 | Y = 1/2) = {conditional_prob_val}")

from scipy.stats import norm

# Parameters of the normal distribution
mu = 3
sigma = 4

# Calculate the probability P(X > -2)
p_x_gt_minus_2 = 1 - norm.cdf(-2, loc=mu, scale=sigma)

p_x_gt_minus_2
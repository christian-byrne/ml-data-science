
import numpy as np
import scipy.stats as stats

sample = [164,182,176,149,184,190,160,139,175,148]

sample_mean = np.mean(sample)
print(f"Sample mean: {sample_mean}")

s = 0
for x in sample:
    s += (x - sample_mean) ** 2


sample_variance = s / (len(sample) - 1)
print(f"Sample variance: {sample_variance}")
print(f"Sample stdv: {np.sqrt(sample_variance)}")

sample_std = np.std(sample, ddof=1)
print(f"Sample standard deviation: {sample_std}")

ninety_five_ci = stats.t.interval(
    df=len(sample) - 1,
    loc=sample_mean,
    scale=sample_std / np.sqrt(len(sample)),
    confidence=0.95,
)

print(f"95% confidence interval: {ninety_five_ci}")


t_value = 2.03

p_value = stats.t.sf(t_value, df=9 - 1) * 2

print(f"p-value: {p_value}")

prob_gt_t = stats.t.sf(t_value, df=9 - 1)

print(f"Probability of observing a t-value greater than 2.03: {prob_gt_t}")

p_value = 1 - stats.t.cdf((9**(.5) * 2) / (8.75 ** (.5)), df=9 - 1)
print(f"p-value: {p_value}")

print(stats.chi2.sf(9.078, df=1))

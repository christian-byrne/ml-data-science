import numpy as np



data = np.array([16, 16, 512])
print(data)
print("Mean:", np.mean(data))
print("Standard Deviation:", np.std(data))
print("Proportion above 40:", np.mean(data > 40))

print(f"Confidence Interval 95% for mu: {np.percentile(data, [2.5, 97.5])}")
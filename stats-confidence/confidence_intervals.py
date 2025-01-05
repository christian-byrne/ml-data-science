import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

# Parameters
population_mean = 50        # True population mean
population_std_dev = 3     # True population standard deviation
sample_size = 40            # Size of each sample
num_samples = 100           # Number of samples to take
confidence_level = 0.05     # Desired confidence level (e.g., 95%)

# Calculate the critical z-value for the given confidence level
z_critical = st.norm.ppf(1 - (1 - confidence_level) / 2)

# Generate samples and calculate confidence intervals
sample_means = []
confidence_intervals = []
captured_population_mean = 0

for _ in range(num_samples):
    sample = np.random.normal(population_mean, population_std_dev, sample_size)
    sample_mean = np.mean(sample)
    sample_means.append(sample_mean)

    margin_of_error = z_critical * (population_std_dev / np.sqrt(sample_size))
    confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
    confidence_intervals.append(confidence_interval)

    if population_mean >= confidence_interval[0] and population_mean <= confidence_interval[1]:
        captured_population_mean += 1

# Plot the results
plt.figure(figsize=(12, 6))
plt.errorbar(
    range(1, num_samples + 1), 
    sample_means, 
    yerr=[(ci[1]-ci[0])/2 for ci in confidence_intervals], 
    fmt='o', 
    capsize=5
)
plt.axhline(population_mean, color='red', linestyle='dashed', linewidth=1, label='True Population Mean')
plt.xlabel('Sample Number')
plt.ylabel('Sample Mean')
plt.title(f'Confidence Intervals ({confidence_level * 100:.0f}%) for {num_samples} Samples (Size={sample_size})')
plt.legend()
plt.grid(axis='y')

# Display capture percentage
capture_percentage = (captured_population_mean / num_samples) * 100
# print(f'Percentage of confidence intervals capturing the true population mean: {capture_percentage:.1f}%')

# Add general annotation
annotation_text = (f"This plot shows {num_samples} confidence intervals "
                   f"for a population mean of {population_mean:.1f} "
                   f"(sample size = {sample_size}).\n"
                   f"{capture_percentage:.1f}% of the intervals capture the "
                   f"true population mean.")
plt.text(0.5, 0.05, annotation_text, ha='center', va='bottom', transform=plt.gca().transAxes)  
annotation_position = (0.5, 0.05)  # Below the plot (uncomment if desired)

plt.text(*annotation_position, annotation_text, ha='center', va='top')  

filename = "confidence_interval_plot2.png"  # Choose your desired filename and extension
plt.savefig(filename)

print(f"Plot saved as '{filename}' in the same directory as this script.")

plt.show()

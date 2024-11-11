import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset_url = "https://raw.githubusercontent.com/cpethe/TED_Talks/master/ted_main.csv"
ted_main = pd.read_csv(dataset_url)

# Filtered data points with copies
four_sets_of_data_points = [
    ted_main[ted_main['comments'] >= 100].copy(),
    ted_main[ted_main['comments'] >= 500].copy(),
    ted_main[ted_main['comments'] >= 1000].copy(),
    ted_main.copy()
]

# Log-transformed values
for data_points in four_sets_of_data_points:
    data_points['log_comments'] = np.log(data_points['comments'])
    data_points['log_views'] = np.log(data_points['views'])

# Scatter plot
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# [(row, column, min_comments, max_comments, title, scatter_size)]
subplot_properties = [
    # With so many data points, we need to reduce the size significantly
    (1, 1, 0, 0, 'No restriction', .3),
    # With so few data points, we should make use of the available space to emphasize the data
    (1, 0, 1000, 0, 'At least 1000 comments', 70), 
    # Scale down the scatter size to about the same change in data point magnitude
    (0, 1, 500, 1000, 'At least 500 comments', 12),
    (0, 0, 100, 500, 'At least 100 comments', 1),
]

for row, column, min_comments, max_comments, title, scatter_size in subplot_properties:
    data_points = four_sets_of_data_points[row * 2 + column]
    axs[row, column].scatter(data_points['log_comments'], data_points['log_views'], s=scatter_size)
    axs[row, column].set_title(title)
    axs[row, column].set_xlabel('Log Comments')
    axs[row, column].set_ylabel('Log Views')

plt.tight_layout()
plt.show()
plt.savefig('scatter_plot.png')
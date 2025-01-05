import json
import pandas as pd


CSV = "https://raw.githubusercontent.com/cpethe/TED_Talks/master/ted_main.csv"

df = pd.read_csv(CSV)
df.drop('main_speaker', axis=1)

# Which talk is the most viewed as compared to its related talks (the one with the maximum difference between its views and the view count of any of its related talks)?

def related_diff(row):
  related_views = []
  for related in row["related_talks"]:
    try:
      related_id = json.loads(related)["id"]
      related_views.append(df[df["id"] == related_id]["views"].values[0])
    except json.JSONDecodeError: # Malformed data
      return 0
  return max(related_views) - row["views"]

df["related_talks"] = df["related_talks"].fillna("[]")
df["related_diff"] = df.apply(related_diff, axis=1)
max_diff = df["related_diff"].max()
most_viewed = df[df["related_diff"] == max_diff]
print(most_viewed["title"])



# Calculate which two cols are most correlated

max_correlated, max_correlation = max(
    [
        ((col1, col2), df[col1].corr(df[col2]))
        for col1 in df.select_dtypes(exclude="object").columns
        for col2 in df.select_dtypes(exclude="object").columns
        if col1 != col2
    ],
    key=lambda x: x[1],
)
      
# Research a way to find out the significance (in terms of p-value) of the correlation of a pair of features. Try it on the pair of columns duration and comments.

from scipy.stats import pearsonr

correlation, p_value = pearsonr(df["duration"], df["comments"])

correlation, p_value

print(f"The correlation between duration and comments is {correlation:.4f} with a p-value of {p_value:.4f}")

"""
The p-value roughly indicates the probability of an uncorrelated system producing datasets that have a Pearson correlation at least as extreme as the one computed from these datasets.

The p-value here represents the probability under the null hypothesis (the two features are uncorrelated) of obtaining a correlation as or more extreme than the one computed from the datasets. Since the probability of obtaining a correlation of 0.1407 is ~0.0000000000010, it's safe to reject the null hypothesis and conclude that the correlation between duration and comments is statistically significant. I.e., the correlation is not due to random chance and exists earnestly in the population of TED talks.
"""

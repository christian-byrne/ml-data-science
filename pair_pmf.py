import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 100_000

dice1 = np.random.randint(1, 7, size=N)
dice2 = np.random.randint(1, 7, size=N)

dicepairs = [(dice1[i], dice2[i]) for i in range(N)]

# Plot the dice pairs
df = pd.DataFrame(dicepairs, columns=["dice1", "dice2"])
df["sum"] = df["dice1"] + df["dice2"]
df["sum"].hist(bins=11, range=(1.5, 12.5), density=True, rwidth=0.8)
plt.xlabel("Sum of dice")
plt.ylabel("Probability")
plt.title("Sum of two dice")
plt.show()

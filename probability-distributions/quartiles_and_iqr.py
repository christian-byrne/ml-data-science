import numpy as np
from rich import print
import statistics

very_good = np.array([44, 49, 63, 42, 16, 77])
good = [74, 30, 52, 52, 69, 44, 46, 28, 55]
test = [71, 30, 53, 53, 67, 43, 48, 28, 54]

def get_quartiles_inclusive_method(dataset_1d: list):
    dataset_1d.sort()
    mid = len(dataset_1d) // 2

    print(f"Sorted dataset: {dataset_1d}")
    print(f"Midpoint: {mid}")

    # Adjust lower half for inclusive method
    lower_half = dataset_1d[:mid + 1] if len(dataset_1d) % 2 == 1 else dataset_1d[:mid]
    
    # Adjust upper half for inclusive method
    upper_half = dataset_1d[mid - 1:] if len(dataset_1d) % 2 == 1 else dataset_1d[mid:]

    q1 = statistics.median(lower_half)
    q2 = statistics.median(dataset_1d)
    q3 = statistics.median(upper_half)
    iqr = q3 - q1

    result = {
        "Q1 (lower quartile)": q1,
        "Q2 (median)": q2,
        "Q3 (upper quartile)": q3,
        "IQR": iqr,
    }
    print(result)

    return result

def get_quartiles_exclusive_method(dataset_1d: list):
    dataset_1d.sort()
    mid = len(dataset_1d) // 2

    q1 = statistics.median(dataset_1d[:mid])
    q2 = statistics.median(dataset_1d)
    q3 = statistics.median(dataset_1d[mid:])
    iqr = q3 - q1

    result = {
        "Q1 (lower quartile)": q1,
        "Q2 (median)": q2,
        "Q3 (upper quartile)": q3,
        "IQR": iqr,
    }
    print(result)

    return result

def get_quartiles_percentage_method(dataset_1d: list):
    q1 = np.percentile(dataset_1d, 25)
    q2 = np.percentile(dataset_1d, 50)
    q3 = np.percentile(dataset_1d, 75)
    iqr = q3 - q1

    result = {
        "Q1 (lower quartile)": q1,
        "Q2 (median)": q2,
        "Q3 (upper quartile)": q3,
        "IQR": iqr,
    }
    print(result)
    return result


# get_quartiles_inclusive_method(good)
# get_quartiles_exclusive_method(good)
# get_quartiles_percentage_method(good)
# get_quartiles(good)



import numpy as np
import matplotlib.pyplot as plt


class Data:
    def __init__(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev
        self.data = np.random.normal(self.mean, self.stdev, 1000)
        self.name = f"Data with mean {self.mean} and stdev {self.stdev}"

    def get_z_score(self, x):
        z_score = (x - self.mean) / self.stdev
        print(f"The z-score of {x} is {z_score}")
        return self

    def __str__(self):
        return f"{self.name}: {self.data}"

    def plot(self):
        plt.hist(self.data, bins=100, density=True)
        plt.title(self.name)
        plt.show()
        return self
    

Data(500, 70).get_z_score(370).get_z_score(545).get_z_score(480).get_z_score(670)
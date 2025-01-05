import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

from rich import print
from rich.panel import Panel
from rich.table import Table

def resid():
    """For when you have the data, but not a model given."""
    # Sample data
    X = np.array([0, 1, 2, 3, 4, 5])

    y = np.array([.07*0, .09*1, .23*2, .31*3, .21*4, .09*5])

    # Fit the linear regression model
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # Print the symbolic representation of the model
    print(model.summary())

    # Calculate leverage, Cook's distance, and DFFITS
    influence = model.get_influence()
    leverage = influence.hat_matrix_diag
    cooks_d = influence.cooks_distance[0]
    dffits = influence.dffits[0]

    # Identify influential points based on thresholds
    n = len(y)
    k = X.shape[1] - 1

    high_leverage_points = np.where(leverage > 2 * (k+1) / n)
    influential_cooks_d = np.where(cooks_d > 4 / n)
    influential_dffits = np.where(np.abs(dffits) > 2 * np.sqrt(k) / np.sqrt(n))

    table = Table(title="Residuals and Influential Points")
    table.add_column("Measure Name", justify="right")
    table.add_column("Value", justify="left")

    table.add_row("Residuals", str(model.resid))
    table.add_row("High leverage points", str(high_leverage_points))
    table.add_row("Influential points based on Cook's distance", str(influential_cooks_d))
    table.add_row("Influential points based on DFFITS", str(influential_dffits))

    q1 = np.percentile(y, 25)
    q2 = np.percentile(y, 50)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1

    print(table)

    table2 = Table(title="Quartiles and IQR", expand=False)
    table2.add_column("Measure Name", justify="right")
    table2.add_column("Value", justify="center")

    table2.add_row("Q1 (lower quartile)", str(q1))
    table2.add_row("Q2 (median)", str(q2))
    table2.add_row("Q3 (upper quartile)", str(q3))
    table2.add_row("IQR", str(iqr))
    table2.add_row("Mean", str(np.mean(y)))
    table2.add_row("Standard Deviation", str(np.std(y)))
    table2.add_row("Variance", str(np.var(y)))
    table2.add_row("Range", str(np.ptp(y)))

    r = model.rsquared
    r_squared = model.rsquared_adj
    table2.add_row("r", str(r))
    table2.add_row("r-squared", str(r_squared))

    print(table2)

    intercept = model.params[0]
    slope = model.params[1]
    print(Panel(
        f"Intercept: {intercept}\nSlope: {slope}\ny = {intercept} + {slope}x",
        title="Model",
        expand=False,
    ))


resid()
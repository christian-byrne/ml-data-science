import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.outliers_influence as oi


def resid():
    """For when you have the data, but not a model given."""
    # Sample data
    X = np.random.rand(100, 3)
    y = np.random.rand(100)
    X = np.array([188, 188, 178, 183, 180, 183, 193])

    y = np.array([95, 91, 72, 93, 78, 82, 98])

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

    print("High leverage points:", high_leverage_points)
    print("Influential points based on Cook's distance:", influential_cooks_d)
    print("Influential points based on DFFITS:", influential_dffits)


class Residuals:
    """For when you are given the model and data already, and need to perform basic residual analysis."""
    def __init__(self, model: callable, X: list, y: list):
        self.model = model
        self.X = X
        self.y = y

    def get_residuals(self):
        residuals_plot_x = []
        residuals_plot_y = []
        for i in range(len(self.X)):
            print(
                f"Input value: {self.X[i]}, Actual: {self.y[i]}, Predicted: {self.model(self.X[i])}, Residual: {self.y[i] - self.model(self.X[i]):.2f}"
            )
            residuals_plot_x.append(self.X[i])
            residuals_plot_y.append(self.y[i] - self.model(self.X[i]))

        self.residuals_x = residuals_plot_x
        self.residuals_y = residuals_plot_y

    def show_plot(self):
        if not hasattr(self, "residuals_x"):
            self.get_residuals()

        plt.scatter(self.residuals_x, self.residuals_y)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("Input value")
        plt.ylabel("Residual")
        plt.title("Residual Plot")
        plt.show()

    def map_predictions(self):
        self.predictions = [self.model(x) for x in self.X]

    def map_predictors(self):
        self.predictors = [self.X[i] for i in range(len(self.X))]
        self.k_predictors = len(self.predictors)

    def map_observations(self):
        self.observations = [self.y[i] for i in range(len(self.y))]
        self.n_observations = len(self.observations)

    def set_leverage_threshold(self):
        # self.leverage_threshold = 2 * (self.X.shape[1] + 1) / self.X.shape[0]
        self.leverage_threshold = 2 * (len(self.X) + 1) / len(self.X)

    # def get_influentials(self):
        
    #     for 


def regression(x):
    return -3.0112 + 0.4881 * x


X = [188, 4, 188, 178, 183, 180, 183, 193]

y = [95, 0.005, 91, 72, 93, 78, 82, 98]

residuals = Residuals(regression, X, y)

residuals.get_residuals()

residuals.show_plot()

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

from utils import generate_sample_data

np.random.seed(1002)

# Styling the matplotlib
plt.style.use("seaborn-v0_8-dark")

for param in ["figure.facecolor", "axes.facecolor", "savefig.facecolor"]:
    plt.rcParams[param] = "#212946"  # bluish dark grey

for param in ["text.color", "axes.labelcolor", "xtick.color", "ytick.color"]:
    plt.rcParams[param] = "0.9"  # very light grey

colors = [
    "#08F7FE",  # teal/cyan
    "#FE53BB",  # pink
    "#F5D300",  # yellow
    "#00ff41",  # matrix green
]


class LocallyWeightedRegression:
    """
    Locally Weighted Regression from scratch.
    """

    def __init__(self, tau: float) -> None:
        self.tau = tau

    def fit(self, X: ndarray, y: ndarray) -> None:
        """
        Training the data
        X: training data
        y: training data
        """

        self.X: ndarray = X
        self.y: ndarray = y.reshape(-1, 1)

    def predict(self, X: ndarray):
        """
        Predict the query
        X: query points
        """
        n, d = X.shape

        # predicted points
        y_pred = np.zeros(n)

        for i in range(n):
            w = self._weight(self.X, X[i], self.tau)
            theta = self._weighted_least_squares(self.X, self.y, w)
            y_pred[i] = np.dot(X[i], theta)

        return y_pred

    def _weight(self, X: ndarray, x: float | int, tau: float) -> ndarray:
        """
        Creating weight matrix
        X: query points
        x: training point
        tau: bandwidth
        """
        return np.diag(np.exp(-np.sum(np.square(X - x), axis=1) / (2 * np.square(tau))))

    def _weighted_least_squares(self, X, y, w) -> ndarray:
        X_tilde = np.dot(np.dot(X.T, w), X)
        y_tilde = np.dot(np.dot(X.T, w), y)
        return np.dot(np.linalg.inv(X_tilde), y_tilde)


def main():
    for i in np.arange(0.1, 1.1, 0.1):
        tau: float = i

        (x_train, y_train), (x_test, y_test) = generate_sample_data()

        lw = LocallyWeightedRegression(tau)

        lw.fit(x_train.reshape(-1, 1), y_train)

        y_pred = lw.predict(x_train.reshape(-1, 1))

        indices = x_train.argsort(0)
        xsort = x_train[indices]

        fig, ax = plt.subplots(figsize=(16, 9))
        fig.canvas.manager.full_screen_toggle()
        ax.scatter(x_train, y_train, color=colors[0])
        ax.plot(xsort, y_pred[indices], color=colors[1])
        ax.set_title(
            f"Locally Weighted Linear Regression\n$\\tau$ = {tau:0.2f}",
            fontsize=18, 
            color=colors[2]
        )
        ax.set_xlim(0, 5)
        ax.set_ylim(-1.5, 2.5)
        ax.grid(color='#2A3459')  # bluish dark grey, but slightly lighter than background

        fig.savefig(f'myplot_{tau:0.2f}.png', dpi=300)
        # plt.show()


if __name__ == "__main__":
    main()

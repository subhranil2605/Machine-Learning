import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

from utils import generate_sample_data


class LocallyWeightedRegression:
    def __init__(self, tau):
        self.tau = tau

    def fit(self, X, y):
        self.X = X
        self.y = y.reshape(-1, 1)

    def predict(self, X):
        m, n = X.shape
        y_pred = np.zeros(m)
        for i in range(m):
            w = self.weight(self.X, X[i], self.tau)
            theta = self.weighted_least_squares(self.X, self.y, w)
            y_pred[i] = np.dot(X[i], theta)
        return y_pred

    def weight(self, X, x, k):
        m, n = X.shape
        w = np.zeros(m)
        for i in range(m):
            xi = X[i]
            distance = np.sum(np.square(xi - x))
            w[i] = np.exp(-distance / (2 * np.square(k)))
        return np.diag(w)

    def weighted_least_squares(self, X, y, w):
        X_tilde = np.dot(np.dot(X.T, w), X)
        y_tilde = np.dot(np.dot(X.T, w), y)
        return np.dot(np.linalg.inv(X_tilde), y_tilde)


def main():
    (x_train, y_train), (x_test, y_test) = generate_sample_data()

    lw = LocallyWeightedRegression(tau=1)

    lw.fit(x_train.reshape(-1, 1), y_train)

    y_pred = lw.predict(x_train.reshape(-1, 1))

    indices = x_train.argsort(0)
    xsort = x_train[indices]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x_train, y_train, color="blue")
    ax.plot(xsort, y_pred[indices], color="red", linewidth=3)
    plt.show()


if __name__ == "__main__":
    main()

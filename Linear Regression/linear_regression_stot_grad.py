import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt


def cost_function(actuals: ndarray, preds: ndarray) -> float:
    return np.mean(np.power(preds - actuals, 2))



def st_gradient_descent(X: ndarray, 
                        y: ndarray, 
                        theta: ndarray,
                        learning_rate: float = 0.01, 
                        n_iters: int = 100):
    losses = []
    for _ in range(n_iters):
        for i in range(len(X)):
            losses.append(cost_function(X.dot(theta), y))
            theta -= learning_rate * (X[i].dot(theta) - y[i]) * X[i].reshape(-1, 1)

    return theta, losses



np.random.seed(54251)

data_size: int = 100

# initialize data
X: ndarray = np.linspace(0, 10, data_size)
y: ndarray = (3 * X + 5 + np.random.randn(data_size) * 3).reshape(-1, 1)
X: ndarray = X.reshape(-1, 1)

# adding intercept
X: ndarray = np.hstack([np.ones_like(X), X])

# initialize weights 
theta: ndarray = np.random.random((X.shape[1], 1))
print(theta)



theta, losses = st_gradient_descent(X , y, theta)
print(theta)


# plt.scatter(X[:, -1], y)
# plt.plot(X[:, -1], 3 * X[:, -1] + 5)
# plt.plot(X[:, -1], X.dot(theta))
plt.plot(losses)
plt.show()

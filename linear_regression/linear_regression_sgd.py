import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from tqdm import tqdm


def to_2d_array(a: ndarray, array_type: str = "col") -> ndarray:
    """
    Turns a 1D Tensor into 2D
    """

    assert a.ndim == 1, "Input tensors must be 1 dimensional"

    if array_type == "col":
        return a.reshape(-1, 1)
    elif array_type == "row":
        return a.reshape(1, -1)


def cost_function(actuals: ndarray, preds: ndarray) -> float:
    """
    Calculates the cost of the current weight
    Mean Squared Error
    """

    return np.mean(np.power(preds - actuals, 2))


def forward_loss(X: ndarray, y: ndarray, weights: ndarray) -> float:
    """
    Calculate the loss of the model
    """
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == weights.shape[0]

    # prediction using the weights
    h: ndarray = X.dot(weights)

    # cost of the current model
    loss: float = cost_function(y, h)

    return loss


def loss_gradient(x: ndarray, y: ndarray, weights: ndarray) -> ndarray:
    """
    Calculates Stotastic Gradient Descent
    """

    if x.ndim == 1:
        x = to_2d_array(x, "row")
    if y.ndim == 1:
        y = to_2d_array(y, "row")
    
    # dLdT = dLdh * dhdT
    dLdh: ndarray = 2 * (y[0] - x[0].dot(weights))
    dhdT: ndarray = np.transpose(x[0])
    dLdT: ndarray = dLdh * dhdT
    return dLdT


def init_weigths(X: ndarray) -> ndarray:
    """
    Initialize weights
    """

    return np.zeros((X.shape[1], 1))


def train(
        X: ndarray, 
        y: ndarray, 
        fit_intercept: bool = True,
        return_weights: bool = True,
        learning_rate: float = 0.01,
        n_iters: int = 100):
    
    X: ndarray = add_intercept(X, fit_intercept)

    if y.ndim == 1:
        y = to_2d_array(y)
    
    weights: ndarray = init_weigths(X)

    if return_weights:
        losses = []

    for _ in tqdm(range(n_iters)):
        loss = forward_loss(X, y, weights)

        if return_weights:
            losses.append(loss)
        
        for i in range(X.shape[0]):
            # calculate the loss gradient of each row
            loss_grads = to_2d_array(loss_gradient(X[i], y[i], weights))

            # update parameters
            weights += learning_rate * loss_grads
    
    if return_weights:
        return losses, weights
    
    return None


def add_intercept(X: ndarray, fit_intercept: bool = True):
    if X.ndim == 1:
        X: ndarray = np.vstack([np.ones(X.shape[0]), X]).T if fit_intercept else to_2d_array(X)
    else:
        X: ndarray = np.hstack([np.ones(X.shape[0]).reshape(-1, 1), X]) if fit_intercept else X
    return X


def func(X: ndarray, coeffs: list | ndarray, intercept: int, add_random: bool = True):
    assert X.shape[1] == len(coeffs)

    s: ndarray = np.zeros((X.shape[0], 1))

    for i in range(X.shape[1]):
        s += to_2d_array(X[:, i]) * coeffs[i]
    
    if add_random:
        s += intercept + np.random.random((X.shape[0], 1)) * 2
        return s
    
    return s + intercept
    



if __name__ == "__main__":
    np.random.seed(510210)
    # Simple data
    # X: ndarray = np.linspace(0, 10, 100)
    # y: ndarray = 3 * X + 10 + np.random.randn(100) * 2

    n_rows: int = 100
    n_features: int = 5

    X: ndarray = np.random.random((n_rows, n_features))

    coeffs: ndarray = np.random.randint(0, 10, size=n_features)
    intercept: int = np.random.randint(-10, 10)
    print(f"Actual Intercept: {intercept}\nCoefficients: {coeffs}")

    y: ndarray = func(X, coeffs=coeffs, intercept=intercept, add_random=True)

    losses, weights = train(X, y, return_weights=True, n_iters=1000)
    print(f"\nPredicted Intercept: {weights[0][0]}\nCoefficients: \n{weights[1:].reshape(1, -1)[0]}")

    plt.plot(losses)
    plt.show()

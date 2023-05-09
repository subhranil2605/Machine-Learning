import numpy as np
from numpy import ndarray


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

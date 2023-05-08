import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal
import pytest

from ..temp import to_2d_array, cost_function, forward_loss, loss_gradient


def test_to_2d_array():
    # test input tensor
    a = np.array([1, 2, 3, 4, 5])

    # test column array
    expected_col_array = np.array([[1], [2], [3], [4], [5]])
    assert_array_equal(to_2d_array(a, "col"), expected_col_array)

    # test row array
    expected_row_array = np.array([[1, 2, 3, 4, 5]])
    assert_array_equal(to_2d_array(a, "row"), expected_row_array)

    # test assertion error
    with pytest.raises(AssertionError):
        to_2d_array(np.array([[1, 2], [3, 4]]))


def test_cost_function():
    # test inputs
    actuals = np.array([1, 2, 3, 4, 5])
    preds = np.array([1.1, 1.9, 3.2, 3.8, 4.9])

    # test output
    expected_output = 0.022

    # test the output of the function
    assert_almost_equal(cost_function(actuals, preds), expected_output, decimal=3)


def test_forward_loss():
    # test inputs
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([3, 5, 7])
    weights = np.array([1, 1])

    # test output
    expected_output = 6.666666666

    # test the output of the function
    assert_almost_equal(forward_loss(X, y, weights), expected_output, decimal=3)

    # test with invalid input
    with pytest.raises(IndexError):
        forward_loss(np.array([1, 2]), np.array([3, 5]), np.array([1, 1, 1]))

# def test_loss_gradient():
#     # test inputs
#     x = np.array([1, 2, 3])
#     y = np.array([4])
#     weights = np.array([0.5, 0.5, 0.5])

#     # expected output
#     expected_output = np.array([-3, -6, -9])

#     # test the output of the function
#     assert np.array_equal(loss_gradient(x, y, weights), expected_output)
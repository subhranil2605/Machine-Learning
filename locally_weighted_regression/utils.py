import numpy as np

np.random.seed(1002)

def generate_sample_data():
    # Generate Sample Data
    x = np.arange(0, 5, 0.1)
    y = np.sin(x) + np.random.normal(scale=0.1, size=len(x))

    # Add some outliers
    x_outliers = np.array([1, 2.5, 3.5])
    y_outliers = np.array([2, -1, 1.5])


    # combine the data
    x_data = np.concatenate([x, x_outliers])
    y_data = np.concatenate([y, y_outliers])


    # shuffle the data
    idx = np.random.permutation(len(x_data))
    x_data = x_data[idx]
    y_data = y_data[idx]

    # Split the data into training and testing sets
    train_size = int(0.8 * len(x_data))
    x_train, y_train = x_data[:train_size], y_data[:train_size]
    x_test, y_test = x_data[train_size:], y_data[train_size:]

    # import matplotlib.pyplot as plt
    # plt.scatter(x_train, y_train, label="Training Data")
    # plt.scatter(x_test, y_test, label="Testing Data")
    # plt.scatter(x_outliers, y_outliers, label="Outliers Data")
    # plt.legend()
    # plt.show()

    return (x_train, y_train), (x_test, y_test)

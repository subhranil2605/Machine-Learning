import numpy as np
from numpy import ndarray


class LinearRegression:
    def __init__(self,
                 lr: float = 0.01,
                 n_iters: int = 1000,
                 fit_intercept: bool=True) -> None:
        
        self.lr: float = lr
        self.n_iters: int = n_iters
        self.fit_intercept: bool = fit_intercept
        self.theta: ndarray = None
        self.intercept_ = None
        self.coef_ = None
        self.is_model_trained = False


    def fit(self, X: ndarray, y: ndarray):
        X: ndarray = self.__adding_intercept(X)

        if y.ndim == 1:
            y: ndarray = y.reshape(-1, 1)

        # store the losses
        losses = np.zeros(self.n_iters)

        # initialize weights
        self.__init_weights(X)

        for i in range(self.n_iters):
            
            loss, forward_info = self.__forward_loss(X, y)
            losses[i] = loss

            # if len(losses) > 2:
            #     if np.abs(losses[len(losses) - 1] - losses[len(losses) - 2]) < 0.01:
            #         break
                
            
            loss_grads = self.__loss_gradients(forward_info)
            self.theta -= self.lr * loss_grads

        self.is_model_trained = True
        return losses

    def __adding_intercept(self, X: ndarray) -> ndarray:
        if X.ndim == 1:
            X: ndarray = np.vstack([np.ones(X.shape[0]), X]).T if self.fit_intercept else X.reshape(-1, 1)
        else:
            X: ndarray = np.hstack([np.ones(X.shape[0]).reshape(-1, 1), X]) if self.fit_intercept else X
        return X

    def predict(self, X_new: ndarray):
        
        y_predict = None
        if self.is_model_trained:
            X_new: ndarray = self.__adding_intercept(X_new)
            y_predict: ndarray = X_new.dot(self.theta)
            
        return y_predict
            


    def __init_weights(self, X: ndarray) -> ndarray:
        assert X.ndim != 1

        self.theta: ndarray = np.zeros((X.shape[1], 1))
        self.intercept_ = 0 if not self.fit_intercept else self.theta[0]
        self.coef_ = self.theta if not self.fit_intercept else self.theta[1:]


    def __forward_loss(self,
                       X: ndarray,
                       y: ndarray) -> tuple[float, dict[str, ndarray]]:

        # the size of the X and y should be same
        assert X.shape[0] == y.shape[0]

        # require for using the dot product
        assert X.shape[1] == self.theta.shape[0]

        # performing dot product
        h: ndarray = X.dot(self.theta)

        # calculating the cost function
        J: float = np.mean(np.power(y - h, 2))

        forward_info: dict[str, ndarray] = {
            "X": X,   
	    "h": h,
	    "y": y
	}
	
        return J, forward_info

    def __loss_gradients(self, forward_info: dict[str, ndarray]) -> ndarray:
        dLdh: ndarray = (2 / forward_info["X"].shape[0]) * (forward_info["h"] - forward_info["y"])
        dhdt: ndarray = np.transpose(forward_info["X"], (1, 0))
        dLdt: ndarray = np.dot(dhdt, dLdh)
        return dLdt



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    linear_regressior = LinearRegression(fit_intercept=True)

    # create sample data
    x: ndarray = np.linspace(0, 10, 100)
    y: ndarray = 3 * x + 10 + np.random.randn(100) * 2
    X: ndarray = np.vstack([np.ones(x.shape[0]), x]).T
    losses = linear_regressior.fit(x, y)

##    data_url = "http://lib.stat.cmu.edu/datasets/boston"
##    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
##    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
##    target = raw_df.values[1::2, 2]
##    s = StandardScaler()
##    data = s.fit_transform(data)
##    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)
##    y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
##    losses = linear_regressior.fit(X_train, y_train)
    
    bias = linear_regressior.intercept_
    weights = linear_regressior.coef_

    print(bias, weights)
    
##    x_test = np.array([[0], [2]])
##    print(linear_regressior.predict(x_test))

    plt.subplot(1, 2, 1)
    plt.scatter(x, y)
    plt.plot(x, X.dot(linear_regressior.theta), 'orange')

    plt.subplot(1, 2, 2)
    plt.plot(losses)
    
    plt.show()
    

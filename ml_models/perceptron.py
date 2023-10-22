import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.003, n_iters=10000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Fix the seed for repeatibility
        np.random.seed(42)
        # initialize weights and bias
        self.weights = np.random.random(X.shape[1])
        self.bias = np.random.random(1)

        for _ in range(self.n_iters):
            self._update_weights(X, y)

    def _update_weights(self, X, y):
        y_lin = np.dot(X, self.weights) + self.bias
        y_proba = self._sigmoid(y_lin) 
        z = self._activation(y_proba) 

        dw = (1.0 / X.shape[0]) * np.dot(X.T, (z - y))
        db = (1.0 / X.shape[0]) * np.sum(z - y)

        self.weights -= self.lr * dw
        self.bias -= self.lr * db

    def _activation(self, z):
        return np.where(z >= 0.5, 1, 0) 

    def _sigmoid(self, z):
        return 1.0 / (1+np.exp(-z))

    def predict(self, X):
        y_lin = np.dot(X, self.weights) + self.bias
        y_proba = self._sigmoid(y_lin) 
        z = self._activation(y_proba) 
        return z 

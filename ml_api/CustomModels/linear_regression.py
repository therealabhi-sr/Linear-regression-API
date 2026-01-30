import numpy as np

class LinearRegression:
    def __init__(self):
        self.w = None
        self.b = None
        self.lr = 0.0001
        self.epochs = 100000
        self.scale = None

    def train(self, x, y):
        X = np.asarray(x, dtype=float)
        ytrue = np.asarray(y, dtype=float)

        # store column-wise scale
        self.scale = np.max(X, axis=0)

        X_scaled = X / self.scale

        m, n = X_scaled.shape
        self.w = np.zeros(n)
        self.b = 0

        for _ in range(self.epochs):
            ypred = X_scaled.dot(self.w) + self.b
            error = ypred - ytrue

            dw = (2/m) * X_scaled.T.dot(error)
            db = (2/m) * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X_scaled = X / self.scale
        return X_scaled.dot(self.w) + self.b

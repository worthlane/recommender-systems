import numpy as np

class EASE:
    def __init__(self, reg=100.0):
        self.reg = reg

    def fit(self, positives):
        self.history = positives
        gram = (self.history.T @ self.history).toarray()
        gram.flat[::gram.shape[0] + 1] += self.reg
        precision = np.linalg.solve(gram, np.eye(gram.shape[0]))
        self.weights = -precision / np.diag(precision)
        np.fill_diagonal(self.weights, 0)
        return self

    def score(self, users):
        return self.history[users] @ self.weights

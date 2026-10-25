import numpy as np

class ItemKNN:
    def __init__(self, neighbors=40):
        self.neighbors = neighbors

    def fit(self, interactions):
        self.history = interactions
        gram = (self.history.T @ self.history).toarray()
        norms = np.sqrt(np.diag(gram))
        similarity = gram / np.maximum(norms[:, None] * norms[None, :], 1e-12)
        np.fill_diagonal(similarity, 0)

        discard = np.argsort(-similarity, axis=0, kind="stable")[self.neighbors:]
        np.put_along_axis(similarity, discard, 0, axis=0)
        self.weights = similarity
        return self

    def score(self, users):
        return self.history[users] @ self.weights

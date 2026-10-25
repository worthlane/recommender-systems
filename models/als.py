import numpy as np

N_FACTORS = 16

class ALS:
    def __init__(self, reg=1.0):
        self.reg = reg

    def fit(self, ratings):
        targets = ratings.copy()
        targets.data = self.make_target(targets.data)
        rng = np.random.default_rng(42)
        self.item_factors = rng.normal(0, 0.05, (ratings.shape[1], N_FACTORS))
        transposed = targets.T.tocsr()
        for _ in range(6):
            self.user_factors = self._update(targets, self.item_factors)
            self.item_factors = self._update(transposed, self.user_factors)
        return self

    def make_target(self, ratings):
        return (ratings >= 4).astype(np.float64)

    def _update(self, targets, fixed):
        base = fixed.T.dot(fixed) + self.reg * np.eye(N_FACTORS)
        result = np.zeros((targets.shape[0], N_FACTORS))
        for row in range(targets.shape[0]):
            start, end = targets.indptr[row:row + 2]
            vectors = fixed[targets.indices[start:end]]
            if not len(vectors):
                continue
            lhs = base + 20 * vectors.T.dot(vectors)
            rhs = 21 * vectors.T.dot(targets.data[start:end])
            result[row] = np.linalg.solve(lhs, rhs)
        return result

    def score(self, users):
        return self.user_factors[users].dot(self.item_factors.T)

import numpy as np
import torch
from torch import nn

class FM(nn.Module):
    def __init__(self, factors=16):
        super().__init__()
        self.factors = factors

    def forward(self, users, items):
        fields = torch.cat((users[:, None], self.item_fields[items]), dim=1)
        vectors = self.embeddings(fields)
        interactions = 0.5 * (vectors.sum(1).square() - vectors.square().sum(1)).sum(1)
        return self.bias + self.linear(fields).sum((1, 2)) + interactions

    def fit(self, ratings, category_codes):
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        n_users, n_items = ratings.shape
        self.n_items = n_items

        self.item_fields = torch.tensor(np.column_stack([
            np.arange(n_items) + n_users, category_codes + n_users + n_items,
        ]))
        n_features = n_users + n_items + int(category_codes.max()) + 1
        self.embeddings = nn.Embedding(n_features, self.factors)
        self.linear = nn.Embedding(n_features, 1)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.embeddings.weight, std=0.05)
        nn.init.zeros_(self.linear.weight)

        observations = ratings.tocoo()
        users = torch.tensor(observations.row, dtype=torch.long)
        items = torch.tensor(observations.col, dtype=torch.long)
        labels = torch.tensor(observations.data, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=1e-5)
        for _ in range(5):
            order = rng.permutation(len(labels))
            for start in range(0, len(order), 4096):
                batch = order[start:start + 4096]
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(self(users[batch], items[batch]), labels[batch])
                loss.backward()
                optimizer.step()
        return self

    @torch.no_grad()
    def score(self, users):
        u = torch.tensor(np.repeat(users, self.n_items), dtype=torch.long)
        i = torch.arange(self.n_items).repeat(len(users))
        return self(u, i).reshape(len(users), self.n_items).numpy()

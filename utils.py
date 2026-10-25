import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import sparse


def load_data():
    directory = Path("data")
    directory.mkdir(exist_ok=True)
    base = "https://www.kaggle.com/api/v1/datasets/download/khoa3chuwxa/amazon-home-kitchen-and-grocery-gourmet/"
    for name in ["Grocery_and_Gourmet_Food_5.json", "meta_Grocery_and_Gourmet_Food.json"]:
        path = directory / name
        if not path.exists():
            temporary = path.with_suffix(".part")
            response = requests.get(base + name + "%2F" + name + "?datasetVersionNumber=1")
            response.raise_for_status()
            temporary.write_bytes(response.content)
            temporary.replace(path)
    with (directory / "Grocery_and_Gourmet_Food_5.json").open() as stream:
        rows = []
        for line in stream:
            row = json.loads(line)
            rows.append([row.get(key) for key in
                         ["reviewerID", "asin", "overall", "unixReviewTime"]])
    reviews = pd.DataFrame(rows, columns=["user", "item", "rating", "timestamp"])
    with (directory / "meta_Grocery_and_Gourmet_Food.json").open() as stream:
        rows = []
        for line in stream:
            row = json.loads(line)
            categories = row.get("category") or []
            category = categories[1] if len(categories) > 1 else "unknown"
            rows.append((row["asin"], category))
    return reviews, pd.DataFrame(rows, columns=["item", "category"])


def prepare_data(raw, metadata):
    clean = raw.dropna()
    clean = clean[clean.rating.between(1, 5) & (clean.timestamp > 0)]
    clean = clean.sort_values("timestamp", kind="stable").drop_duplicates(["user", "item"])
    cut_train, cut_valid = clean.timestamp.quantile([0.70, 0.85]).astype(int)
    popularity = (clean[clean.timestamp < cut_train].groupby("item").size()
                  .rename("count").reset_index()
                  .sort_values(["count", "item"], ascending=[False, True]))
    items = sorted(popularity.head(1500).item)
    sample = clean[clean.item.isin(items)].copy()
    sample["u"] = pd.factorize(sample.user, sort=True)[0]
    sample["i"] = pd.factorize(sample.item, sort=True)[0]
    train = sample[sample.timestamp < cut_train]
    validation = sample[(sample.timestamp >= cut_train) & (sample.timestamp < cut_valid)]
    test = sample[sample.timestamp >= cut_valid]
    item_metadata = (metadata.drop_duplicates("item").set_index("item")
                     .reindex(items).fillna("unknown"))
    return sample, train, validation, test, item_metadata


def rating_matrix(frame, shape):
    return sparse.csr_matrix((frame.rating, (frame.u, frame.i)), shape=shape)


def positive_matrix(ratings):
    result = ratings.copy()
    result.data = (result.data >= 4).astype(float)
    result.eliminate_zeros()
    return result


def evaluation_set(heldout, warm_users):
    relevant = heldout[(heldout.rating >= 4) & heldout.u.isin(warm_users)]
    truth = {user: group.i.to_numpy() for user, group in relevant.groupby("u")}
    users = np.array(sorted(truth))
    return users, truth


def recommend(model, users, seen):
    scores = model.score(users)
    rows, columns = seen[users].nonzero()
    scores[rows, columns] = -np.inf
    return np.argsort(-scores, axis=1, kind="stable")[:, :10]


def evaluate(model, users, truth, seen):
    recalls, ndcgs = [], []
    discounts = 1 / np.log2(np.arange(2, 12))
    for start in range(0, len(users), 64):
        batch = users[start:start + 64]
        for user, top in zip(batch, recommend(model, batch, seen)):
            relevant = truth[user]
            hits = np.isin(top, relevant)
            recalls.append(hits.sum() / len(relevant))
            ndcgs.append((hits * discounts).sum() / discounts[:min(10, len(relevant))].sum())
    return {"Recall@10": np.mean(recalls), "NDCG@10": np.mean(ndcgs)}

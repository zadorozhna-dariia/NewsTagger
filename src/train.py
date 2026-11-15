"""Train and evaluate NewsTagger's rule baselines and model on the time split (design doc 5.4)."""

import os
import time

import numpy as np
import pandas as pd
import psycopg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

FEATURE_VERSION = "v1"
TRAIN_END = "2017-01-01"  # train: 2012-01 .. 2016-12
VALID_END = "2018-01-01"  # validation: 2017
TEST_END = "2018-06-01"  # test: 2018-01 .. 2018-05

MODEL_PARAMS = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_features": 200_000,
    "C": 4.0,
    "class_weight": "balanced",
}


def load_features():
    url = (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ.get('POSTGRES_HOST', 'localhost')}:{os.environ.get('POSTGRES_PORT', '5433')}"
        "/newstagger"
    )
    columns = ["date", "text", "author_top_category", "category"]
    query = f"SELECT {', '.join(columns)} FROM features WHERE feature_version = %s"
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(query, (FEATURE_VERSION,))
        features = pd.DataFrame(cur.fetchall(), columns=columns)
    features["date"] = pd.to_datetime(features["date"])
    return features


def time_split(features):
    date = features["date"]
    train = features[date < TRAIN_END]
    valid = features[(date >= TRAIN_END) & (date < VALID_END)]
    test = features[(date >= VALID_END) & (date < TEST_END)]
    return train, valid, test


def build_model():
    vectorizer = TfidfVectorizer(
        ngram_range=MODEL_PARAMS["ngram_range"],
        min_df=MODEL_PARAMS["min_df"],
        max_features=MODEL_PARAMS["max_features"],
        sublinear_tf=True,
    )
    classifier = LogisticRegression(
        C=MODEL_PARAMS["C"], class_weight=MODEL_PARAMS["class_weight"], max_iter=1000
    )
    return make_pipeline(vectorizer, classifier)


def top3_accuracy(model, texts, labels):
    proba = model.predict_proba(texts)
    top3 = model.classes_[np.argsort(-proba, axis=1)[:, :3]]
    return float(np.mean([label in row for label, row in zip(labels, top3, strict=True)]))


def macro_f1(labels, predicted):
    return f1_score(labels, predicted, average="macro", zero_division=0)


def main():
    features = load_features()
    train, valid, test = time_split(features)
    unseen = sorted(set(test["category"]) - set(train["category"]))
    print(f"rows: train={len(train)} valid={len(valid)} test={len(test)}")
    print(f"test categories never seen in training: {unseen}")

    majority = train["category"].mode()[0]
    rules = {
        f"Majority class ({majority})": lambda s: np.full(len(s), majority),
        "Author rule": lambda s: np.where(
            s["author_top_category"] != "unknown", s["author_top_category"], majority
        ),
    }

    results = []
    for name, predict in rules.items():
        results.append(
            {
                "model": name,
                "valid_macro_f1": macro_f1(valid["category"], predict(valid)),
                "test_macro_f1": macro_f1(test["category"], predict(test)),
                "test_accuracy": accuracy_score(test["category"], predict(test)),
                "test_top3": None,
            }
        )

    started = time.time()
    model = build_model().fit(train["text"], train["category"])
    print(f"model trained in {time.time() - started:.0f} s")
    results.append(
        {
            "model": "TF-IDF + logistic regression",
            "valid_macro_f1": macro_f1(valid["category"], model.predict(valid["text"])),
            "test_macro_f1": macro_f1(test["category"], model.predict(test["text"])),
            "test_accuracy": accuracy_score(test["category"], model.predict(test["text"])),
            "test_top3": top3_accuracy(model, test["text"], test["category"]),
        }
    )

    print("\n| Model | Valid macro-F1 | Test macro-F1 | Test accuracy | Test top-3 |")
    print("|---|---|---|---|---|")
    for r in results:
        top3 = "n/a" if r["test_top3"] is None else f"{r['test_top3']:.3f}"
        print(
            f"| {r['model']} | {r['valid_macro_f1']:.3f} | {r['test_macro_f1']:.3f} "
            f"| {r['test_accuracy']:.3f} | {top3} |"
        )


if __name__ == "__main__":
    main()

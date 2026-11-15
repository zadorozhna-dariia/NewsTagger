"""Train and evaluate NewsTagger's rule baselines and model on the time split (design doc 5.4).

Every run is logged to MLflow with its parameters, metrics and lineage tags (data version,
git commit, feature version, feature checksum). With --register the model is registered as a
new version of newstagger-classifier; with --reproduces N the run is checked against version N.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import psycopg
import yaml
from mlflow import MlflowClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DVC_FILE = REPO_ROOT / "data" / "raw" / "News_Category_Dataset_v3.json.dvc"

EXPERIMENT = "newstagger"
MODEL_NAME = "newstagger-classifier"
REPRODUCE_TOLERANCE = 0.005  # design doc 4.2: macro-F1 within +-0.005

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
    columns = ["article_id", "date", "text", "author_top_category", "category"]
    query = (
        f"SELECT {', '.join(columns)} FROM features "
        f"WHERE feature_version = %s AND date < %s ORDER BY article_id"
    )
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(query, (FEATURE_VERSION, TEST_END))
        features = pd.DataFrame(cur.fetchall(), columns=columns)
    features["date"] = pd.to_datetime(features["date"])
    return features


def features_checksum(features):
    """Fingerprint of the exact rows used, so a reproduction can prove it read the same data."""
    digest = hashlib.md5()
    for article_id, label in zip(features["article_id"], features["category"], strict=True):
        digest.update(f"{article_id}:{label}\n".encode())
    return digest.hexdigest()


def data_version():
    """DVC hash of the raw dataset, read from its pointer file."""
    return yaml.safe_load(RAW_DVC_FILE.read_text())["outs"][0]["md5"]


def git_state():
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    return git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))


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


def evaluate(train, valid, test):
    majority = train["category"].mode()[0]
    rules = {
        "majority": lambda s: np.full(len(s), majority),
        "author_rule": lambda s: np.where(
            s["author_top_category"] != "unknown", s["author_top_category"], majority
        ),
    }
    metrics = {}
    for name, predict in rules.items():
        metrics[f"{name}_valid_macro_f1"] = macro_f1(valid["category"], predict(valid))
        metrics[f"{name}_test_macro_f1"] = macro_f1(test["category"], predict(test))
        metrics[f"{name}_test_accuracy"] = accuracy_score(test["category"], predict(test))

    started = time.time()
    model = build_model().fit(train["text"], train["category"])
    # MLflow saves sklearn models with skops, which writes every numpy scalar as its own file;
    # with numpy ints in the 200k-term vocabulary that takes minutes. Plain ints save in seconds.
    vectorizer = model.named_steps["tfidfvectorizer"]
    vectorizer.vocabulary_ = {term: int(index) for term, index in vectorizer.vocabulary_.items()}
    metrics["train_seconds"] = time.time() - started
    metrics["valid_macro_f1"] = macro_f1(valid["category"], model.predict(valid["text"]))
    metrics["test_macro_f1"] = macro_f1(test["category"], model.predict(test["text"]))
    metrics["test_accuracy"] = accuracy_score(test["category"], model.predict(test["text"]))
    metrics["test_top3"] = top3_accuracy(model, test["text"], test["category"])
    return model, metrics, majority


def print_table(metrics, majority):
    rows = [
        (f"Majority class ({majority})", "majority_", None),
        ("Author rule", "author_rule_", None),
        ("TF-IDF + logistic regression", "", metrics["test_top3"]),
    ]
    print("\n| Model | Valid macro-F1 | Test macro-F1 | Test accuracy | Test top-3 |")
    print("|---|---|---|---|---|")
    for name, prefix, top3 in rows:
        top3_text = "n/a" if top3 is None else f"{top3:.3f}"
        print(
            f"| {name} | {metrics[prefix + 'valid_macro_f1']:.3f} "
            f"| {metrics[prefix + 'test_macro_f1']:.3f} "
            f"| {metrics[prefix + 'test_accuracy']:.3f} | {top3_text} |"
        )


def check_reproduction(client, version, checksum, metrics):
    """Compare this run with registered version `version`; return True if it reproduces it."""
    original = client.get_model_version(MODEL_NAME, version)
    expected_checksum = original.tags["features_checksum"]
    if checksum != expected_checksum:
        print(f"FAIL: feature rows differ (checksum {checksum} != {expected_checksum})")
        return False
    expected = client.get_run(original.run_id).data.metrics["test_macro_f1"]
    diff = abs(metrics["test_macro_f1"] - expected)
    verdict = "PASS" if diff <= REPRODUCE_TOLERANCE else "FAIL"
    print(
        f"{verdict}: test_macro_f1 {metrics['test_macro_f1']:.4f} vs version {version} "
        f"{expected:.4f} (|diff| {diff:.4f}, tolerance {REPRODUCE_TOLERANCE})"
    )
    return diff <= REPRODUCE_TOLERANCE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true", help="register the model in MLflow")
    parser.add_argument("--reproduces", type=int, help="check this run against model version N")
    args = parser.parse_args()

    commit, dirty = git_state()
    if args.register and dirty:
        sys.exit("Refusing to register from uncommitted code: commit first so the tag is true.")

    features = load_features()
    train, valid, test = time_split(features)
    checksum = features_checksum(features)
    unseen = sorted(set(test["category"]) - set(train["category"]))
    print(f"rows: train={len(train)} valid={len(valid)} test={len(test)}")
    print(f"test categories never seen in training: {unseen}")

    lineage = {
        "data_version": data_version(),
        "git_commit": commit,
        "git_dirty": str(dirty).lower(),
        "feature_version": FEATURE_VERSION,
        "features_checksum": checksum,
    }

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.set_tags(lineage)
        if args.reproduces:
            mlflow.set_tag("reproduces_version", str(args.reproduces))
        mlflow.log_params(
            {**MODEL_PARAMS, "train_end": TRAIN_END, "valid_end": VALID_END, "test_end": TEST_END}
        )
        model, metrics, majority = evaluate(train, valid, test)
        mlflow.log_metrics(metrics)
        print(f"model trained in {metrics['train_seconds']:.0f} s")
        print_table(metrics, majority)

        client = MlflowClient()
        if args.register:
            info = mlflow.sklearn.log_model(model, name="model", registered_model_name=MODEL_NAME)
            version = info.registered_model_version
            for key, value in lineage.items():
                client.set_model_version_tag(MODEL_NAME, version, key, value)
            print(f"\nregistered {MODEL_NAME} version {version} (run {run.info.run_id})")
            print(f"promote it with: make promote VERSION={version}")

    if args.reproduces and not check_reproduction(client, args.reproduces, checksum, metrics):
        sys.exit(1)


if __name__ == "__main__":
    main()

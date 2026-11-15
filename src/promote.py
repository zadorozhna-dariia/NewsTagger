"""Point the champion alias at a registered model version (manual promotion, design doc 5.5)."""

import sys

from mlflow import MlflowClient

from train import MODEL_NAME

ALIAS = "champion"


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: make promote VERSION=<n>")
    version = sys.argv[1]
    client = MlflowClient()
    tags = client.get_model_version(MODEL_NAME, version).tags
    client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
    print(
        f"{MODEL_NAME}@{ALIAS} -> version {version} "
        f"(data {tags.get('data_version')}, commit {tags.get('git_commit')})"
    )


if __name__ == "__main__":
    main()

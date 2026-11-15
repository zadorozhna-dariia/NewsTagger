"""Reproduce a registered model version from its lineage tags with one command.

Checks out the version's git commit into a temporary worktree, pulls the raw data at that
commit with DVC and verifies its hash, then retrains there with --reproduces, which compares
the feature checksum and test macro-F1 with the registered version.
"""

import hashlib
import subprocess
import sys

from mlflow import MlflowClient

from train import MODEL_NAME, REPO_ROOT

WORKTREES = REPO_ROOT / ".reproduce"
RAW_PATH = "data/raw/News_Category_Dataset_v3.json"


def md5_of(path):
    digest = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(*cmd, cwd=REPO_ROOT):
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: make reproduce VERSION=<n>")
    version = sys.argv[1]
    tags = MlflowClient().get_model_version(MODEL_NAME, version).tags
    commit, data_hash = tags["git_commit"], tags["data_version"]
    print(f"version {version}: commit {commit[:7]}, data {data_hash}, {tags['feature_version']}")

    worktree = WORKTREES / commit[:7]
    if worktree.exists():
        run("git", "worktree", "remove", "--force", str(worktree))
    run("git", "worktree", "add", "--detach", str(worktree), commit)
    try:
        run("uv", "run", "dvc", "pull", "-q", RAW_PATH, cwd=worktree)
        pulled = md5_of(worktree / RAW_PATH)
        if pulled != data_hash:
            sys.exit(f"FAIL: pulled data {pulled} does not match data_version {data_hash}")
        print(f"data verified: {pulled}")
        run("uv", "run", "python", "src/train.py", "--reproduces", version, cwd=worktree)
    finally:
        run("git", "worktree", "remove", "--force", str(worktree))


if __name__ == "__main__":
    main()

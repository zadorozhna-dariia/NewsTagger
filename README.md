# NewsTagger: news category classification that keeps learning

A news publisher needs a section category for every new article, and editors choosing it by hand is slow and inconsistent.
NewsTagger suggests the top-3 categories from the headline and short description when the article is submitted.
Editors' final choices come back as labels, and the model is retrained monthly as topics drift.
The system watches its own inputs, predictions and override rate, and alerts a human when quality drops.
Data: [HuffPost News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) (R. Misra, CC BY 4.0), ~210k articles 2012–2022, replayed month by month.

## Quick start

```bash
make setup   # installs the pinned environment (uv) and the pre-commit hook
make lint    # runs ruff
```

Design document: [docs/design.md](docs/design.md)

## Data

Raw data: HuffPost News Category Dataset v3 (87 MB, not sampled), versioned with DVC 3.67.1. Remote: MinIO (S3-compatible) in Docker, bucket `dvc`.

```bash
cp .env.example .env
docker compose up -d
make setup
make data-pull
```

Expected file: `data/raw/News_Category_Dataset_v3.json`, md5 `bcd4791916daeac3fa5dc76786dd8bd5`.

## Maturity scorecard

| Property | Evidence a reviewer checks | Added in |
|---|---|---|
| Repository reproducible from a clean clone | CI is green; `make setup` and `make lint` succeed on a fresh clone | Lab 1 |
| Design document exists and is current | `docs/design.md` follows the template, every section filled or marked as an assumption | Lab 1 |

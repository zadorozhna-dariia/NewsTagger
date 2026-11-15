# Design document: NewsTagger

## 1 Overview

A news publisher publishes about 100 articles a day, and editors choose each article's section category by hand, which is slow and inconsistent. The set of topics keeps changing as new stories appear, so fixed rules go out of date. NewsTagger suggests the three most likely categories for each article, from its headline and short description, at the moment it is submitted. It learns from editors' corrections and is retrained every month. Editors accept the top suggestion for most articles, and the system notices and reports when its quality drops instead of failing silently.

**Assumption:** the publisher is hypothetical; the HuffPost News Category Dataset (2012–2022) stands in for its article history and is replayed month by month as the incoming feed. The volume of ~100 articles/day is an assumption used for sizing throughout this document.

## 2 Motivation

A one-off notebook could classify the 2012–2018 archive once, but it would be out of date with the next news cycle: in 2020, COVID coverage reshaped the health, wellness and politics sections within weeks. NewsTagger needs to be a running system because:

- **Who consumes the predictions:** the publisher's CMS, which requests a category for every submitted article, and the editors, who see the top-3 suggestions in the article form and confirm or correct them.
- **How often:** for every article at the moment it is submitted, about 100 times a day, with bursts during breaking news. The editor is waiting, so a suggestion must arrive within about a second. Editors' final choices come back as labels the same day.
- **What happens when they are wrong:** the article appears in the wrong section and reaches fewer of its intended readers until an editor notices, and each correction costs about a minute of editor time. A single error is cheap, but many errors add up, so the system must detect when its error rate rises rather than fail silently.

**Assumption:** a correction costs about one minute of editor time; the CMS waits at most one second for a suggestion before showing the form without one.

## 3 Success metrics

| Layer | Metric | Target / assumption | Type |
|---|---|---|---|
| Business | **Editor override rate**: share of articles whose final category differs from NewsTagger's top-1 suggestion, measured weekly | ≤ 25 % in the first three months, ≤ 15 % after | Optimizing |
| Model | **Macro-F1** on the time-held-out test period; **top-3 accuracy** as secondary | Macro-F1 ≥ trivial baseline + 0.10; top-3 accuracy ≥ 0.85 | Optimizing |
| System | **p95 latency** of a prediction request; **model freshness** (age of the newest data the serving model was trained on) | p95 ≤ 300 ms; freshness ≤ 35 days (monthly retraining plus slack) | Satisficing |

## 4 Requirements and constraints

### 4.1 Functional

| # | Requirement | Caller | Input | Output |
|---|---|---|---|---|
| FR1 | Predict category | CMS, on article submit | article_id, headline, short_description, authors, submitted_at | top-3 categories with probabilities, model_version |
| FR2 | Record label | CMS, when editor publishes | article_id, final_category, decided_at | stored label |
| FR3 | Batch scoring | Scheduler, nightly | articles not scored online | predictions in the database |
| FR4 | Retrain | Scheduler, monthly | new month of labeled articles | candidate model compared with current model |
| FR5 | Monitor | Scheduler, daily | predictions, inputs, labels | drift and quality report; alert to ML engineer |
| FR6 | Second opinion | Editor, from article form | article_id | Jev's category with calibrated confidence, using similar past articles as context |

### 4.2 Non-functional

| Category | Requirement |
|---|---|
| Freshness | Predictions made at submission time; serving model trained on data ≤ 35 days old (monthly retraining) |
| Latency and throughput | p95 ≤ 300 ms per request; sustained ~0.1 requests/s, peak 1 request/s |
| Availability and fallback | **Assumption:** 99 % during 06:00–24:00. If the current model fails to load, the service rolls back to the previous registered version. If the service is down or slower than 1 s, the CMS shows no suggestion and the editor chooses manually; publishing is never blocked. If top-1 confidence < 0.3, no suggestion is shown. Missed articles are scored by the nightly batch job |
| Reproducibility | Any registered model can be rebuilt from its git commit, data version and parameters with one command |
| Adaptability | New data enters monthly; a new category is added to the category list within 1 week and included in the next retraining |
| Observability without ground truth | Before labels arrive (≤ 24 h): prediction distribution drift, mean prediction confidence, share of unknown words. After labels: daily override rate, weekly macro-F1 |
| Cost and resource ceilings | Local: one laptop, CPU only, ≤ 8 GB RAM. Cloud estimate ≤ $800/month (see 6.7) |

### 4.3 Scope

**In scope:** one category per article for English-language articles, from headline and short description; real-time suggestions plus nightly batch scoring; monthly retraining; drift and quality monitoring with alerts; zero-shot second opinion from Jev; an agent that can look up and correct predictions.

**Out of scope:** full article text (not in the dataset); several categories per article; languages other than English; reader personalization and recommendations; integration with a real CMS (simulated by replaying the dataset).

## 5 Methodology
### 5.1 Problem statement

Multi-class, single-label text classification. Input: headline and short description, plus author and submission date. Output: a probability for each category; the top 3 are shown to the editor. **Assumption:** about 42 categories in the raw data, fewer after merging renamed duplicates.

**Rejected framing: multi-label classification.** Many articles fit more than one section, but the dataset has exactly one label per article, so there is no ground truth for multiple labels, and the CMS needs one primary section. Top-3 suggestions already show the editor the next most likely categories.

### 5.2 Data

**Source:** News Category Dataset v3 (R. Misra, Kaggle `rmisra/news-category-dataset`, CC BY 4.0): 209,527 HuffPost articles, 2012-01 → 2022-09, 87 MB JSON lines, not sampled. Fields: `category, headline, short_description, authors, date, link`.

**Data version in use:** DVC md5 `bcd4791916daeac3fa5dc76786dd8bd5`, remote `s3://dvc` on MinIO.

**Batch grain:** one calendar month, `[data_interval_start, data_interval_end)` in UTC, taken from the Airflow run (monthly `CronDataIntervalTimetable`); never from the current time.

**Arrival pattern:** the dataset is replayed as monthly batches, emulating a live feed. Rows per year: ~30,000–34,500 in 2012–2017, 9,734 in 2018, ~1,400–2,100 in 2019–2022.

**Labels and delay:** the label is the editor's final category. In the dataset it is present immediately; in the system it arrives ≤ 24 h after the prediction.

**Layers:** raw JSON (DVC) → monthly staging Parquet (`raw.parquet`, `clean.parquet`) → PostgreSQL `clean_articles` → PostgreSQL `features`.

**Cleaning (whole dataset):** 6 rows with an empty headline dropped; 112 exact duplicates (headline, description, date) dropped; 19,712 empty descriptions kept as empty text; 37,418 empty authors set to `unknown`; `link` dropped. Renamed sections merged via `dags/category_map.yaml` (26,712 rows, 42 → 33 categories).

| Field | Training | Serving |
|---|---|---|
| headline, short_description | yes | yes, in request |
| authors | yes | yes, in request; new authors unseen |
| date | yes | yes, as `submitted_at` |
| link | yes | **excluded**: created after the category is chosen and may contain the section name (label leakage) |
| category | label | arrives ≤ 24 h later |

**Data checklist:**
- **Time axis:** `date` on every row, so monthly batches come directly from the data and none have to be manufactured.
- **Holdout period:** train 2012-01 → 2016-12 (162,271 rows), validate 2017 (29,882), test 2018-01 → 2018-05 (8,606). Later months are sparse and are kept as the future stream for drift replay.
- **Label delay:** immediate in the dataset; assumed ≤ 24 h in the system (editor decision).
- **Text corpus:** the articles themselves (headline and description) serve as the retrieval corpus for the Jev comparison.

### 5.3 Techniques

1. **Rule baselines:** the majority class of the training window (WELLNESS), and an author rule: the author's most frequent past category, else WELLNESS.
2. **Candidate:** TF-IDF on `text` (word 1–2-grams, `min_df=2`, at most 200,000 terms, sublinear tf) followed by logistic regression (`C=4`, class weights for imbalance), saved as one scikit-learn pipeline.
3. **Later:** Jev, a zero-shot classifier given retrieved similar articles as context, compared against the candidate.

Table `features`, one row per article, `feature_version = v1`:

| Feature | Computed from | Serving-time source | Used by |
|---|---|---|---|
| `text` | headline + " " + short_description | request body | candidate |
| `has_description` | short_description is non-empty | request body | not yet |
| `day_of_week`, `month` | `date` | `submitted_at` in the request | not yet |
| `author_top_category` | the author's most frequent category in months before the article's month (windowed aggregate built by the pipeline) | lookup in PostgreSQL by author, as of the previous month; new author → `unknown` | author rule |
| `author_n_articles` | the author's article count in months before the article's month | same lookup; new author → 0 | not yet |

`author_top_category` is the feature whose training and serving sources differ: in training it is computed per month from history before that month (point-in-time correct); at serving it is read from the lookup as of the previous month.

### 5.4 Experimentation and validation

**Split:** by date. Train 2012-01 → 2016-12 (162,271 rows), validation 2017 (29,882), test 2018-01 → 2018-05 (8,606, reported once). Model settings are fixed; validation is reported, not used for tuning. Author features use only months before each article's month, so no split sees the future.

**Why not a random split:** articles about the same story would land on both sides and overstate the score, and it would hide drift that the time split shows: WELLNESS, the majority class of 2012–2016, is 0.2 % of 2018 articles, and `U.S. NEWS` first appears in May 2018 (11 test articles no model trained on the past can predict).

**Offline metric:** macro-F1 (primary), top-3 accuracy (secondary).

**New vs current model (champion / challenger):** each month the challenger is trained on data up to month M−1, and both models are scored on month M. The challenger is promoted only if its macro-F1 is ≥ champion + 0.01 and no category with ≥ 50 test articles loses more than 0.05 recall. Mechanics: `make register` adds a version with lineage tags; the ML engineer reviews it and runs `make promote VERSION=n`; `make reproduce VERSION=n` rebuilds any version from its commit and data and must match its test macro-F1 within 0.005.

### 5.5 Human in the loop

- **Drift alert:** reviewed by the ML engineer on call within 1 working day; decision is one of ignore, retrain early, or roll back.
- **Model promotion:** manual. The pipeline registers a challenger with its comparison report; the ML engineer reviews it and moves the `champion` alias. Rollback is moving the alias back.
- **Override:** editors choosing a different category than suggested; every override is stored as a label. The CMS can also switch suggestions off entirely with a feature flag.


## 6 Implementation
### 6.1 High-level design

#### System context

![System context](diagrams/system-context.svg)

#### Data flow

![Data flow](diagrams/data-flow.svg)

#### Data layers

![Data layers](diagrams/data-layers.svg)

#### Pipeline task graph

![Pipeline task graph](diagrams/task-graph.svg)

#### Tracking server and registry

![Tracking server and registry](diagrams/tracking-registry.svg)

#### Deployment (first draft)

![Deployment](diagrams/deployment.svg)

### 6.2 Infra

Local stack runs with `docker compose` (`docker-compose.yml`); secrets come from `.env` (template `.env.example`).

| Component | Role in NewsTagger | Local (pinned) | GCP | Status |
|---|---|---|---|---|
| Object storage | DVC remote (bucket `dvc`), MLflow artifacts (bucket `mlflow`) | MinIO-compatible `pgsty/silo:RELEASE.2026-09-16T00-00-00Z`, ports 9000/9001 | Cloud Storage | running |
| Data versioning | raw dataset, md5 `bcd4791…` | DVC 3.67.1 + dvc-s3 3.3.0, remote `s3://dvc` | GCS remote | running |
| Database | `newstagger` (clean_articles, features, author_stats, Feast online schema), `airflow`, `mlflow` | PostgreSQL 16, host port 5433 | Cloud SQL | running |
| Orchestrator | DAG `newstagger_etl`: monthly batches, backfills | Airflow 3.3.1, LocalExecutor (api-server, scheduler, dag-processor), UI on 8080 | Managed Service for Apache Airflow (formerly Cloud Composer) | running |
| Experiment tracking and registry | runs with lineage tags; `newstagger-classifier@champion` | MLflow 3.16.1 server (official image + psycopg2, boto3), Postgres backend, MinIO artifacts served through the server, UI on 5001 | Vertex AI Experiments and Model Registry | running |
| Feature store | entity `author`, feature view `author_stats` | Feast 0.66.0: offline store = Postgres `author_stats`, online store = Postgres schema `feast_online` | Vertex AI Feature Store | running |
| Container registry | images of the prediction service | Minikube built-in registry | Artifact Registry | planned |
| Cluster | runs the prediction service | Minikube 1.33.1 | GKE | planned |
| Model monitoring | drift reports | Evidently | Vertex AI Model Monitoring | planned |
| Metrics and dashboards | latency, errors, alerts | Prometheus, Grafana | Cloud Monitoring | planned |
| Vector store | similar past articles for Jev | pgvector 0.8.7 | Vertex AI Vector Search | planned |
| Zero-shot classifier | second opinion | Jev API (TypeSafe AI) | none: third-party API, called the same from any cloud | planned |

### 6.3 Performance

**Decision: online inference, with a nightly batch job as a helper.** The editor is waiting in the article form, so the suggestion must be returned at submission time; a batch-only design would deliver it up to 24 h late. The nightly batch scores articles that missed the online call and builds the daily monitoring table.

| Budget | Value |
|---|---|
| p95 latency per request | ≤ 300 ms: ~20 ms feature lookup, ~10 ms model, rest network and serialization |
| Throughput | ~0.1 requests/s sustained, 1 request/s peak; one replica with 0.5 CPU and 512 MB RAM |
| Jev second opinion | asynchronous (70–500 ms per call), excluded from the latency budget |

### 6.4 Security

**Secrets:** database, MinIO and Airflow secrets live in a git-ignored `.env`; `.env.example` holds placeholders only. Compose passes them to containers as environment variables; Airflow reaches Postgres through `AIRFLOW_CONN_NEWSTAGGER_DB`, so no password appears in DAG code. Airflow's JWT secret and `[api] secret_key` are random values shared by all Airflow containers. On Minikube secrets become Kubernetes Secrets, in the cloud Secret Manager.

**Access:** all published ports are bound to `127.0.0.1`, so the Airflow UI (no login, all users admin), MLflow (no login), MinIO and Postgres are reachable only from the local machine. Local Postgres connections are unencrypted (`sslmode: disable`); in the cloud TLS is required. The prediction service will be reachable only inside the cluster and require an API key from the CMS.

**Model artifacts:** MLflow stores the model in the skops format, which does not execute code when loaded, unlike pickle; only versions registered from committed code can be promoted.

### 6.5 Data privacy

The dataset contains public headlines and descriptions, and author names, which are personal data under GDPR even though they are public bylines. Author names are kept in the raw file (local and DVC remote), the monthly staging files, `clean_articles`, `author_stats` and the Feast online store, where they are the lookup key; they are used only for the aggregated author features. Pipeline logs contain row counts only. The model is trained on `text` alone; its vocabulary can include names of people mentioned in headlines. Headlines and descriptions are sent to the Jev API (third party); author names are not. No reader data is collected. Data is kept until the end of the course (January 2027), then deleted with `docker compose down -v` (Postgres, MinIO, Airflow volumes) and by removing `data/`. The dataset is licensed CC BY 4.0; attribution is given in the README.

### 6.6 Monitoring and alarms

Logged per request: article_id, timestamp, model_version, top-3 categories with probabilities, latency. Inputs are kept for 90 days.

| Signal | Checked | Alert when | Alert to |
|---|---|---|---|
| Data freshness | each pipeline run | newest batch older than expected + 2 days, or identical to the previous batch | ML engineer |
| Quality gate | each ingest | any check fails | ML engineer |
| Prediction drift (PSI vs training) | daily | PSI > 0.2 | ML engineer |
| Mean top-1 confidence | daily | drops > 0.10 below 30-day mean | ML engineer |
| Unknown-word rate | daily | > 1.5× training rate | ML engineer |
| Override rate | daily, weekly average | weekly > 25 % | ML engineer, editor lead |
| Macro-F1 on last labeled week | weekly | drops > 0.05 below test score | ML engineer |
| p95 latency, error rate | continuous | p95 > 300 ms or errors > 1 % for 10 min | ML engineer |

Alerts are sent by email.

### 6.7 Cost

Estimated with the [Google Cloud pricing calculator](https://cloud.google.com/products/calculator?dl=CjhDaVE1T1RjeVl6QTBNQzFqWm1VMUxUUm1OVEV0WVdKaU1DMWpOemsxWW1ObU4yTmxOamtRQVE9PRApGiQwQTIyQjEyQS1GNjQwLTQ2NzgtQjkzMy02NkE1Q0Q2MDJGRTA) on 2026-10-19, region europe-west1, on-demand prices, 730 hours/month.

| Service | Sizing | $/month |
|---|---|---|
| Managed Service for Apache Airflow | Gen 3, small environment | 379.36 |
| Agent Platform Feature Store | 1 optimized online serving node; 3 node-hours ingestion | 219.24 |
| Agent Platform Vector Search | 1 × e2-standard-2 node; 210k vectors × 384 dims, rebuilt monthly | 76.24 |
| GKE | 1 zonal cluster, 2 × e2-medium, 30 GB standard disks | 128.01 |
| Cloud SQL for PostgreSQL | micro instance, 20 GB | 11.07 |
| Jev API (TypeSafe AI) | 3,000 requests/month; **Assumption:** price not published, to be measured | — |
| Artifact Registry | 5 GB | 0.45 |
| Cloud Storage | 10 GB standard | 0.10 |
| Cloud Monitoring | 10 custom metrics | 0.00 |
| Secret Manager | 5 secrets, 10k accesses | 0.00 |
| **Total (calculator)** | | **814.47** |
| **Total after GKE free-tier credit** (−$73 zonal cluster fee) | | **≈ 742** |

**Assumptions:** 100 articles/day; all services always on; no committed-use discounts.

Four always-on services (Airflow, Feature Store, Vector Search, GKE cluster fee) are ~92 % of the cost. First saving: replace the managed Feature Store with Feast's online store in the existing Cloud SQL instance (−$219/month).

### 6.8 Integration points

| Party | Direction | Contract |
|---|---|---|
| CMS article feed (replayed from the DVC-tracked dataset) | upstream: articles | JSON lines with `headline, short_description, authors, category, date`; a month's batch is complete before its run starts (runs start after the month ends). Enforced by `validate`: required columns present, every `date` inside the run's month, empty-headline rate ≤ 1 %; a failing batch stops the run before anything is loaded. Schema changes announced ≥ 2 weeks ahead. |
| Editors (via CMS) | upstream: labels | `article_id, final_category, decided_at` within 24 h; category must be in the current category list; renamed sections are mapped by `dags/category_map.yaml`. |
| Prediction service | downstream: model and features | loads `models:/newstagger-classifier@champion` from MLflow; reads `author_top_category` and `author_n_articles` from the Feast online store by `authors`, falling back to `unknown` / 0. |
| CMS | downstream: predictions | `POST /predict` returns top-3 and `model_version` within 1 s; no answer means no suggestion. |
| ML engineer | downstream: alerts | a failed `validate` or `quality_check` task in Airflow names every failed check; email with signal, value and threshold is planned with monitoring. |

### 6.9 Risks and uncertainties

| Risk | Mitigation |
|---|---|
| Post-2018 data too sparse to show drift | count rows per month; if too few, replay 2017–2018 months and inject an artificial shift |
| Renamed or merged categories | versioned category mapping; quality gate rejects unknown categories |
| Label leakage via `link` or same-story duplicates | `link` dropped in preprocessing; time-based split |
| Inconsistent labels between similar sections | review the confusion matrix; merge consistently confused sections |
| Headline-only articles give too little text | top-3 suggestions; comparison with Jev |
| Laptop resources too small for the full stack | Minikube capped at 4 GB; Airflow LocalExecutor; start services on demand |
| Cloud cost above estimate | replace the managed feature store with Cloud SQL first (−$219/month) |


## 7 Appendix
### 7.1 Alternatives

| Alternative | Pros | Cons | Why rejected |
|---|---|---|---|
| AG News dataset instead of HuffPost | large, clean, balanced | no timestamps, only 4 classes | no time axis: no monthly batches, no time holdout, no drift |
| Fine-tuned DistilBERT instead of TF-IDF + logistic regression | likely higher macro-F1 | slow CPU training and inference, harder to retrain monthly on a laptop | the simple model meets the latency budget and is easy to rebuild |
| Streaming ingestion (Kafka + Spark) | realistic for large publishers | heavy local footprint | ~0.1 requests/s needs no streaming; batch + online API meets freshness |
| AWS instead of GCP | most widely used cloud | higher estimate for the same managed services | GCP estimate fits the cost ceiling |
| Generative LLM for text explanations | readable reasons for editors | slower, costlier, free text is hard to evaluate | Jev returns typed, calibrated outputs that can be scored like the model |

### 7.2 Experiment results

From MLflow experiment `newstagger` (time split of 5.4; test = 2018-01 → 2018-05, 8,606 articles):

| Model | Valid macro-F1 | Test macro-F1 | Test accuracy | Test top-3 |
|---|---|---|---|---|
| Majority class (WELLNESS) | 0.003 | 0.000 | 0.002 | n/a |
| Author rule | 0.366 | 0.248 | 0.494 | n/a |
| TF-IDF + logistic regression | 0.388 | 0.324 | 0.616 | 0.860 |

**Registered:** `newstagger-classifier` version 2 = `@champion`, run `852cbbb2`, tags `data_version=bcd4791…`, `git_commit=7c08e96`, `feature_version=v1`, `features_checksum=a12d6111…`.

**Reproduced:** `make reproduce VERSION=2` (run `e35521e7`, tag `reproduces_version=2`) gave test macro-F1 0.3238 vs 0.3238, |diff| 0.0000 within the 0.005 tolerance.

**Against the targets (section 3):** macro-F1 ≥ trivial baseline + 0.10 met (0.324 vs 0.000); top-3 accuracy ≥ 0.85 met (0.860). The author rule, using no text, comes within 0.022 of the model on validation; on test the model leads by 0.076. 11 test articles are `U.S. NEWS`, a category absent from training, and are always misclassified.

### 7.3 Performance benchmarks

To be filled in Lab 4: measured p50/p95 latency and throughput of the prediction service on Minikube.

### 7.4 Milestones

- **Lab 2:** monthly ingestion with DVC versioning and the quality gate; baseline and TF-IDF + logistic regression tracked and registered in MLflow.
- **Lab 3:** Jev as a zero-shot baseline with retrieved similar articles, compared with the Lab 2 model on macro-F1, cost and latency.
- **Lab 4:** prediction service and batch job containerized and deployed on Minikube with health checks, plus scheduled drift monitoring and alerts.
- **Lab 5:** an agent that lets editors look up a prediction and Jev's second opinion, find similar articles and submit a category correction.

from pathlib import Path

import pandas as pd
import pendulum
import yaml
from airflow.exceptions import AirflowFailException
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.sdk import Param, dag, task
from airflow.timetables.interval import CronDataIntervalTimetable

RAW_FILE = Path("/opt/airflow/data/raw/News_Category_Dataset_v3.json")
STAGING_DIR = Path("/opt/airflow/data/staging")
DB_CONN_ID = "newstagger_db"
READ_CHUNK_ROWS = 20_000
CATEGORY_MAP_FILE = Path("/opt/airflow/dags/category_map.yaml")
EXPECTED_COLUMNS = {"headline", "short_description", "authors", "category", "date"}
MAX_EMPTY_HEADLINE_RATE = 0.01


def interval_bounds(data_interval_start, data_interval_end):
    """Run interval as naive dates, comparable with the dataset's date column."""
    return (
        pd.Timestamp(data_interval_start.strftime("%Y-%m-%d")),
        pd.Timestamp(data_interval_end.strftime("%Y-%m-%d")),
    )


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS clean_articles (
    headline          TEXT NOT NULL,
    short_description TEXT,
    authors           TEXT,
    category          TEXT NOT NULL,
    date              DATE NOT NULL,
    batch_month       DATE NOT NULL,
    loaded_at         TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

QUALITY_SQL = """
SELECT
    count(*) AS row_count,
    count(*) FILTER (WHERE headline = '' OR category = '') AS empty_required,
    count(*) - count(DISTINCT (headline, short_description, date)) AS duplicates,
    count(*) FILTER (WHERE category = ANY(%s)) AS old_categories
FROM clean_articles
WHERE batch_month = %s
"""

FEATURE_VERSION = "v1"

CREATE_FEATURES_SQL = """
CREATE TABLE IF NOT EXISTS features (
    article_id          TEXT NOT NULL,
    date                DATE NOT NULL,
    batch_month         DATE NOT NULL,
    text                TEXT NOT NULL,
    has_description     BOOLEAN NOT NULL,
    day_of_week         SMALLINT NOT NULL,
    month               SMALLINT NOT NULL,
    author_top_category TEXT NOT NULL,
    author_n_articles   INTEGER NOT NULL,
    category            TEXT NOT NULL,
    feature_version     TEXT NOT NULL,
    PRIMARY KEY (article_id, feature_version)
)
"""

# Author history uses only months loaded before this batch (point-in-time correct).
INSERT_FEATURES_SQL = """
WITH history AS (
    SELECT authors, category, count(*) AS n
    FROM clean_articles
    WHERE batch_month < %(batch_month)s AND authors <> 'unknown'
    GROUP BY authors, category
),
author_stats AS (
    SELECT authors,
           sum(n) AS n_articles,
           (array_agg(category ORDER BY n DESC, category))[1] AS top_category
    FROM history
    GROUP BY authors
)
INSERT INTO features (
    article_id, date, batch_month, text, has_description, day_of_week, month,
    author_top_category, author_n_articles, category, feature_version
)
SELECT
    md5(a.headline || '|' || a.short_description || '|' || a.date),
    a.date,
    a.batch_month,
    trim(a.headline || ' ' || a.short_description),
    a.short_description <> '',
    extract(isodow FROM a.date),
    extract(month FROM a.date),
    coalesce(s.top_category, 'unknown'),
    coalesce(s.n_articles, 0),
    a.category,
    %(feature_version)s
FROM clean_articles a
LEFT JOIN author_stats s USING (authors)
WHERE a.batch_month = %(batch_month)s
"""


@dag(
    schedule=CronDataIntervalTimetable("@monthly", timezone="UTC"),
    start_date=pendulum.datetime(2012, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["newstagger"],
    default_args={"retries": 1, "retry_delay": pendulum.duration(seconds=30)},
    params={
        "inject_fault": Param(
            "none",
            enum=["none", "empty_headlines"],
            description="Test hook: break this run's batch on purpose to prove the gates fail.",
        )
    },
)
def newstagger_etl():
    @task
    def extract(data_interval_start=None, data_interval_end=None, params=None):
        start, end = interval_bounds(data_interval_start, data_interval_end)

        # Read in chunks and keep only this interval, so memory stays small.
        with pd.read_json(RAW_FILE, lines=True, chunksize=READ_CHUNK_ROWS) as reader:
            parts = [chunk[(chunk["date"] >= start) & (chunk["date"] < end)] for chunk in reader]
        batch = pd.concat(parts, ignore_index=True)

        if params["inject_fault"] == "empty_headlines":
            broken = batch.index[::2]
            batch.loc[broken, "headline"] = ""
            print(f"Injected fault 'empty_headlines': blanked {len(broken)} headlines")

        out_dir = STAGING_DIR / start.strftime("%Y-%m")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "raw.parquet"
        batch.to_parquet(out_path, index=False)

        print(f"Extracted {len(batch)} rows for [{start.date()}, {end.date()}) -> {out_path}")
        return {"path": str(out_path), "rows": len(batch)}

    @task
    def validate(extracted, data_interval_start=None, data_interval_end=None):
        batch = pd.read_parquet(extracted["path"])
        problems = []

        missing = EXPECTED_COLUMNS - set(batch.columns)
        if missing:
            problems.append(f"missing columns: {sorted(missing)}")
        else:
            start, end = interval_bounds(data_interval_start, data_interval_end)
            outside = int(((batch["date"] < start) | (batch["date"] >= end)).sum())
            if outside:
                problems.append(f"{outside} rows dated outside [{start.date()}, {end.date()})")

            if len(batch):
                empty_rate = (batch["headline"].fillna("").str.strip() == "").mean()
                if empty_rate > MAX_EMPTY_HEADLINE_RATE:
                    limit = f"{MAX_EMPTY_HEADLINE_RATE:.0%}"
                    problems.append(f"empty headline rate {empty_rate:.1%} > {limit}")

        if problems:
            raise AirflowFailException("Validation failed: " + "; ".join(problems))

        print(f"Validation passed: {len(batch)} rows, schema ok, dates in interval")
        return extracted

    @task
    def transform(extracted):
        batch = pd.read_parquet(extracted["path"])
        rows_in = len(batch)

        batch = batch.drop(columns=["link"])
        for col in ["headline", "short_description", "authors", "category"]:
            batch[col] = batch[col].fillna("").str.strip()

        batch = batch[batch["headline"] != ""]
        empty_headlines = rows_in - len(batch)

        before = len(batch)
        batch = batch.drop_duplicates(subset=["headline", "short_description", "date"])
        duplicates = before - len(batch)

        unknown_authors = int((batch["authors"] == "").sum())
        batch["authors"] = batch["authors"].replace("", "unknown")

        category_map = yaml.safe_load(CATEGORY_MAP_FILE.read_text())
        remapped = int(batch["category"].isin(category_map).sum())
        batch["category"] = batch["category"].replace(category_map)

        out_path = Path(extracted["path"]).with_name("clean.parquet")
        batch.to_parquet(out_path, index=False)

        print(
            f"Transform: in={rows_in} empty_headline={empty_headlines} "
            f"duplicates={duplicates} unknown_authors={unknown_authors} "
            f"remapped={remapped} out={len(batch)}"
        )
        return {"path": str(out_path), "rows": len(batch)}

    @task
    def load(cleaned, data_interval_start=None):
        batch_month = data_interval_start.strftime("%Y-%m-%d")
        batch = pd.read_parquet(cleaned["path"])
        rows = [
            (r.headline, r.short_description, r.authors, r.category, r.date.date(), batch_month)
            for r in batch.itertuples()
        ]

        conn = PostgresHook(postgres_conn_id=DB_CONN_ID).get_conn()
        with conn, conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            cur.execute("DELETE FROM clean_articles WHERE batch_month = %s", (batch_month,))
            cur.executemany(
                "INSERT INTO clean_articles "
                "(headline, short_description, authors, category, date, batch_month) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                rows,
            )
            cur.execute(
                "SELECT count(*) FROM clean_articles WHERE batch_month = %s", (batch_month,)
            )
            count = cur.fetchone()[0]
        conn.close()

        print(f"Loaded {count} rows for batch_month={batch_month}")
        return count

    @task
    def build_features(loaded_rows, data_interval_start=None):
        batch_month = data_interval_start.strftime("%Y-%m-%d")
        args = {"batch_month": batch_month, "feature_version": FEATURE_VERSION}

        conn = PostgresHook(postgres_conn_id=DB_CONN_ID).get_conn()
        with conn, conn.cursor() as cur:
            cur.execute(CREATE_FEATURES_SQL)
            cur.execute(
                "DELETE FROM features "
                "WHERE batch_month = %(batch_month)s AND feature_version = %(feature_version)s",
                args,
            )
            cur.execute(INSERT_FEATURES_SQL, args)
            count = cur.rowcount
        conn.close()

        print(f"Built {count} feature rows ({FEATURE_VERSION}) for batch_month={batch_month}")
        return count

    @task
    def quality_check(cleaned, loaded_rows, feature_rows, data_interval_start=None):
        batch_month = data_interval_start.strftime("%Y-%m-%d")
        old_names = list(yaml.safe_load(CATEGORY_MAP_FILE.read_text()))

        conn = PostgresHook(postgres_conn_id=DB_CONN_ID).get_conn()
        with conn, conn.cursor() as cur:
            cur.execute(QUALITY_SQL, (old_names, batch_month))
            row_count, empty_required, duplicates, old_categories = cur.fetchone()
        conn.close()

        problems = []
        if row_count == 0:
            problems.append(f"empty batch: 0 rows for batch_month={batch_month}")
        if row_count != cleaned["rows"] or row_count != loaded_rows:
            problems.append(
                f"row count mismatch: table={row_count} transform={cleaned['rows']} "
                f"load={loaded_rows}"
            )
        if feature_rows != row_count:
            problems.append(f"feature rows {feature_rows} != article rows {row_count}")
        if empty_required:
            problems.append(f"{empty_required} rows with empty headline or category")
        if duplicates:
            problems.append(f"{duplicates} duplicate articles")
        if old_categories:
            problems.append(f"{old_categories} rows with an unmapped old category")

        if problems:
            raise AirflowFailException("Quality check failed: " + "; ".join(problems))

        print(f"Quality check passed: {row_count} rows for batch_month={batch_month}")

    cleaned = transform(validate(extract()))
    loaded = load(cleaned)
    quality_check(cleaned, loaded, build_features(loaded))


newstagger_etl()

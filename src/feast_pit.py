"""Feast demo for the author_stats feature view (lab 2 step 4, optional point).

Default: a point-in-time retrieval for a timestamp between two feature rows (must return the
earlier row), then a check that Feast's offline values equal the training features table.
With --online AUTHOR: read that author's latest stats from the online store (after materialize).
"""

import argparse
import sys

import pandas as pd
import psycopg
from feast import FeatureStore

from train import REPO_ROOT, db_url

FEATURES = ["author_stats:author_top_category", "author_stats:author_n_articles"]

TWO_ROWS_SQL = """
SELECT authors, event_timestamp, author_top_category, author_n_articles,
       next_ts, next_category, next_n
FROM (
    SELECT *,
           lead(event_timestamp) OVER w AS next_ts,
           lead(author_top_category) OVER w AS next_category,
           lead(author_n_articles) OVER w AS next_n
    FROM author_stats
    WINDOW w AS (PARTITION BY authors ORDER BY event_timestamp)
) s
WHERE next_category <> author_top_category AND next_ts - event_timestamp >= interval '2 months'
ORDER BY authors, event_timestamp
LIMIT 1
"""

TRAINING_ROWS_SQL = """
SELECT a.authors, f.batch_month, f.author_top_category, f.author_n_articles
FROM features f
JOIN clean_articles a
  ON md5(a.headline || '|' || a.short_description || '|' || a.date) = f.article_id
WHERE f.author_top_category <> 'unknown'
ORDER BY f.article_id
LIMIT %s
"""


def historical(store, entity_df):
    return store.get_historical_features(entity_df=entity_df, features=FEATURES).to_df()


def point_in_time_demo(store, cur):
    cur.execute(TWO_ROWS_SQL)
    author, t1, cat1, n1, t2, cat2, n2 = cur.fetchone()
    asked = t1 + (t2 - t1) / 2
    print(f"author {author!r} has feature rows:")
    print(f"  {t1:%Y-%m-%d}: author_top_category={cat1}, author_n_articles={n1}")
    print(f"  {t2:%Y-%m-%d}: author_top_category={cat2}, author_n_articles={n2}")

    got = historical(store, pd.DataFrame({"authors": [author], "event_timestamp": [asked]}))
    category, count = got.loc[0, "author_top_category"], int(got.loc[0, "author_n_articles"])
    print(f"Feast at {asked:%Y-%m-%d} (between the two rows): {category}, {count}")
    ok = (category, count) == (cat1, n1)
    print("PASS: returned the earlier row" if ok else "FAIL: did not return the earlier row")
    return ok


def consistency_check(store, cur, sample):
    cur.execute(TRAINING_ROWS_SQL, (sample,))
    rows = pd.DataFrame(
        cur.fetchall(), columns=["authors", "batch_month", "expected_category", "expected_n"]
    ).drop_duplicates()
    rows["event_timestamp"] = pd.to_datetime(rows["batch_month"]).dt.tz_localize("UTC")
    got = historical(store, rows[["authors", "event_timestamp"]])
    # Feast may hand timestamps back without a time zone; compare both sides as UTC.
    got["event_timestamp"] = pd.to_datetime(got["event_timestamp"], utc=True)
    merged = rows.merge(got, on=["authors", "event_timestamp"])
    mismatches = (merged["author_top_category"] != merged["expected_category"]) | (
        merged["author_n_articles"].astype(int) != merged["expected_n"]
    )
    print(
        f"Feast vs training features table: {len(merged)} (author, month) pairs, "
        f"{int(mismatches.sum())} mismatches"
    )
    return not mismatches.any()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", metavar="AUTHOR", help="read AUTHOR from the online store")
    parser.add_argument("--sample", type=int, default=2000, help="articles to compare")
    args = parser.parse_args()

    store = FeatureStore(repo_path=str(REPO_ROOT / "feature_repo"))
    if args.online:
        rows = [{"authors": args.online}]
        print(store.get_online_features(features=FEATURES, entity_rows=rows).to_dict())
        return

    with psycopg.connect(db_url()) as conn, conn.cursor() as cur:
        ok = point_in_time_demo(store, cur)
        ok = consistency_check(store, cur, args.sample) and ok
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

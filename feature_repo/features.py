"""Feast definitions: the author entity and the author_stats feature view (design doc 5.3)."""

from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Int64, String
from feast.value_type import ValueType

author = Entity(name="author", join_keys=["authors"], value_type=ValueType.STRING)

author_stats_source = PostgreSQLSource(
    name="author_stats_source",
    query=(
        "SELECT authors, event_timestamp, author_top_category, author_n_articles "
        "FROM author_stats"
    ),
    timestamp_field="event_timestamp",
)

author_stats = FeatureView(
    name="author_stats",
    entities=[author],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="author_top_category", dtype=String),
        Field(name="author_n_articles", dtype=Int64),
    ],
    source=author_stats_source,
    online=True,
)

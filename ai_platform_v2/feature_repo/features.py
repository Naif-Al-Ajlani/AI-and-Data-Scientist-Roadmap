from datetime import datetime, timedelta
import pandas as pd
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

user = Entity(name="user_id", join_keys=["user_id"])

source = FileSource(
    name="demo_source",
    path="feature_repo/data/demo.parquet",
    timestamp_field="event_timestamp",
)

demo_view = FeatureView(
    name="demo_stats",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name="mean_f", dtype=Float32),
        Field(name="std_f", dtype=Float32),
        Field(name="label", dtype=Int64),
    ],
    online=True,
    source=source,
)

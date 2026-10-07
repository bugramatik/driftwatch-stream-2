"""Rebuild reference profiles from historical Parquet feature events.

Run with spark-submit:

    spark-submit spark/build_profiles.py --input s3://driftwatch/events/ \
        --model churn-risk --version v7 --sample 20000 --out ./profiles
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from driftwatch.profiles import ProfileStore

DEFAULT_FEATURES = ["tenure_days", "sessions_7d", "support_tickets", "plan_tier"]


def build_profile(spark: SparkSession, path: str, features: list[str], sample: int) -> dict[str, list[float]]:
    """Read Parquet events, flatten the feature struct and return a bounded sample per feature."""
    frame: DataFrame = spark.read.parquet(path)
    if "features" not in frame.columns:
        raise ValueError(f"{path} has no features column")

    present = [name for name in features if name in frame.select("features.*").columns]
    if not present:
        raise ValueError("none of the requested features exist in the snapshot")

    flat = frame.select(*[F.col(f"features.{name}").cast("double").alias(name) for name in present])
    flat = flat.dropna()

    total = flat.count()
    if total == 0:
        raise ValueError("snapshot has no complete feature rows")

    fraction = min(1.0, sample / total)
    sampled = flat.sample(withReplacement=False, fraction=fraction, seed=17).limit(sample)

    profile: dict[str, list[float]] = {}
    for name in present:
        values = [row[0] for row in sampled.select(name).collect() if row[0] is not None]
        if values:
            profile[name] = sorted(float(v) for v in values)
    return profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build drift reference profiles from Parquet events")
    parser.add_argument("--input", required=True, help="Parquet path or glob of historical events")
    parser.add_argument("--model", required=True, help="model name the profile belongs to")
    parser.add_argument("--version", required=True, help="model version the profile belongs to")
    parser.add_argument("--sample", type=int, default=20000, help="values kept per feature")
    parser.add_argument("--out", default="./profiles", help="reference profile directory")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES), help="comma separated feature names")
    args = parser.parse_args(argv)

    if args.sample < 100:
        parser.error("--sample must be at least 100")

    features = [name.strip() for name in args.features.split(",") if name.strip()]
    spark = SparkSession.builder.appName("driftwatch-build-profiles").getOrCreate()
    try:
        profile = build_profile(spark, args.input, features, args.sample)
    finally:
        spark.stop()

    if not profile:
        print("no features collected, nothing written", file=sys.stderr)
        return 1

    store = ProfileStore(args.out)
    written: Path = store.save(args.model, args.version, profile)
    print(f"wrote {len(profile)} features to {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

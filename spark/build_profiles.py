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

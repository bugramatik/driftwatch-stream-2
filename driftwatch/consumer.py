"""Kafka consumer loop: fill rolling windows, evaluate drift, publish alerts."""

from __future__ import annotations

import json
import logging
import signal
import threading
import time
from typing import Any, Protocol

from confluent_kafka import Consumer, KafkaError, Producer

from driftwatch.config import Settings, load_settings
from driftwatch.drift import FeatureVerdict, evaluate_feature
from driftwatch.profiles import ProfileStore
from driftwatch.window import RollingWindow

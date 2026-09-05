"""Shared deterministic fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_fraud.config import (
    AppConfig,
    DataConfig,
    ModelConfig,
    OutputConfig,
    SplitConfig,
)
from credit_fraud.data import FraudDataset


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        data=DataConfig(tmp_path / "creditcard.csv", "Class", "Amount", "Time"),
        split=SplitConfig(0.25, 123),
        model=ModelConfig("smote", 3, 2, (0.1, 1.0), 200, 0.5),
        output=OutputConfig(
            tmp_path / "outputs", "predictions.csv", "metrics.json", "model.skops"
        ),
    )


@pytest.fixture
def fraud_dataset() -> FraudDataset:
    rng = np.random.default_rng(123)
    normal_count = 90
    fraud_count = 30
    labels = np.concatenate(
        (np.zeros(normal_count, dtype=np.int64), np.ones(fraud_count, dtype=np.int64))
    )
    signal = labels * 2.5 + rng.normal(0, 0.7, len(labels))
    features = pd.DataFrame(
        {
            "Amount": np.abs(signal * 25 + 50),
            "Time": np.arange(len(labels), dtype=np.float64),
            "V1": signal,
            "V2": rng.normal(size=len(labels)),
        }
    )
    return FraudDataset(
        features=features,
        labels=pd.Series(labels, name="Class"),
        row_ids=pd.Series(features.index.astype(str), name="row_id"),
    )

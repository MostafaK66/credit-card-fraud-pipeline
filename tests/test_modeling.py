"""Model training and evaluation tests."""

from __future__ import annotations

import pandas as pd
import pytest

from credit_fraud.config import ModelConfig, SplitConfig
from credit_fraud.data import FraudDataset
from credit_fraud.errors import TrainingError
from credit_fraud.modeling import train_and_evaluate


@pytest.mark.parametrize("sampling", ["smote", "class_weight"])
def test_trains_deterministically_with_both_imbalance_strategies(
    fraud_dataset: FraudDataset, sampling: str
) -> None:
    config = ModelConfig(sampling, 3, 2, (0.1, 1.0), 200, 0.5)
    first = train_and_evaluate(fraud_dataset, SplitConfig(0.25, 123), config)
    second = train_and_evaluate(fraud_dataset, SplitConfig(0.25, 123), config)
    pd.testing.assert_frame_equal(first.predictions, second.predictions)
    assert 0.0 <= first.metrics["average_precision"] <= 1.0
    assert 0.0 <= first.metrics["roc_auc"] <= 1.0
    assert first.metrics["test_transactions"] == 30
    assert set(first.best_parameters) == {"C"}


def test_rejects_too_few_minority_samples(fraud_dataset: FraudDataset) -> None:
    small = FraudDataset(
        fraud_dataset.features.iloc[:94],
        fraud_dataset.labels.iloc[:94],
        fraud_dataset.row_ids.iloc[:94],
    )
    config = ModelConfig("smote", 3, 2, (1.0,), 100, 0.5)
    with pytest.raises(TrainingError, match="minority"):
        train_and_evaluate(small, SplitConfig(0.2, 1), config)


def test_wraps_impossible_stratified_split(fraud_dataset: FraudDataset) -> None:
    tiny = FraudDataset(
        fraud_dataset.features.iloc[[0, 1, 90]],
        fraud_dataset.labels.iloc[[0, 1, 90]],
        fraud_dataset.row_ids.iloc[[0, 1, 90]],
    )
    config = ModelConfig("class_weight", 2, 1, (1.0,), 100, 0.5)
    with pytest.raises(TrainingError, match="stratified holdout"):
        train_and_evaluate(tiny, SplitConfig(0.2, 1), config)

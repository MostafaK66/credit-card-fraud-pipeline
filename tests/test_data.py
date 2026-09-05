"""Dataset loader tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from credit_fraud.config import DataConfig
from credit_fraud.data import load_dataset
from credit_fraud.errors import DataValidationError


def config() -> DataConfig:
    return DataConfig(Path("unused.csv"), "Class", "Amount", "Time")


def valid_records() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": [0, 1, 2],
            "Amount": [10.0, 20.0, 30.0],
            "V1": [0.1, 0.2, 0.3],
            "Class": [0, 0, 1],
        },
        index=[10, 11, 12],
    )


def test_loads_numeric_binary_dataset() -> None:
    result = load_dataset(config(), reader=lambda _: valid_records())
    assert result.features.columns.tolist() == ["Time", "Amount", "V1"]
    assert result.labels.tolist() == [0, 0, 1]
    assert result.row_ids.tolist() == ["10", "11", "12"]


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (pd.DataFrame(), "missing columns"),
        (pd.DataFrame(columns=["Class", "Amount", "Time"]), "no transactions"),
        (valid_records().assign(V1="bad"), "numeric and non-null"),
        (valid_records().assign(V1=float("inf")), "non-finite"),
        (valid_records().assign(Class=0), "both binary classes"),
        (valid_records().assign(Class=[0, 1, 1.5]), "both binary classes"),
    ],
)
def test_rejects_invalid_data(records: pd.DataFrame, message: str) -> None:
    with pytest.raises(DataValidationError, match=message):
        load_dataset(config(), reader=lambda _: records)


def test_wraps_reader_failure() -> None:
    def fail(_: Path) -> pd.DataFrame:
        raise OSError("missing")

    with pytest.raises(DataValidationError, match="Cannot read dataset"):
        load_dataset(config(), reader=fail)

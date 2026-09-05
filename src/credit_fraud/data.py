"""Input loading and fraud-dataset validation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from credit_fraud.config import DataConfig
from credit_fraud.errors import DataValidationError

CsvReader = Callable[..., pd.DataFrame]


@dataclass(frozen=True, slots=True)
class FraudDataset:
    """Validated numeric feature matrix and binary labels."""

    features: pd.DataFrame
    labels: pd.Series
    row_ids: pd.Series


def load_dataset(config: DataConfig, *, reader: CsvReader = pd.read_csv) -> FraudDataset:
    """Read CSV data and enforce a finite numeric binary-classification schema."""
    try:
        records = reader(config.path)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        message = f"Cannot read dataset '{config.path}': {error}"
        raise DataValidationError(message) from error
    required = {config.label_column, config.amount_column, config.time_column}
    missing = sorted(required.difference(records.columns))
    if missing:
        raise DataValidationError(f"Dataset is missing columns: {', '.join(missing)}")
    if records.empty:
        raise DataValidationError("Dataset contains no transactions")
    if records.columns.has_duplicates:
        raise DataValidationError("Dataset column names must be unique")

    numeric = records.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        message = "Dataset features and labels must be numeric and non-null"
        raise DataValidationError(message)
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise DataValidationError("Dataset contains a non-finite numeric value")
    labels = numeric[config.label_column]
    if not np.equal(labels, np.floor(labels)).all() or set(labels.astype(int)) != {0, 1}:
        raise DataValidationError("Label column must contain both binary classes 0 and 1")
    features = numeric.drop(columns=config.label_column).astype("float64")
    return FraudDataset(
        features=features,
        labels=labels.astype("int64"),
        row_ids=pd.Series(records.index.astype(str), index=records.index, name="row_id"),
    )

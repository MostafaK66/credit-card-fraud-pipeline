"""Immutable application configuration."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

from credit_fraud.errors import ConfigurationError

T = TypeVar("T")


def _section(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise ConfigurationError(f"Missing or invalid [{name}] section")
    return cast(dict[str, Any], value)


def _value(section: dict[str, Any], name: str, expected: type[T]) -> T:
    value = section.get(name)
    invalid_integer = expected is int and isinstance(value, bool)
    if not isinstance(value, expected) or invalid_integer:
        raise ConfigurationError(f"'{name}' must be a {expected.__name__}")
    return value


def _plain_filename(value: str, name: str, suffix: str) -> str:
    candidate = Path(value)
    if candidate.name != value or candidate.suffix != suffix:
        raise ConfigurationError(f"'{name}' must be a plain {suffix} filename")
    return value


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Input path and required semantic column names."""

    path: Path
    label_column: str
    amount_column: str
    time_column: str

    def __post_init__(self) -> None:
        columns = (self.label_column, self.amount_column, self.time_column)
        if any(not column.strip() for column in columns):
            raise ConfigurationError("Data column names cannot be empty")
        if len(set(columns)) != len(columns):
            raise ConfigurationError("Data column names must be unique")


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Holdout split controls."""

    test_fraction: float
    random_seed: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.test_fraction) or not 0.0 < self.test_fraction < 1.0:
            raise ConfigurationError("'test_fraction' must be between 0 and 1")
        if self.random_seed < 0:
            raise ConfigurationError("'random_seed' must be non-negative")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Resampling, tuning, and decision settings."""

    sampling: str
    cv_splits: int
    smote_neighbors: int
    c_values: tuple[float, ...]
    max_iterations: int
    decision_threshold: float

    def __post_init__(self) -> None:
        if self.sampling not in {"smote", "class_weight"}:
            raise ConfigurationError("'sampling' must be 'smote' or 'class_weight'")
        if self.cv_splits < 2:
            raise ConfigurationError("'cv_splits' must be at least 2")
        if self.smote_neighbors < 1:
            raise ConfigurationError("'smote_neighbors' must be positive")
        if not self.c_values or any(
            not math.isfinite(value) or value <= 0 for value in self.c_values
        ):
            raise ConfigurationError("'c_values' must contain positive finite numbers")
        if self.max_iterations < 1:
            raise ConfigurationError("'max_iterations' must be positive")
        if (
            not math.isfinite(self.decision_threshold)
            or not 0.0 < self.decision_threshold < 1.0
        ):
            raise ConfigurationError("'decision_threshold' must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Output locations."""

    directory: Path
    predictions_file: str
    metrics_file: str
    model_file: str

    def __post_init__(self) -> None:
        _plain_filename(self.predictions_file, "predictions_file", ".csv")
        _plain_filename(self.metrics_file, "metrics_file", ".json")
        _plain_filename(self.model_file, "model_file", ".skops")


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Validated root configuration."""

    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    output: OutputConfig

    @classmethod
    def from_toml(cls, path: Path) -> AppConfig:
        """Load TOML and resolve data/output paths relative to the file."""
        try:
            with path.open("rb") as stream:
                document = tomllib.load(stream)
        except OSError as error:
            message = f"Cannot read configuration '{path}': {error}"
            raise ConfigurationError(message) from error
        except tomllib.TOMLDecodeError as error:
            raise ConfigurationError(f"Invalid TOML in '{path}': {error}") from error
        data = _section(document, "data")
        split = _section(document, "split")
        model = _section(document, "model")
        output = _section(document, "output")
        raw_c_values = model.get("c_values")
        if not isinstance(raw_c_values, list) or any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            for value in raw_c_values
        ):
            raise ConfigurationError("'c_values' must be a numeric array")
        base = path.resolve().parent
        return cls(
            data=DataConfig(
                path=base / _value(data, "path", str),
                label_column=_value(data, "label_column", str),
                amount_column=_value(data, "amount_column", str),
                time_column=_value(data, "time_column", str),
            ),
            split=SplitConfig(
                test_fraction=float(_value(split, "test_fraction", float)),
                random_seed=_value(split, "random_seed", int),
            ),
            model=ModelConfig(
                sampling=_value(model, "sampling", str),
                cv_splits=_value(model, "cv_splits", int),
                smote_neighbors=_value(model, "smote_neighbors", int),
                c_values=tuple(float(value) for value in raw_c_values),
                max_iterations=_value(model, "max_iterations", int),
                decision_threshold=float(_value(model, "decision_threshold", float)),
            ),
            output=OutputConfig(
                directory=base / _value(output, "directory", str),
                predictions_file=_value(output, "predictions_file", str),
                metrics_file=_value(output, "metrics_file", str),
                model_file=_value(output, "model_file", str),
            ),
        )

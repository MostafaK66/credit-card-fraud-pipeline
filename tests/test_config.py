"""Configuration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from credit_fraud.config import (
    AppConfig,
    DataConfig,
    ModelConfig,
    OutputConfig,
    SplitConfig,
)
from credit_fraud.errors import ConfigurationError


def test_loads_example_configuration() -> None:
    config = AppConfig.from_toml(Path("config.example.toml"))
    assert config.model.sampling == "smote"
    assert config.model.c_values == (0.01, 0.1, 1.0, 10.0)
    assert config.data.path.is_absolute()


@pytest.mark.parametrize("content", ["", "[data\n", "[data]\npath = 1"])
def test_rejects_missing_or_malformed_toml(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(content)
    with pytest.raises(ConfigurationError):
        AppConfig.from_toml(path)


def test_missing_file_has_context(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="Cannot read configuration"):
        AppConfig.from_toml(tmp_path / "missing.toml")


@pytest.mark.parametrize("columns", [("", "A", "T"), ("C", "C", "T")])
def test_data_column_validation(columns: tuple[str, str, str]) -> None:
    with pytest.raises(ConfigurationError):
        DataConfig(Path("data.csv"), *columns)


@pytest.mark.parametrize("arguments", [(0.0, 1), (1.0, 1), (0.2, -1), (float("nan"), 1)])
def test_split_validation(arguments: tuple[float, int]) -> None:
    with pytest.raises(ConfigurationError):
        SplitConfig(*arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        ("bad", 3, 2, (1.0,), 100, 0.5),
        ("smote", 1, 2, (1.0,), 100, 0.5),
        ("smote", 3, 0, (1.0,), 100, 0.5),
        ("smote", 3, 2, (), 100, 0.5),
        ("smote", 3, 2, (float("nan"),), 100, 0.5),
        ("smote", 3, 2, (1.0,), 0, 0.5),
        ("smote", 3, 2, (1.0,), 100, 1.0),
    ],
)
def test_model_validation(arguments: tuple[object, ...]) -> None:
    with pytest.raises(ConfigurationError):
        ModelConfig(*arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "index,value",
    [(0, "folder/p.csv"), (0, "p.json"), (1, "m.csv"), (2, "model.pkl")],
)
def test_output_validation(index: int, value: str) -> None:
    names = ["p.csv", "m.json", "model.skops"]
    names[index] = value
    with pytest.raises(ConfigurationError):
        OutputConfig(Path("output"), *names)

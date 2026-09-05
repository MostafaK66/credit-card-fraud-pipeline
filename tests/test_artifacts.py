"""Artifact persistence tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from credit_fraud.artifacts import write_artifacts
from credit_fraud.config import AppConfig, OutputConfig
from credit_fraud.data import FraudDataset
from credit_fraud.errors import ArtifactError
from credit_fraud.modeling import TrainingResult, train_and_evaluate


def output(tmp_path: Path) -> OutputConfig:
    return OutputConfig(tmp_path / "out", "pred.csv", "metrics.json", "model.skops")


def result() -> TrainingResult:
    return TrainingResult(
        model=object(),
        predictions=pd.DataFrame({"score": [0.5]}),
        metrics={"average_precision": 0.8},
        best_parameters={"C": 1.0},
    )


def test_writes_predictions_metrics_and_model(tmp_path: Path) -> None:
    def dump(_: object, path: Path) -> None:
        path.write_text("safe model")

    paths = write_artifacts(result(), output(tmp_path), dumper=dump)
    assert all(path.is_file() for path in paths)
    document = json.loads(paths[1].read_text())
    assert document["best_parameters"] == {"C": 1.0}


def test_wraps_artifact_error(tmp_path: Path) -> None:
    def fail(_: object, __: Path) -> None:
        raise OSError("disk full")

    with pytest.raises(ArtifactError, match="Cannot write"):
        write_artifacts(result(), output(tmp_path), dumper=fail)


def test_persists_real_pipeline_with_skops(
    app_config: AppConfig, fraud_dataset: FraudDataset
) -> None:
    trained = train_and_evaluate(fraud_dataset, app_config.split, app_config.model)
    paths = write_artifacts(trained, app_config.output)
    assert paths[2].suffix == ".skops"
    assert paths[2].stat().st_size > 0

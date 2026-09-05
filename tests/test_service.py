"""Service orchestration tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from credit_fraud.config import AppConfig
from credit_fraud.data import FraudDataset
from credit_fraud.modeling import TrainingResult
from credit_fraud.service import FraudDetectionService


def test_orchestrates_injected_boundaries(app_config: AppConfig) -> None:
    dataset = FraudDataset(
        pd.DataFrame({"V1": [0.0, 1.0]}),
        pd.Series([0, 1]),
        pd.Series(["a", "b"]),
    )
    result = TrainingResult(
        object(),
        pd.DataFrame(),
        {"average_precision": 0.8, "roc_auc": 0.9},
        {"C": 1.0},
    )
    seen: list[object] = []

    def writer(*args: object) -> tuple[Path, Path, Path]:
        seen.extend(args)
        return (Path("p"), Path("m"), Path("model"))

    summary = FraudDetectionService(
        loader=lambda _: dataset,
        trainer=lambda *_: result,
        writer=writer,
    ).run(app_config)
    assert summary.transaction_count == 2
    assert summary.average_precision == 0.8
    assert summary.roc_auc == 0.9
    assert summary.artifact_paths[2] == Path("model")
    assert any(value is result for value in seen)

"""Application orchestration independent of the CLI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from credit_fraud.artifacts import write_artifacts
from credit_fraud.config import (
    AppConfig,
    DataConfig,
    ModelConfig,
    OutputConfig,
    SplitConfig,
)
from credit_fraud.data import FraudDataset, load_dataset
from credit_fraud.modeling import TrainingResult, train_and_evaluate

Loader = Callable[[DataConfig], FraudDataset]
Trainer = Callable[[FraudDataset, SplitConfig, ModelConfig], TrainingResult]
Writer = Callable[[TrainingResult, OutputConfig], tuple[Path, Path, Path]]


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Headline model metrics and artifact locations."""

    transaction_count: int
    average_precision: float
    roc_auc: float
    artifact_paths: tuple[Path, Path, Path]


class FraudDetectionService:
    """Coordinate loading, training, evaluation, and persistence boundaries."""

    def __init__(
        self,
        *,
        loader: Loader = load_dataset,
        trainer: Trainer = train_and_evaluate,
        writer: Writer = write_artifacts,
    ) -> None:
        self._loader = loader
        self._trainer = trainer
        self._writer = writer

    def run(self, config: AppConfig) -> RunSummary:
        """Execute one configured training run."""
        dataset = self._loader(config.data)
        result = self._trainer(dataset, config.split, config.model)
        paths = self._writer(result, config.output)
        return RunSummary(
            transaction_count=len(dataset.labels),
            average_precision=float(result.metrics["average_precision"]),
            roc_auc=float(result.metrics["roc_auc"]),
            artifact_paths=paths,
        )

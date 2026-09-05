"""Result persistence, including safer non-pickle model serialization."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from credit_fraud.config import OutputConfig
from credit_fraud.errors import ArtifactError
from credit_fraud.modeling import TrainingResult

ModelDumper = Callable[[object, Path], None]


def _dump_model(model: object, path: Path) -> None:
    from skops.io import dump

    dump(model, path)


def write_artifacts(
    result: TrainingResult,
    output: OutputConfig,
    *,
    dumper: ModelDumper = _dump_model,
) -> tuple[Path, Path, Path]:
    """Persist holdout predictions, metrics, and the selected estimator."""
    predictions_path = output.directory / output.predictions_file
    metrics_path = output.directory / output.metrics_file
    model_path = output.directory / output.model_file
    document = {
        "best_parameters": result.best_parameters,
        "metrics": result.metrics,
    }
    try:
        output.directory.mkdir(parents=True, exist_ok=True)
        result.predictions.to_csv(predictions_path, index=False)
        metrics_path.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        dumper(result.model, model_path)
    except (OSError, TypeError, ValueError) as error:
        raise ArtifactError(f"Cannot write result artifacts: {error}") from error
    return predictions_path, metrics_path, model_path

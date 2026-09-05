"""Leakage-safe model selection and holdout evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler

from credit_fraud.config import ModelConfig, SplitConfig
from credit_fraud.data import FraudDataset
from credit_fraud.errors import TrainingError

MetricValue = float | int


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Fitted estimator, holdout predictions, and fraud-specific metrics."""

    model: object
    predictions: pd.DataFrame
    metrics: dict[str, MetricValue]
    best_parameters: dict[str, object]


def train_and_evaluate(
    dataset: FraudDataset,
    split_config: SplitConfig,
    model_config: ModelConfig,
) -> TrainingResult:
    """Split first, tune a fold-local pipeline, and evaluate untouched holdout data."""
    try:
        (
            train_features,
            test_features,
            train_labels,
            test_labels,
            _train_ids,
            test_ids,
        ) = train_test_split(
            dataset.features,
            dataset.labels,
            dataset.row_ids,
            test_size=split_config.test_fraction,
            random_state=split_config.random_seed,
            stratify=dataset.labels,
        )
    except ValueError as error:
        raise TrainingError(f"Cannot create stratified holdout split: {error}") from error

    class_counts = train_labels.value_counts()
    minority_count = int(class_counts.min())
    if minority_count < model_config.cv_splits:
        raise TrainingError(
            "Training data has fewer minority samples than cross-validation folds"
        )
    fold_training_minority = minority_count - math.ceil(
        minority_count / model_config.cv_splits
    )
    if (
        model_config.sampling == "smote"
        and fold_training_minority <= model_config.smote_neighbors
    ):
        raise TrainingError(
            "SMOTE requires more minority samples in every training fold than "
            "the configured smote_neighbors"
        )

    classifier = LogisticRegression(
        class_weight="balanced" if model_config.sampling == "class_weight" else None,
        max_iter=model_config.max_iterations,
        random_state=split_config.random_seed,
        solver="liblinear",
    )
    steps: list[tuple[str, Any]] = [("scale", RobustScaler())]
    if model_config.sampling == "smote":
        steps.append(
            (
                "sample",
                SMOTE(
                    k_neighbors=model_config.smote_neighbors,
                    random_state=split_config.random_seed,
                ),
            )
        )
    steps.append(("classifier", classifier))
    pipeline = Pipeline(steps)
    cross_validation = StratifiedKFold(
        n_splits=model_config.cv_splits,
        shuffle=True,
        random_state=split_config.random_seed,
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"classifier__C": model_config.c_values},
        scoring="average_precision",
        cv=cross_validation,
        refit=True,
        error_score="raise",
        n_jobs=1,
    )
    try:
        search.fit(train_features, train_labels)
        scores = np.asarray(search.predict_proba(test_features)[:, 1], dtype=np.float64)
    except ValueError as error:
        raise TrainingError(f"Model fitting failed: {error}") from error
    predicted = (scores >= model_config.decision_threshold).astype(np.int64)
    actual = test_labels.to_numpy(dtype=np.int64)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()
    metrics: dict[str, MetricValue] = {
        "average_precision": float(average_precision_score(actual, scores)),
        "best_cv_average_precision": float(search.best_score_),
        "decision_threshold": model_config.decision_threshold,
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "false_negative": int(false_negative),
        "false_positive": int(false_positive),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, scores)),
        "test_transactions": len(actual),
        "true_negative": int(true_negative),
        "true_positive": int(true_positive),
    }
    best_parameters = {
        name.removeprefix("classifier__"): value
        for name, value in search.best_params_.items()
    }
    predictions = pd.DataFrame(
        {
            "row_id": test_ids.to_numpy(),
            "actual": actual,
            "fraud_probability": scores,
            "predicted": predicted,
        }
    ).sort_values("row_id", ignore_index=True)
    return TrainingResult(
        model=search.best_estimator_,
        predictions=predictions,
        metrics=metrics,
        best_parameters=best_parameters,
    )

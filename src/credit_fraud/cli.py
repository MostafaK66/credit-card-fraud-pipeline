"""Thin command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from credit_fraud.config import AppConfig
from credit_fraud.errors import FraudDetectionError
from credit_fraud.service import FraudDetectionService


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Train a credit-card fraud detector")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="TOML configuration path (default: config.toml)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the application and translate domain failures into concise errors."""
    arguments = build_parser().parse_args(argv)
    try:
        summary = FraudDetectionService().run(AppConfig.from_toml(arguments.config))
    except FraudDetectionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"completed: {summary.transaction_count} transactions")
    print(f"average precision: {summary.average_precision:.6f}")
    print(f"ROC AUC: {summary.roc_auc:.6f}")
    for path in summary.artifact_paths:
        print(f"artifact: {path}")
    return 0

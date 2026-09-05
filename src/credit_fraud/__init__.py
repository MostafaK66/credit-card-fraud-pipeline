"""Leakage-safe credit-card fraud detection pipeline."""

from credit_fraud.config import AppConfig
from credit_fraud.service import FraudDetectionService

__all__ = ["AppConfig", "FraudDetectionService"]
__version__ = "1.0.0"

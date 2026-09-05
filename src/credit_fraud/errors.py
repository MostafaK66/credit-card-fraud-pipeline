"""Domain-specific errors."""


class FraudDetectionError(Exception):
    """Base exception for expected application failures."""


class ConfigurationError(FraudDetectionError):
    """Configuration is missing, malformed, or invalid."""


class DataValidationError(FraudDetectionError):
    """Input data does not satisfy the fraud-dataset contract."""


class TrainingError(FraudDetectionError):
    """The configured model cannot be trained or evaluated."""


class ArtifactError(FraudDetectionError):
    """A result artifact cannot be persisted."""

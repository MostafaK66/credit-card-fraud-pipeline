# Credit Card Fraud Pipeline

A leakage-safe, reproducible baseline for binary credit-card transaction fraud
detection. The project turns the original exploratory script collection into a
maintainable Python 3.11+ package with validated input, fold-local imbalance handling,
probability-aware metrics, and safer model persistence.

This is an educational benchmark, not a production payment-risk system. Results from
one historical dataset do not establish suitability for live financial decisions.
Thresholds, drift, calibration, fairness, latency, privacy, and human-review policy
must be validated for the deployment environment.

## What changed

- Uses a Hatchling `src/` layout, immutable TOML configuration, strict typing,
  domain-specific exceptions, and a thin CLI.
- Removes personal absolute paths, import-time tracing, warning suppression, duplicate
  fold calls, generated images, bytecode, and IDE metadata.
- Validates that every feature is numeric and finite and that the target contains both
  binary classes.
- Creates one stratified holdout split before any transformation.
- Fits `RobustScaler` and SMOTE inside each cross-validation training fold, preventing
  preprocessing and resampling leakage.
- Tunes deterministic logistic regression using average precision instead of accuracy,
  which is misleading for highly imbalanced fraud data.
- Reports average precision, ROC AUC, precision, recall, F1, and all confusion-matrix
  counts on untouched holdout transactions.
- Saves the fitted pipeline in `.skops` format rather than unsafe pickle/joblib format.

## Installation

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

For development, install `.[dev]`.

## Dataset contract

The dataset is deliberately not included. Place a CSV at the configured path; the
default is `data/creditcard.csv`. It must contain:

- `Class`: binary labels `0` and `1`.
- `Amount` and `Time`: numeric columns retained as model features.
- At least one numeric feature column; all values must be finite and non-null.
- Enough minority examples for the configured stratified folds and SMOTE neighbors.

All column names are configurable. Additional numeric feature columns such as `V1`
through `V28` are accepted automatically. Payment data and outputs are ignored by Git.

## Usage

```bash
cp config.example.toml config.toml
fraud-detect --config config.toml
# Equivalent:
python -m credit_fraud --config config.toml
```

On Windows, use `Copy-Item config.example.toml config.toml`.

The output directory receives:

- `predictions.csv`: row identifiers, labels, probabilities, and thresholded results.
- `metrics.json`: holdout metrics and selected hyperparameters.
- `fraud_detector.skops`: fitted scaler/sampler/classifier pipeline.

Set `sampling = "class_weight"` to use balanced logistic-regression weights without
SMOTE. `decision_threshold` changes classifications and confusion metrics without
changing probability metrics.

## Architecture

```text
TOML -> validated config -> validated numeric CSV
     -> stratified holdout split
     -> CV [robust scaling -> optional SMOTE -> logistic regression]
     -> holdout probabilities -> fraud metrics -> CSV/JSON/.skops artifacts
```

The loader, trainer, and artifact writer are injected into the service, keeping
filesystem and estimator boundaries independently testable.

## Development

```bash
make install
make quality
```

Tests use small deterministic synthetic transaction data. They do not need the real
dataset, a network connection, external service, GPU, or cloud account.

## Attribution and license

Copyright © 2026 MostafaK66. Released under the [MIT License](LICENSE). See
[NOTICE](NOTICE) for repository history and data exclusions.

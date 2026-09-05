# Engineering contract

- Support Python 3.11 and 3.12 using a `src/` package layout.
- Validate configuration and input schemas before fitting any estimator.
- Split before scaling or resampling; fit every transformation on training folds only.
- Prefer probability-aware imbalance metrics over accuracy.
- Keep model, filesystem, and data-loading boundaries injectable.
- Tests must be deterministic, offline, and use only small synthetic datasets.
- Run Ruff, strict mypy, branch-aware coverage, and compile checks before merging.
- Never commit payment data, model artifacts, generated reports, caches, or credentials.

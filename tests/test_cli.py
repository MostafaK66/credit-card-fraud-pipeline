"""CLI tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pytest import CaptureFixture, MonkeyPatch

from credit_fraud import cli
from credit_fraud.errors import ConfigurationError


def test_cli_success(monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]) -> None:
    summary = SimpleNamespace(
        transaction_count=100,
        average_precision=0.75,
        roc_auc=0.9,
        artifact_paths=(Path("metrics.json"),),
    )
    monkeypatch.setattr(cli.AppConfig, "from_toml", lambda _: object())
    service = SimpleNamespace(run=lambda _: summary)
    monkeypatch.setattr(cli, "FraudDetectionService", lambda: service)
    assert cli.main(["--config", "example.toml"]) == 0
    output = capsys.readouterr().out
    assert "100 transactions" in output
    assert "average precision: 0.750000" in output
    assert "artifact: metrics.json" in output


def test_cli_error(monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]) -> None:
    def fail(_: Path) -> object:
        raise ConfigurationError("bad config")

    monkeypatch.setattr(cli.AppConfig, "from_toml", fail)
    assert cli.main([]) == 2
    assert "error: bad config" in capsys.readouterr().err


def test_parser_default() -> None:
    assert cli.build_parser().parse_args([]).config == Path("config.toml")

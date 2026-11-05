"""Command-line interface: ``cocoasense <stage>``."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import typer

from cocoasense import pipeline
from cocoasense.config import ProjectConfig, load_config, setup_logging

app = typer.Typer(
    help="CocoaSense AI: cocoa price forecasting and forward-sales simulation (simulated "
    "results on public prices, not Ghana's actual outcomes).",
    no_args_is_help=True,
    add_completion=False,
)

RootOpt = typer.Option(None, "--root", help="Project root (defaults to the repository root).")


def _cfg(root: Path | None) -> ProjectConfig:
    setup_logging(logging.INFO)
    return load_config(root)


@app.command()
def ingest(
    refresh: bool = typer.Option(False, help="Download fresh copies (old files are archived)."),
    root: Path | None = RootOpt,
) -> None:
    """Download or reuse raw files and update data/raw/manifest.json."""
    for e in pipeline.stage_ingest(_cfg(root), refresh=refresh):
        typer.echo(f"{e.path}: {e.rows} rows {e.date_min}..{e.date_max} ({e.origin})")


@app.command("validate")
def validate_cmd(root: Path | None = RootOpt) -> None:
    """Validate and align series; write data/interim and artifacts/data_quality.md."""
    m = pipeline.stage_validate(_cfg(root))
    typer.echo(f"monthly table: {len(m)} rows {m.index.min():%Y-%m}..{m.index.max():%Y-%m}")


@app.command()
def features(root: Path | None = RootOpt) -> None:
    """Build the documented feature table in data/processed."""
    t = pipeline.stage_features(_cfg(root))
    typer.echo(f"feature table: {t.shape[0]} rows x {t.shape[1]} columns")


@app.command()
def train(root: Path | None = RootOpt) -> None:
    """Development stage: fit on data to 2015 and forecast the 2016–2019 validation period."""
    pipeline.stage_train(_cfg(root))
    typer.echo("validation predictions and leaderboards written to artifacts/")


@app.command()
def freeze(
    overwrite: bool = typer.Option(False, help="Replace an existing frozen.yaml."),
    root: Path | None = RootOpt,
) -> None:
    """Choose settings from validation results and write configs/frozen.yaml."""
    frozen = pipeline.stage_freeze(_cfg(root), overwrite=overwrite)
    typer.echo(
        f"frozen: simulation driver={frozen['simulation']['driver_model']} "
        f"a={frozen['simulation']['a']}. Commit configs/frozen.yaml before --final."
    )


@app.command()
def evaluate(
    final: bool = typer.Option(False, "--final", help="Score the held-out test period (once)."),
    root: Path | None = RootOpt,
) -> None:
    """Score validation predictions, or with --final run the frozen models on the test period."""
    cfg = _cfg(root)
    if final:
        pipeline.stage_evaluate_final(cfg)
        typer.echo("test predictions and metrics written; run logged in artifacts/final_runs.log")
    else:
        pipeline.stage_score_validation(cfg)
        typer.echo("validation metrics written (test period untouched)")


@app.command()
def simulate(root: Path | None = RootOpt) -> None:
    """Forward-sales simulation on the test period with the frozen rule."""
    r = pipeline.stage_simulate(_cfg(root))
    typer.echo(f"simulated {r['decision_date'].nunique()} decision dates")


@app.command()
def report(root: Path | None = RootOpt) -> None:
    """Write figures to artifacts/figures."""
    pipeline.stage_report(_cfg(root))
    typer.echo("figures written to artifacts/figures")


@app.command()
def learn(root: Path | None = RootOpt) -> None:
    """Learning track: refit every month on all prices known so far; forecast ahead (D19)."""
    rows = pipeline.stage_learn(_cfg(root))
    typer.echo(f"learning track: {len(rows)} forecasts ({int(rows['y_true'].isna().sum())} live)")


@app.command()
def insights(root: Path | None = RootOpt) -> None:
    """Confidence for every forecast, success scorecards, calibration and live forecasts."""
    rows = pipeline.stage_insights(_cfg(root))
    typer.echo(f"insights written for {len(rows)} forecasts (artifacts/metrics/scorecard.md)")


@app.command()
def update(
    refresh: bool = typer.Option(True, help="Download the latest prices (--no-refresh: reuse)."),
    root: Path | None = RootOpt,
) -> None:
    """Fetch the latest prices, score earlier forecasts, refit the learning track, forecast."""
    rec = pipeline.stage_update(_cfg(root), refresh=refresh)
    typer.echo(f"data through {rec['data_through']} (was {rec['data_through_before']})")


lstm_app = typer.Typer(
    help="LSTM stretch model: its own validation -> freeze -> final cycle (docs/decisions.md D16).",
    no_args_is_help=True,
)
app.add_typer(lstm_app, name="lstm")


@lstm_app.command("train")
def lstm_train(root: Path | None = RootOpt) -> None:
    """Fit the LSTM on data to 2015 and forecast the 2016–2019 validation period."""
    pipeline.stage_lstm_train(_cfg(root))
    typer.echo("LSTM validation predictions and metrics written to artifacts/")


@lstm_app.command("freeze")
def lstm_freeze(
    overwrite: bool = typer.Option(False, help="Replace an existing frozen_lstm.yaml."),
    root: Path | None = RootOpt,
) -> None:
    """Choose the LSTM training start from validation results; write configs/frozen_lstm.yaml."""
    frozen = pipeline.stage_lstm_freeze(_cfg(root), overwrite=overwrite)
    typer.echo(
        f"frozen LSTM: train_start={frozen['model']['train_start']}. "
        "Commit configs/frozen_lstm.yaml before `cocoasense lstm evaluate --final`."
    )


@lstm_app.command("evaluate")
def lstm_evaluate(
    final: bool = typer.Option(False, "--final", help="Score the held-out test period (once)."),
    root: Path | None = RootOpt,
) -> None:
    """Score LSTM validation predictions, or with --final run the frozen LSTM on the test period."""
    cfg = _cfg(root)
    if final:
        pipeline.stage_lstm_evaluate_final(cfg)
        typer.echo("LSTM test predictions written; run logged in artifacts/final_runs.log")
    else:
        pipeline.stage_lstm_score_validation(cfg)
        typer.echo("LSTM validation metrics written (test period untouched)")


@app.command("all")
def run_all(
    refresh: bool = typer.Option(False, help="Re-download raw data first."),
    root: Path | None = RootOpt,
) -> None:
    """Run every stage from raw data to artifacts."""
    pipeline.stage_all(_cfg(root), refresh=refresh)
    typer.echo("pipeline complete")


@app.command()
def dashboard() -> None:  # pragma: no cover - launches a server
    """Start the Streamlit dashboard."""
    app_path = Path(__file__).parent / "dashboard" / "Overview.py"
    raise typer.Exit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--client.toolbarMode",
                "viewer",
            ]
        )
    )


@app.command("app")
def desktop_app() -> None:  # pragma: no cover - opens a window
    """Open the dashboard in its own app window; closing the window stops the server."""
    from cocoasense.desktop import launch

    launch()


@app.command("install-shortcut")
def install_shortcut() -> None:  # pragma: no cover - writes to the Windows desktop
    """Put a 'CocoaSense AI' shortcut on the Windows desktop that opens the app window."""
    from cocoasense.desktop import create_shortcut

    typer.echo(f"shortcut created: {create_shortcut()}")


if __name__ == "__main__":  # pragma: no cover
    app()

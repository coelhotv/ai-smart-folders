from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .config import load_config
from .pipeline import build_pipeline

try:
    import typer
except Exception:
    typer = None


def _dump_model(value: object) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump()  # type: ignore[no-any-return]
    if hasattr(value, "dict"):
        return value.dict()  # type: ignore[no-any-return]
    return dict(value)  # type: ignore[arg-type]


def _run_impl(config_path: Optional[Path], dry_run: bool, limit: Optional[int]) -> int:
    config = load_config(config_path)
    config.dry_run = dry_run
    pipeline = build_pipeline(config=config)
    try:
        metrics = pipeline.run(dry_run=dry_run, limit=limit)
    finally:
        pipeline.close()
    print(json.dumps(_dump_model(metrics), indent=2, default=str))
    return 0


def _benchmark_impl(config_path: Optional[Path], limit: Optional[int], dataset: Optional[Path]) -> int:
    config = load_config(config_path)
    pipeline = build_pipeline(config=config)
    try:
        metrics = pipeline.benchmark(dataset_path=dataset, sample_limit=limit)
    finally:
        pipeline.close()
    print(json.dumps(_dump_model(metrics), indent=2, default=str))
    return 0


def _undo_impl(config_path: Optional[Path]) -> int:
    config = load_config(config_path)
    pipeline = build_pipeline(config=config)
    try:
        restored = pipeline.undo_last_run()
    finally:
        pipeline.close()
    print(json.dumps({"restored_files": restored}, indent=2))
    return 0


def _reindex_impl(config_path: Optional[Path]) -> int:
    from .taxonomy import scan_existing_taxonomy

    config = load_config(config_path)
    taxonomy = scan_existing_taxonomy(config)
    print(json.dumps(taxonomy, indent=2, default=str))
    return 0


def _argparse_main() -> None:
    parser = argparse.ArgumentParser(description="AI Smart Folders CLI")
    parser.add_argument("--config", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run")
    run_parser.add_argument("--limit", type=int, default=None)

    dry_parser = sub.add_parser("dry-run")
    dry_parser.add_argument("--limit", type=int, default=None)

    benchmark_parser = sub.add_parser("benchmark")
    benchmark_parser.add_argument("--limit", type=int, default=None)
    benchmark_parser.add_argument("--dataset", type=Path, default=None)

    sub.add_parser("undo-last-run")
    sub.add_parser("reindex-taxonomy")

    args = parser.parse_args()
    if args.command == "run":
        raise SystemExit(_run_impl(args.config, False, args.limit))
    if args.command == "dry-run":
        raise SystemExit(_run_impl(args.config, True, args.limit))
    if args.command == "benchmark":
        raise SystemExit(_benchmark_impl(args.config, args.limit, args.dataset))
    if args.command == "undo-last-run":
        raise SystemExit(_undo_impl(args.config))
    if args.command == "reindex-taxonomy":
        raise SystemExit(_reindex_impl(args.config))


if typer is not None:
    app = typer.Typer(add_completion=False, help="AI Smart Folders CLI")

    @app.command("run")
    def run_command(config: Optional[Path] = typer.Option(None), limit: Optional[int] = typer.Option(None)) -> None:
        raise typer.Exit(_run_impl(config, False, limit))

    @app.command("dry-run")
    def dry_run_command(config: Optional[Path] = typer.Option(None), limit: Optional[int] = typer.Option(None)) -> None:
        raise typer.Exit(_run_impl(config, True, limit))

    @app.command("benchmark")
    def benchmark_command(
        config: Optional[Path] = typer.Option(None),
        limit: Optional[int] = typer.Option(None),
        dataset: Optional[Path] = typer.Option(None),
    ) -> None:
        raise typer.Exit(_benchmark_impl(config, limit, dataset))

    @app.command("undo-last-run")
    def undo_command(config: Optional[Path] = typer.Option(None)) -> None:
        raise typer.Exit(_undo_impl(config))

    @app.command("reindex-taxonomy")
    def reindex_command(config: Optional[Path] = typer.Option(None)) -> None:
        raise typer.Exit(_reindex_impl(config))

    def main() -> None:
        app()
else:
    def main() -> None:
        _argparse_main()

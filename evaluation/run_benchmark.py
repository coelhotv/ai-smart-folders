#!/usr/bin/env python3
"""
OCR Benchmark Runner
====================
Runs the AI Smart Folders benchmark for one or more model configurations
and saves timestamped results to evaluation/results/.

Usage:
    python evaluation/run_benchmark.py --configs test-deepseek.yaml test-numarkdown.yaml
    python evaluation/run_benchmark.py --configs test-deepseek.yaml --dataset evaluation/ocr_benchmark_folders.jsonl
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "evaluation" / "results"
DEFAULT_DATASET = ROOT / "evaluation" / "ocr_benchmark_folders.jsonl"


def run_benchmark_for_config(config_path: Path, dataset_path: Path) -> dict:
    """Run the benchmark CLI for a single config and return the parsed JSON report."""
    cmd = [
        sys.executable,
        str(ROOT / "main.py"),
        "benchmark",
        "--config", str(config_path),
        "--dataset", str(dataset_path),
    ]
    print(f"\n{'=' * 60}")
    print(f"  Running benchmark with config: {config_path.name}")
    print(f"  Dataset:                       {dataset_path.name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"[ERROR] Benchmark failed (exit {result.returncode})")
        print(result.stderr[-3000:] if result.stderr else "(no stderr)")
        return {"error": result.stderr, "exit_code": result.returncode}

    # The CLI prints JSON to stdout; log stderr for debugging
    if result.stderr:
        for line in result.stderr.strip().splitlines()[-20:]:
            print(f"  LOG | {line}")

    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Could not parse JSON output: {exc}")
        print(result.stdout[:500])
        return {"error": str(exc), "raw_stdout": result.stdout}

    report["_meta"] = {
        "config_file": config_path.name,
        "dataset_file": dataset_path.name,
        "wall_clock_seconds": round(elapsed, 2),
        "run_at": datetime.now(timezone.utc).isoformat(),
    }
    return report


def extract_ocr_model(config_path: Path) -> str:
    """Parse the YAML config to extract ocr_model name (no pyyaml dependency)."""
    try:
        for line in config_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("ocr_model:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return config_path.stem


def summarize(report: dict) -> None:
    """Print a quick human-readable summary of a BenchmarkReport."""
    total = report.get("total_cases", "?")
    l1 = report.get("matched_level1", "?")
    l2 = report.get("matched_level2", "?")
    full = report.get("full_matches", "?")
    review = report.get("matched_review_flag", "?")
    failures = report.get("failures", "?")
    meta = report.get("_meta", {})

    print("\n  ┌─────────────────────────────────────────┐")
    print(f"  │  Config:        {meta.get('config_file', '?'):<26}│")
    print(f"  │  Total cases:   {total:<26}│")
    print(f"  │  Full matches:  {full}/{total:<25}│")
    print(f"  │  L1 matches:    {l1}/{total:<25}│")
    print(f"  │  L2 matches:    {l2}/{total:<25}│")
    print(f"  │  Review flag:   {review}/{total:<25}│")
    print(f"  │  Failures:      {failures:<26}│")
    print(f"  │  Wall clock:    {meta.get('wall_clock_seconds', '?')}s{'':<23}│")
    print(f"  └─────────────────────────────────────────┘")


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR Benchmark Runner — runs and persists benchmark results")
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        metavar="CONFIG.yaml",
        help="One or more config YAML files (relative to repo root or absolute)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"JSONL dataset file (default: {DEFAULT_DATASET.relative_to(ROOT)})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Directory to write results (default: {RESULTS_DIR.relative_to(ROOT)})",
    )
    args = parser.parse_args()

    dataset_path = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    if not dataset_path.exists():
        sys.exit(f"[ERROR] Dataset not found: {dataset_path}")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    all_reports: dict[str, dict] = {}

    for cfg_arg in args.configs:
        config_path = Path(cfg_arg) if Path(cfg_arg).is_absolute() else ROOT / cfg_arg
        if not config_path.exists():
            print(f"[WARN] Config not found, skipping: {config_path}")
            continue

        ocr_model = extract_ocr_model(config_path)
        report = run_benchmark_for_config(config_path, dataset_path)
        all_reports[config_path.name] = report
        summarize(report)

        # Save individual result
        safe_model = ocr_model.replace("/", "_").replace(":", "_")
        individual_path = output_dir / f"{run_timestamp}_{safe_model}.json"
        individual_path.write_text(json.dumps(report, indent=2, default=str))
        print(f"\n  ✔ Saved → {individual_path.relative_to(ROOT)}")

    # Save combined comparison report
    if len(all_reports) > 1:
        comparison = {
            "run_at": run_timestamp,
            "dataset": str(dataset_path.relative_to(ROOT)),
            "models": all_reports,
        }
        comparison_path = output_dir / f"{run_timestamp}_comparison.json"
        comparison_path.write_text(json.dumps(comparison, indent=2, default=str))
        print(f"\n  ✔ Comparison saved → {comparison_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

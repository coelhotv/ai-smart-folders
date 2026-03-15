from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .models import BenchmarkRecord, BenchmarkReport, DocumentEnvelope


def load_benchmark_dataset(dataset_path: Path) -> List[BenchmarkRecord]:
    records: List[BenchmarkRecord] = []
    with open(dataset_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records.append(BenchmarkRecord(**payload))
    return records


def build_benchmark_report(
    dataset_path: Path,
    records: List[BenchmarkRecord],
    results: List[DocumentEnvelope],
) -> BenchmarkReport:
    by_path = {item.source_path.resolve(): item for item in results}
    report = BenchmarkReport(dataset_path=dataset_path, total_cases=len(records))

    for record in records:
        actual = by_path.get(record.source_path.expanduser().resolve())
        case = {
            "source_path": str(record.source_path),
            "expected_category_l1": record.expected_category_l1,
            "expected_category_l2": record.expected_category_l2,
            "expected_needs_review": record.expected_needs_review,
        }
        if actual is None:
            report.failures += 1
            case["status"] = "missing"
            report.cases.append(case)
            continue

        case.update(
            {
                "actual_category_l1": actual.category_l1,
                "actual_category_l2": actual.category_l2,
                "actual_needs_review": actual.needs_review,
                "status": actual.status,
            }
        )

        level1_match = (
            record.expected_category_l1 is None
            or record.expected_category_l1 == actual.category_l1
        )
        level2_match = (
            record.expected_category_l2 is None
            or record.expected_category_l2 == actual.category_l2
        )
        review_match = (
            record.expected_needs_review is None
            or record.expected_needs_review == actual.needs_review
        )

        if level1_match:
            report.matched_level1 += 1
        if level2_match:
            report.matched_level2 += 1
        if review_match:
            report.matched_review_flag += 1
        if level1_match and level2_match and review_match:
            report.full_matches += 1

        report.cases.append(case)

    return report

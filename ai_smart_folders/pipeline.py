from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import load_config
from .evaluation import build_benchmark_report, load_benchmark_dataset
from .extractors import extract, sniff_mime_type
from .llm import classify_document, understand_document
from .models import AppConfig, BenchmarkReport, ClassificationResult, DocumentEnvelope, RunMetrics
from .storage import ClassificationCache, Database
from .taxonomy import align_with_existing_taxonomy, normalize_categories, scan_existing_taxonomy, technical_destination


class SmartFoldersPipeline:
    def __init__(self, config: AppConfig, logger: logging.Logger, api_logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.api_logger = api_logger
        self.cache = ClassificationCache(config.data_dir / "file_cache.pkl")
        self.db = Database(config.data_dir / "file_organization.db")

    def _ingest_documents(self, run_id: str) -> List[DocumentEnvelope]:
        docs: List[DocumentEnvelope] = []
        if not self.config.inbox_dir.exists():
            return docs
        for file_path in sorted(self.config.inbox_dir.iterdir()):
            if not file_path.is_file() or file_path.name.startswith("."):
                continue
            docs.append(self._build_envelope(run_id, file_path))
        return docs

    def _build_envelope(self, run_id: str, file_path: Path) -> DocumentEnvelope:
        try:
            size = file_path.stat().st_size
        except Exception:
            size = None
        return DocumentEnvelope(
            document_id=uuid.uuid4().hex,
            run_id=run_id,
            source_path=file_path,
            filename=file_path.name,
            extension=file_path.suffix.lower(),
            file_size=size,
            mime_type=sniff_mime_type(file_path),
        )

    def _apply_cached_result(self, envelope: DocumentEnvelope) -> bool:
        cached = self.cache.get(envelope.source_path, self.config)
        if not cached:
            return False
        envelope.file_hash = cached.get("file_hash")
        envelope.summary = cached.get("summary")
        envelope.keywords = cached.get("keywords", [])
        envelope.language = cached.get("language")
        envelope.document_type = cached.get("document_type")
        envelope.category_l1 = cached.get("category_l1")
        envelope.category_l2 = cached.get("category_l2")
        envelope.confidence = float(cached.get("confidence", 0.0) or 0.0)
        envelope.reason = cached.get("reason")
        envelope.needs_review = bool(cached.get("needs_review", False))
        envelope.cached = True
        envelope.status = "classified"
        return True

    def _extract_stage(self, envelope: DocumentEnvelope) -> str:
        self.logger.info("Run %s extract %s", envelope.run_id, envelope.filename)
        file_hash = self.cache.file_hash(envelope.source_path)
        envelope.file_hash = file_hash
        if file_hash:
            existing = self.db.find_existing_by_hash(file_hash)
            if existing is not None:
                envelope.category_l1 = "_Duplicates"
                envelope.category_l2 = existing["category_l2"] or "KnownDuplicate"
                envelope.confidence = 1.0
                envelope.reason = f"Duplicate of {existing['filename']}"
                envelope.needs_review = False
                envelope.status = "duplicate"
                return "act"

        if self._apply_cached_result(envelope):
            envelope.metadata["decision_source"] = "cache"
            return "act"

        extraction = extract(envelope.source_path)
        envelope.extracted_text = extraction.extracted_text
        envelope.metadata.update(extraction.metadata)
        envelope.extraction_quality = extraction.extraction_quality
        envelope.ocr_used = extraction.ocr_used
        envelope.conversion_used = extraction.conversion_used
        envelope.errors.extend(extraction.errors)
        envelope.status = "extracted"
        envelope.metadata["decision_source"] = envelope.metadata.get("decision_source", "extractor")

        if not envelope.extracted_text:
            envelope.needs_review = True
            envelope.reason = "Extraction produced no usable text"
            envelope.status = "review"
            return "act"
        return "llm"

    def _llm_stage(self, envelope: DocumentEnvelope, existing_taxonomy: Dict[str, List[str]]) -> None:
        self.logger.info("Run %s llm %s", envelope.run_id, envelope.filename)
        understanding = understand_document(
            self.config,
            envelope.filename,
            str(envelope.source_path),
            envelope.extracted_text or "",
        )
        envelope.summary = understanding.summary
        envelope.keywords = understanding.keywords
        envelope.language = understanding.language
        envelope.document_type = understanding.document_type
        envelope.understanding_model = understanding.model_name
        envelope.prompt_version = understanding.prompt_version
        envelope.status = "understood"

        classification = classify_document(
            self.config,
            envelope.filename,
            str(envelope.source_path),
            envelope.mime_type,
            understanding,
            envelope.extracted_text or "",
            existing_taxonomy,
        )
        self._review_classification(envelope, classification)
        envelope.status = "classified"
        envelope.metadata["decision_source"] = "llm"
        self.cache.set(
            envelope.source_path,
            self.config,
            {
                "summary": envelope.summary,
                "keywords": envelope.keywords,
                "language": envelope.language,
                "document_type": envelope.document_type,
                "category_l1": envelope.category_l1,
                "category_l2": envelope.category_l2,
                "confidence": envelope.confidence,
                "reason": envelope.reason,
                "needs_review": envelope.needs_review,
            },
        )

    def _review_classification(self, envelope: DocumentEnvelope, classification: ClassificationResult) -> None:
        envelope.category_l1 = classification.category_l1
        envelope.category_l2 = classification.category_l2
        envelope.confidence = classification.confidence
        envelope.reason = classification.reason
        envelope.needs_review = classification.needs_review
        envelope.classification_model = classification.model_name
        envelope.prompt_version = classification.prompt_version
        if envelope.confidence < self.config.thresholds.review_confidence:
            envelope.needs_review = True
            if envelope.reason:
                envelope.reason += " | Below review threshold"
            else:
                envelope.reason = "Below review threshold"

    def _act(self, envelope: DocumentEnvelope, dry_run: bool, existing_taxonomy: Dict[str, List[str]]) -> None:
        if envelope.category_l1 == "_Duplicates":
            destination = technical_destination(self.config, "_Duplicates", envelope.filename)
        elif envelope.needs_review:
            destination = technical_destination(self.config, "_NeedsReview", envelope.filename)
        elif not envelope.extracted_text:
            destination = technical_destination(self.config, "_FailedExtraction", envelope.filename)
        elif not envelope.category_l1 or not envelope.category_l2:
            destination = technical_destination(self.config, "_NeedsReview", envelope.filename)
        else:
            level1, level2 = normalize_categories(self.config, envelope.category_l1, envelope.category_l2)
            level1, level2 = align_with_existing_taxonomy(existing_taxonomy, level1, level2)
            envelope.category_l1 = level1
            envelope.category_l2 = level2
            destination = self.config.organized_dir / level1 / level2 / envelope.filename

        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            stem = destination.stem
            suffix = destination.suffix
            counter = 1
            while True:
                candidate = destination.with_name(f"{stem}_{counter}{suffix}")
                if not candidate.exists():
                    destination = candidate
                    break
                counter += 1

        envelope.destination_path = destination
        if dry_run:
            envelope.status = "planned"
            return

        shutil.move(str(envelope.source_path), str(destination))
        envelope.status = "moved"

    def _document_report(self, envelope: DocumentEnvelope) -> Dict[str, object]:
        if envelope.category_l1 == "_Duplicates":
            source = "duplicate"
        else:
            source = str(envelope.metadata.get("decision_source", "unknown"))
        return {
            "document_id": envelope.document_id,
            "filename": envelope.filename,
            "source_path": str(envelope.source_path),
            "status": envelope.status,
            "decision_source": source,
            "category_l1": envelope.category_l1,
            "category_l2": envelope.category_l2,
            "confidence": envelope.confidence,
            "needs_review": envelope.needs_review,
            "reason": envelope.reason,
            "destination_path": str(envelope.destination_path) if envelope.destination_path else None,
            "mime_type": envelope.mime_type,
            "document_type": envelope.document_type,
            "cached": envelope.cached,
            "ocr_used": envelope.ocr_used,
            "conversion_used": envelope.conversion_used,
            "errors": list(envelope.errors),
        }

    def _run_documents(self, docs: Sequence[DocumentEnvelope], dry_run: bool) -> List[DocumentEnvelope]:
        existing_taxonomy = scan_existing_taxonomy(self.config)
        results: List[DocumentEnvelope] = []
        results_lock = threading.Lock()

        extract_queue: "queue.Queue[Optional[DocumentEnvelope]]" = queue.Queue()
        llm_queue: "queue.Queue[Optional[DocumentEnvelope]]" = queue.Queue()
        act_queue: "queue.Queue[Optional[DocumentEnvelope]]" = queue.Queue()

        extract_workers = max(1, min(self.config.max_workers, self.config.workers.extract or self.config.workers.io))
        llm_workers = max(1, min(self.config.max_workers, self.config.workers.llm))
        act_workers = max(1, min(self.config.max_workers, self.config.workers.act or self.config.workers.io))

        def extract_worker() -> None:
            while True:
                envelope = extract_queue.get()
                if envelope is None:
                    extract_queue.task_done()
                    break
                try:
                    next_stage = self._extract_stage(envelope)
                    if next_stage == "llm":
                        llm_queue.put(envelope)
                    else:
                        act_queue.put(envelope)
                except Exception as exc:
                    envelope.errors.append(str(exc))
                    envelope.status = "failed"
                    act_queue.put(envelope)
                finally:
                    extract_queue.task_done()

        def llm_worker() -> None:
            while True:
                envelope = llm_queue.get()
                if envelope is None:
                    llm_queue.task_done()
                    break
                try:
                    self._llm_stage(envelope, existing_taxonomy)
                except Exception as exc:
                    envelope.errors.append(str(exc))
                    envelope.status = "failed"
                finally:
                    act_queue.put(envelope)
                    llm_queue.task_done()

        def act_worker() -> None:
            while True:
                envelope = act_queue.get()
                if envelope is None:
                    act_queue.task_done()
                    break
                try:
                    if envelope.status != "failed":
                        self._act(envelope, dry_run=dry_run, existing_taxonomy=existing_taxonomy)
                except Exception as exc:
                    envelope.errors.append(str(exc))
                    envelope.status = "failed"
                finally:
                    self.db.log_document(envelope, dry_run=dry_run)
                    with results_lock:
                        results.append(envelope)
                    act_queue.task_done()

        extract_threads = [
            threading.Thread(target=extract_worker, daemon=True, name=f"extract-worker-{idx}")
            for idx in range(extract_workers)
        ]
        llm_threads = [
            threading.Thread(target=llm_worker, daemon=True, name=f"llm-worker-{idx}")
            for idx in range(llm_workers)
        ]
        act_threads = [
            threading.Thread(target=act_worker, daemon=True, name=f"act-worker-{idx}")
            for idx in range(act_workers)
        ]

        for thread in [*extract_threads, *llm_threads, *act_threads]:
            thread.start()

        for doc in docs:
            extract_queue.put(doc)
        for _ in extract_threads:
            extract_queue.put(None)

        extract_queue.join()
        for _ in llm_threads:
            llm_queue.put(None)
        llm_queue.join()
        for _ in act_threads:
            act_queue.put(None)
        act_queue.join()

        for thread in [*extract_threads, *llm_threads, *act_threads]:
            thread.join(timeout=1.0)

        return results

    def run(self, dry_run: bool = False, limit: Optional[int] = None) -> RunMetrics:
        start = time.perf_counter()
        run_id = uuid.uuid4().hex
        self.db.create_run(run_id, dry_run=dry_run)
        docs = self._ingest_documents(run_id)
        if limit is not None:
            docs = docs[:limit]

        self.logger.info("=" * 60)
        self.logger.info("Run %s started", run_id)
        self.logger.info("Inbox: %s", self.config.inbox_dir)
        self.logger.info("Output: %s", self.config.organized_dir)
        self.logger.info(
            "Models: understand=%s classify=%s",
            self.config.models.understanding_model,
            self.config.models.classification_model,
        )

        if not docs:
            self.logger.info("No files found to organize.")
            metrics = RunMetrics(run_id=run_id, dry_run=dry_run, duration_seconds=0.0)
            self.db.complete_run(metrics)
            return metrics

        results = self._run_documents(docs, dry_run=dry_run)

        processed = [item for item in results if item.status in {"moved", "planned"}]
        failed = [item for item in results if item.status == "failed"]
        review = [item for item in results if item.needs_review]
        avg_confidence = (
            sum(item.confidence for item in results) / len(results) if results else 0.0
        )
        metrics = RunMetrics(
            run_id=run_id,
            total_files=len(docs),
            processed_files=len(processed),
            failed_files=len(failed),
            review_files=len(review),
            dry_run=dry_run,
            avg_confidence=avg_confidence,
            duration_seconds=time.perf_counter() - start,
            documents=[self._document_report(item) for item in sorted(results, key=lambda doc: doc.filename.lower())],
        )
        self.db.complete_run(metrics)
        self.logger.info(
            "Total files: %d, Successful: %d, Failed: %d",
            metrics.total_files,
            metrics.processed_files,
            metrics.failed_files,
        )
        self.logger.info("Total time: %.2fs", metrics.duration_seconds)
        return metrics

    def benchmark(self, dataset_path: Optional[Path] = None, sample_limit: Optional[int] = None) -> RunMetrics | BenchmarkReport:
        if dataset_path is None:
            return self.run(dry_run=True, limit=sample_limit)

        start = time.perf_counter()
        run_id = uuid.uuid4().hex
        self.db.create_run(run_id, dry_run=True)
        records = load_benchmark_dataset(dataset_path)
        if sample_limit is not None:
            records = records[:sample_limit]

        docs = [
            self._build_envelope(run_id, record.source_path.expanduser().resolve())
            for record in records
            if record.source_path.expanduser().resolve().exists()
        ]
        results = self._run_documents(docs, dry_run=True)
        report = build_benchmark_report(dataset_path, records, results)
        self.logger.info(
            "Benchmark %s finished in %.2fs with %d/%d full matches",
            dataset_path,
            time.perf_counter() - start,
            report.full_matches,
            report.total_cases,
        )
        return report

    def undo_last_run(self) -> int:
        run_id = self.db.get_latest_run(include_dry_run=False)
        if not run_id:
            return 0

        restored = 0
        for row in self.db.get_run_documents(run_id):
            source = Path(row["source_path"])
            destination = Path(row["destination_path"])
            if not destination.exists():
                continue
            source.parent.mkdir(parents=True, exist_ok=True)
            if source.exists():
                source = source.with_name(f"{source.stem}_restored{source.suffix}")
            shutil.move(str(destination), str(source))
            restored += 1
        return restored

    def close(self) -> None:
        self.db.close()


def build_pipeline(config: Optional[AppConfig] = None, logger: Optional[logging.Logger] = None, api_logger: Optional[logging.Logger] = None) -> SmartFoldersPipeline:
    from .logging_utils import setup_logging

    effective_config = config or load_config()
    effective_logger, effective_api_logger = (logger, api_logger) if logger and api_logger else setup_logging(effective_config.data_dir)
    return SmartFoldersPipeline(effective_config, effective_logger, effective_api_logger)

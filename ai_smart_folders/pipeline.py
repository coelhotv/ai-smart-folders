from __future__ import annotations

import logging
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from .config import load_config
from .extractors import extract, sniff_mime_type
from .llm import classify_document, understand_document
from .models import AppConfig, ClassificationResult, DocumentEnvelope, RunMetrics
from .storage import ClassificationCache, Database
from .taxonomy import normalize_categories, scan_existing_taxonomy, technical_destination


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
            try:
                size = file_path.stat().st_size
            except Exception:
                size = None
            docs.append(
                DocumentEnvelope(
                    document_id=uuid.uuid4().hex,
                    run_id=run_id,
                    source_path=file_path,
                    filename=file_path.name,
                    extension=file_path.suffix.lower(),
                    file_size=size,
                    mime_type=sniff_mime_type(file_path),
                )
            )
        return docs

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

    def _act(self, envelope: DocumentEnvelope, dry_run: bool) -> None:
        if envelope.needs_review:
            destination = technical_destination(self.config, "_NeedsReview", envelope.filename)
        elif not envelope.extracted_text:
            destination = technical_destination(self.config, "_FailedExtraction", envelope.filename)
        elif not envelope.category_l1 or not envelope.category_l2:
            destination = technical_destination(self.config, "_NeedsReview", envelope.filename)
        else:
            level1, level2 = normalize_categories(self.config, envelope.category_l1, envelope.category_l2)
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

    def _process_document(self, envelope: DocumentEnvelope, existing_taxonomy: Dict[str, List[str]], dry_run: bool) -> DocumentEnvelope:
        try:
            self.logger.info("Run %s processing %s", envelope.run_id, envelope.filename)
            file_hash = self.cache.file_hash(envelope.source_path)
            envelope.file_hash = file_hash

            if not self._apply_cached_result(envelope):
                extraction = extract(envelope.source_path)
                envelope.extracted_text = extraction.extracted_text
                envelope.metadata.update(extraction.metadata)
                envelope.extraction_quality = extraction.extraction_quality
                envelope.ocr_used = extraction.ocr_used
                envelope.conversion_used = extraction.conversion_used
                envelope.errors.extend(extraction.errors)
                envelope.status = "extracted"

                if not envelope.extracted_text:
                    envelope.needs_review = True
                    envelope.reason = "Extraction produced no usable text"
                    envelope.status = "review"
                else:
                    understanding = understand_document(
                        self.config,
                        envelope.filename,
                        str(envelope.source_path),
                        envelope.extracted_text,
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
                        envelope.extracted_text,
                        existing_taxonomy,
                    )
                    self._review_classification(envelope, classification)
                    envelope.status = "classified"
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

            self._act(envelope, dry_run=dry_run)
        except Exception as exc:
            envelope.errors.append(str(exc))
            envelope.status = "failed"
        return envelope

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

        existing_taxonomy = scan_existing_taxonomy(self.config)
        results: List[DocumentEnvelope] = []
        with ThreadPoolExecutor(max_workers=max(1, self.config.max_workers)) as executor:
            future_to_doc = {
                executor.submit(self._process_document, doc, existing_taxonomy, dry_run): doc for doc in docs
            }
            for future in as_completed(future_to_doc):
                envelope = future.result()
                self.db.log_document(envelope, dry_run=dry_run)
                results.append(envelope)

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

    def benchmark(self, sample_limit: Optional[int] = None) -> RunMetrics:
        return self.run(dry_run=True, limit=sample_limit)

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

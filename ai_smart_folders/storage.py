from __future__ import annotations

import hashlib
import os
import pickle
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .models import AppConfig, DocumentEnvelope, RunMetrics


class ClassificationCache:
    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.lock = threading.RLock()
        self.cache: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if not self.cache_file.exists():
            return {}
        try:
            with open(self.cache_file, "rb") as handle:
                return pickle.load(handle) or {}
        except Exception:
            return {}

    def _save(self) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.cache_file.with_suffix(f"{self.cache_file.suffix}.tmp")
        with open(temp_path, "wb") as handle:
            pickle.dump(self.cache, handle)
        os.replace(temp_path, self.cache_file)

    @staticmethod
    def file_hash(file_path: Path) -> Optional[str]:
        try:
            hasher = hashlib.md5()
            with open(file_path, "rb") as handle:
                for chunk in iter(lambda: handle.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return None

    @staticmethod
    def cache_key(file_hash: str, config: AppConfig) -> str:
        return "|".join(
            [
                file_hash,
                config.prompt_version,
                config.models.understanding_model,
                config.models.classification_model,
            ]
        )

    def get(self, file_path: Path, config: AppConfig) -> Optional[Dict[str, Any]]:
        file_hash = self.file_hash(file_path)
        if not file_hash:
            return None
        with self.lock:
            return self.cache.get(self.cache_key(file_hash, config))

    def set(self, file_path: Path, config: AppConfig, payload: Dict[str, Any]) -> Optional[str]:
        file_hash = self.file_hash(file_path)
        if not file_hash:
            return None
        with self.lock:
            self.cache[self.cache_key(file_hash, config)] = {
                **payload,
                "file_hash": file_hash,
                "cached_at": datetime.utcnow().isoformat(),
            }
            self._save()
        return file_hash


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    dry_run INTEGER DEFAULT 0,
                    total_files INTEGER DEFAULT 0,
                    processed_files INTEGER DEFAULT 0,
                    failed_files INTEGER DEFAULT 0,
                    review_files INTEGER DEFAULT 0,
                    avg_confidence REAL DEFAULT 0
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS file_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    destination_path TEXT,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    category_l1 TEXT,
                    category_l2 TEXT,
                    confidence REAL,
                    reason TEXT,
                    needs_review INTEGER DEFAULT 0,
                    file_hash TEXT,
                    prompt_version TEXT,
                    understanding_model TEXT,
                    classification_model TEXT,
                    ocr_used INTEGER DEFAULT 0,
                    conversion_used INTEGER DEFAULT 0,
                    dry_run INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_file_events_run_id ON file_events(run_id);")
            self.conn.commit()

    def create_run(self, run_id: str, dry_run: bool) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, dry_run) VALUES (?, ?)",
                (run_id, int(bool(dry_run))),
            )
            self.conn.commit()

    def complete_run(self, metrics: RunMetrics) -> None:
        with self.lock:
            self.conn.execute(
                """
                UPDATE runs
                SET completed_at = CURRENT_TIMESTAMP,
                    total_files = ?,
                    processed_files = ?,
                    failed_files = ?,
                    review_files = ?,
                    avg_confidence = ?
                WHERE run_id = ?
                """,
                (
                    metrics.total_files,
                    metrics.processed_files,
                    metrics.failed_files,
                    metrics.review_files,
                    metrics.avg_confidence,
                    metrics.run_id,
                ),
            )
            self.conn.commit()

    def log_document(self, envelope: DocumentEnvelope, dry_run: bool) -> None:
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO file_events (
                    run_id, document_id, source_path, destination_path, filename, status,
                    category_l1, category_l2, confidence, reason, needs_review, file_hash,
                    prompt_version, understanding_model, classification_model, ocr_used,
                    conversion_used, dry_run, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope.run_id,
                    envelope.document_id,
                    str(envelope.source_path),
                    str(envelope.destination_path) if envelope.destination_path else None,
                    envelope.filename,
                    envelope.status,
                    envelope.category_l1,
                    envelope.category_l2,
                    envelope.confidence,
                    envelope.reason,
                    int(bool(envelope.needs_review)),
                    envelope.file_hash,
                    envelope.prompt_version,
                    envelope.understanding_model,
                    envelope.classification_model,
                    int(bool(envelope.ocr_used)),
                    int(bool(envelope.conversion_used)),
                    int(bool(dry_run)),
                    "\n".join(envelope.errors) if envelope.errors else None,
                ),
            )
            self.conn.commit()

    def get_latest_run(self, include_dry_run: bool = False) -> Optional[str]:
        with self.lock:
            query = "SELECT run_id FROM runs"
            params: Tuple[Any, ...] = ()
            if not include_dry_run:
                query += " WHERE dry_run = 0"
            query += " ORDER BY started_at DESC LIMIT 1"
            row = self.conn.execute(query, params).fetchone()
            return row["run_id"] if row else None

    def get_run_documents(self, run_id: str) -> List[sqlite3.Row]:
        with self.lock:
            return list(
                self.conn.execute(
                    """
                    SELECT * FROM file_events
                    WHERE run_id = ? AND destination_path IS NOT NULL AND dry_run = 0
                    ORDER BY id DESC
                    """,
                    (run_id,),
                ).fetchall()
            )

    def close(self) -> None:
        with self.lock:
            self.conn.close()

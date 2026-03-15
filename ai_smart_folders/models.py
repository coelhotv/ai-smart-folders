from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ModelSettings(BaseModel):
    router_model: str = "gemma3:270m"
    understanding_model: str = "gemma3:1b"
    classification_model: str = "gemma3:1b"
    fallback_model: Optional[str] = None


class ThresholdSettings(BaseModel):
    review_confidence: float = 0.55


class WorkerSettings(BaseModel):
    io: int = 4
    llm: int = 1


class TaxonomyConfig(BaseModel):
    level1_default: str = "General"
    level2_default: str = "Unsorted"
    aliases: Dict[str, str] = Field(default_factory=dict)
    level1_aliases: Dict[str, str] = Field(default_factory=dict)
    level2_aliases: Dict[str, str] = Field(default_factory=dict)
    technical_folders: List[str] = Field(
        default_factory=lambda: [
            "_NeedsReview",
            "_Unsupported",
            "_Duplicates",
            "_FailedExtraction",
        ]
    )


class AppConfig(BaseModel):
    inbox_dir: Path = Field(default_factory=lambda: Path.home() / "Dropbox" / "_courses")
    organized_dir: Path = Field(default_factory=lambda: Path.home() / "OrganizedFiles")
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".ai-smart-folders-data")
    ignore_dirs: List[str] = Field(default_factory=lambda: ["_Unprocessed", "_Ignored"])
    max_content_length: int = 20000
    max_workers: int = 4
    dry_run: bool = False
    prompt_version: str = "v3"
    models: ModelSettings = Field(default_factory=ModelSettings)
    thresholds: ThresholdSettings = Field(default_factory=ThresholdSettings)
    workers: WorkerSettings = Field(default_factory=WorkerSettings)
    taxonomy: TaxonomyConfig = Field(default_factory=TaxonomyConfig)


class ExtractionResult(BaseModel):
    extracted_text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    extraction_quality: float = 0.0
    ocr_used: bool = False
    conversion_used: bool = False
    extractor_name: Optional[str] = None
    errors: List[str] = Field(default_factory=list)


class UnderstandingResult(BaseModel):
    summary: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    language: Optional[str] = None
    document_type: Optional[str] = None
    content_signals: List[str] = Field(default_factory=list)
    model_name: Optional[str] = None
    prompt_version: Optional[str] = None


class ClassificationResult(BaseModel):
    category_l1: Optional[str] = None
    category_l2: Optional[str] = None
    confidence: float = 0.0
    reason: Optional[str] = None
    needs_review: bool = False
    model_name: Optional[str] = None
    prompt_version: Optional[str] = None


class DocumentEnvelope(BaseModel):
    document_id: str
    run_id: str
    source_path: Path
    filename: str
    extension: str
    mime_type: Optional[str] = None
    file_hash: Optional[str] = None
    file_size: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    extracted_text: Optional[str] = None
    extraction_quality: float = 0.0
    ocr_used: bool = False
    conversion_used: bool = False
    summary: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    language: Optional[str] = None
    document_type: Optional[str] = None
    category_l1: Optional[str] = None
    category_l2: Optional[str] = None
    confidence: float = 0.0
    reason: Optional[str] = None
    needs_review: bool = False
    destination_path: Optional[Path] = None
    status: str = "ingested"
    errors: List[str] = Field(default_factory=list)
    cached: bool = False
    understanding_model: Optional[str] = None
    classification_model: Optional[str] = None
    prompt_version: Optional[str] = None


class RunMetrics(BaseModel):
    run_id: str
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    review_files: int = 0
    dry_run: bool = False
    avg_confidence: float = 0.0
    duration_seconds: float = 0.0


class BenchmarkRecord(BaseModel):
    source_path: Path
    expected_category_l1: Optional[str] = None
    expected_category_l2: Optional[str] = None
    expected_needs_review: Optional[bool] = None


class BenchmarkReport(BaseModel):
    dataset_path: Path
    total_cases: int = 0
    matched_level1: int = 0
    matched_level2: int = 0
    matched_review_flag: int = 0
    full_matches: int = 0
    failures: int = 0
    cases: List[Dict[str, Any]] = Field(default_factory=list)

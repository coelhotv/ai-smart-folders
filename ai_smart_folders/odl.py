"""
odl.py — OpenDataLoader PDF wrapper for ai-smart-folders.

Provides a single batch_extract() call that converts all PDFs at once,
amortising the JVM startup cost across the entire inbox run.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------


def is_available() -> bool:
    """Return True if opendataloader-pdf is importable or on PATH."""
    try:
        import opendataloader_pdf  # noqa: F401

        return True
    except Exception:
        pass
    return shutil.which("opendataloader-pdf") is not None


def is_hybrid_available() -> bool:
    """Return True if the hybrid OCR backend CLI is on PATH."""
    return shutil.which("opendataloader-pdf-hybrid") is not None


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------


def batch_extract(
    pdf_paths: List[Path],
    timeout: int = 180,
    use_hybrid: bool = False,
    hybrid_url: str = "http://localhost:5002",
    hybrid_timeout: int = 120_000,
    hybrid_fallback: bool = True,
    reading_order: str = "xycut",
    ocr_lang: Optional[str] = None,
    pages: Optional[str] = None,
) -> Dict[Path, Optional[str]]:
    """
    Convert *all* PDFs in one JVM call and return extracted Markdown text.

    Parameters
    ----------
    pdf_paths:
        List of absolute paths to PDF files.
    timeout:
        Seconds to wait for the JVM process to complete. Default 180 s.
    use_hybrid:
        If True, route complex/scanned pages to the hybrid AI backend.
    hybrid_url:
        URL of the running opendataloader-pdf-hybrid server.
    hybrid_timeout:
        Milliseconds before a page request to the backend times out.
    hybrid_fallback:
        Fall back to Java-only processing if the backend is unreachable.
    reading_order:
        Reading-order algorithm passed to opendataloader (``"xycut"`` | ``"off"``).
    ocr_lang:
        Comma-separated OCR language codes, e.g. ``"pt,en"``. Only relevant
        when *use_hybrid* is True and the backend was started with ``--force-ocr``.

    Returns
    -------
    dict
        ``{original_pdf_path: extracted_markdown_text_or_None}``
    """
    if not pdf_paths:
        return {}

    try:
        import opendataloader_pdf
    except ImportError:
        log.warning("opendataloader-pdf is not installed; skipping batch extraction")
        return {p: None for p in pdf_paths}

    results: Dict[Path, Optional[str]] = {}

    with tempfile.TemporaryDirectory(prefix="ai-smart-odl-") as tmp_str:
        tmp_dir = Path(tmp_str)
        input_paths = [str(p) for p in pdf_paths]

        convert_kwargs: Dict = {
            "input_path": input_paths,
            "output_dir": str(tmp_dir),
            "format": "markdown",
            "reading_order": reading_order,
            "quiet": True,
            "sanitize": True,  # strip potentially injected content
        }

        if pages:
            convert_kwargs["pages"] = pages

        if use_hybrid:
            convert_kwargs["hybrid"] = "docling-fast"
            convert_kwargs["hybrid_url"] = hybrid_url
            convert_kwargs["hybrid_timeout"] = str(hybrid_timeout)
            convert_kwargs["hybrid_fallback"] = hybrid_fallback

        log.info(
            "ODL batch: converting %d PDF(s) [hybrid=%s]",
            len(pdf_paths),
            use_hybrid,
        )

        try:
            opendataloader_pdf.convert(**convert_kwargs)
        except Exception as exc:
            log.warning("ODL batch extraction failed: %s", exc)
            return {p: None for p in pdf_paths}

        # Map each original PDF to its generated .md file
        for pdf_path in pdf_paths:
            md_file = tmp_dir / f"{pdf_path.stem}.md"
            if md_file.exists():
                try:
                    text = md_file.read_text(encoding="utf-8").strip()
                    results[pdf_path] = text if text else None
                except Exception as exc:
                    log.warning("ODL: failed reading output for %s: %s", pdf_path.name, exc)
                    results[pdf_path] = None
            else:
                log.debug("ODL: no output file found for %s", pdf_path.name)
                results[pdf_path] = None

    return results


# ---------------------------------------------------------------------------
# Helper used by pipeline to build an ExtractionResult from precomputed text
# ---------------------------------------------------------------------------


def _quality_from_text(text: Optional[str]) -> float:
    """Mirror of extractors._quality_from_text (avoid circular import)."""
    if not text:
        return 0.0
    size = len(text.strip())
    if size >= 4000:
        return 0.95
    if size >= 1000:
        return 0.8
    if size >= 250:
        return 0.6
    if size >= 50:
        return 0.35
    return 0.15

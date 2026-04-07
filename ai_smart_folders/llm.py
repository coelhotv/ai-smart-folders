from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from .models import AppConfig, ClassificationResult, UnderstandingResult

try:
    import ollama
except Exception:
    ollama = None


def _smart_preview(text: str, max_chars: int) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= max_chars:
        return normalized
    head_len = int(max_chars * 0.6)
    tail_len = max_chars - head_len
    return normalized[:head_len] + " ... [CONTENT SKIPPED] ... " + normalized[-tail_len:]


def _parse_json_candidate(raw: Any) -> Dict[str, Any]:
    # Handle the new Ollama library objects vs old dicts
    if hasattr(raw, "message") and hasattr(raw.message, "content"):
        raw = raw.message.content
    elif isinstance(raw, dict):
        if "message" in raw and isinstance(raw["message"], dict):
            raw = raw["message"].get("content", raw)
        elif "content" in raw:
            raw = raw["content"]

    if not isinstance(raw, str):
        return raw if isinstance(raw, dict) else {}

    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}


def _chat_json(model: str, prompt: str, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if ollama is None:
        return {}
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        format=schema or "json",
        think=False,
        options={"num_ctx": 16384}
    )
    content = response.get("message", {}).get("content", "")
    if not content and hasattr(response, "message"):
        content = response.message.content
    print(f"DEBUG: model={model} content={content}")
    return _parse_json_candidate(response)


def _heuristic_keywords(text: str) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text.lower())
    stop = {"this", "that", "with", "from", "have", "your", "para", "como", "mais", "will", "they"}
    freq: Dict[str, int] = {}
    for word in words:
        if word in stop:
            continue
        freq[word] = freq.get(word, 0) + 1
    return [item[0] for item in sorted(freq.items(), key=lambda pair: (-pair[1], pair[0]))[:8]]


def deterministic_classification(filename: str, mime_type: Optional[str], text: str) -> Optional[ClassificationResult]:
    lower_name = filename.lower()
    combined = f"{lower_name}\n{text[:1000].lower()}"
    rules = [
        (r"invoice|receipt|bill|nota fiscal|boleto", ("Finance", "Invoices", 0.92, "Matched finance rule")),
        (r"resume|curriculum|cv\b", ("Career", "Resumes", 0.95, "Matched resume rule")),
        (r"slide|presentation|deck", ("Knowledge", "Presentations", 0.88, "Matched presentation rule")),
        (r"\.py\b|def |class |import ", ("Engineering", "Code", 0.84, "Matched code rule")),
    ]
    for pattern, (level1, level2, confidence, reason) in rules:
        if re.search(pattern, combined, flags=re.IGNORECASE):
            return ClassificationResult(
                category_l1=level1,
                category_l2=level2,
                confidence=confidence,
                reason=reason,
                needs_review=False,
            )
    return None


def understand_document(config: AppConfig, filename: str, file_path: str, text: str) -> UnderstandingResult:
    preview = _smart_preview(text, config.max_content_length)
    prompt = f"""
You are a file understanding assistant.
Return only JSON with keys: summary, keywords, language, document_type, content_signals.

Filename: "{filename}"
Path: "{file_path}"
ContentSample: "{preview}"
"""
    payload = _chat_json(config.models.understanding_model, prompt, schema=UnderstandingResult.model_json_schema())
    if payload:
        return UnderstandingResult(
            summary=payload.get("summary"),
            keywords=[str(item) for item in payload.get("keywords", [])][:8],
            language=payload.get("language"),
            document_type=payload.get("document_type"),
            content_signals=[str(item) for item in payload.get("content_signals", [])][:8],
            model_name=config.models.understanding_model,
            prompt_version=config.prompt_version,
        )

    keywords = _heuristic_keywords(text)
    summary = (text.strip()[:240] + "...") if len(text.strip()) > 240 else text.strip()
    language = "pt-BR" if re.search(r"\b(nao|para|uma|documento|arquivo)\b", text.lower()) else "en"
    document_type = "document"
    return UnderstandingResult(
        summary=summary or None,
        keywords=keywords,
        language=language,
        document_type=document_type,
        content_signals=keywords[:5],
        model_name=None,
        prompt_version=config.prompt_version,
    )


def classify_document(
    config: AppConfig,
    filename: str,
    file_path: str,
    mime_type: Optional[str],
    understanding: UnderstandingResult,
    text: str,
    existing_taxonomy: Dict[str, List[str]],
) -> ClassificationResult:
    deterministic = deterministic_classification(filename, mime_type, text)
    if deterministic:
        deterministic.prompt_version = config.prompt_version
        return deterministic

    preview = _smart_preview(text, config.max_content_length)
    prompt = f"""
You are a file organization classification system.
Your goal is to categorize the document into two levels: category_l1 (broad) and category_l2 (specific).
CRITICAL RULES:
1. Base your classification heavily on the provided Summary, Keywords, and ContentSample. The content represents the file, NOT the extension or the existing taxonomy.
2. An Existing Taxonomy is provided below. You MAY reuse an existing category_l1/category_l2 ONLY IF it is a highly accurate semantic match (e.g., >80% relevance) to the document's content.
3. DO NOT force fit a document into the Existing Taxonomy if the topics don't align. It is strictly better to invent a new, highly accurate category than to reuse a mismatched one.

Existing taxonomy: {existing_taxonomy}
Filename: "{filename}"
Path: "{file_path}"
Summary: "{understanding.summary or ''}"
Keywords: {understanding.keywords}
DocumentType: "{understanding.document_type or ''}"
Language: "{understanding.language or ''}"
ContentSample: "{preview}"

Return only JSON matching the schema parameters: category_l1, category_l2, confidence, reason, needs_review.
"""
    payload = _chat_json(config.models.classification_model, prompt, schema=ClassificationResult.model_json_schema())
    if payload:
        return ClassificationResult(
            category_l1=payload.get("category_l1"),
            category_l2=payload.get("category_l2"),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            reason=payload.get("reason"),
            needs_review=bool(payload.get("needs_review", False)),
            model_name=config.models.classification_model,
            prompt_version=config.prompt_version,
        )

    fallback_l1 = "General"
    fallback_l2 = understanding.document_type or "Unsorted"
    confidence = 0.4 if understanding.summary else 0.2
    return ClassificationResult(
        category_l1=fallback_l1,
        category_l2=fallback_l2,
        confidence=confidence,
        reason="Fallback heuristic classification",
        needs_review=confidence < config.thresholds.review_confidence,
        model_name=None,
        prompt_version=config.prompt_version,
    )

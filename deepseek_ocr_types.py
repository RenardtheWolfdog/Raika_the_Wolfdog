from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class PdfOcrResult:
    """DeepSeek-OCR 처리 결과를 표현하는 데이터 클래스."""

    full_text: str
    page_texts: List[str]
    page_count: int
    file_hash: str
    meta: Dict[str, Optional[str]]


def serialize_pdf_ocr_result(result: PdfOcrResult) -> Dict[str, Any]:
    """
    PdfOcrResult를 JSON 직렬화 가능한 dict로 변환합니다.
    """
    return {
        "full_text": result.full_text,
        "page_texts": result.page_texts,
        "page_count": result.page_count,
        "file_hash": result.file_hash,
        "meta": result.meta,
    }


def deserialize_pdf_ocr_result(
    payload: Mapping[str, Any], default_hash: Optional[str] = None
) -> PdfOcrResult:
    """
    dict payload로부터 PdfOcrResult를 복원합니다.
    """
    page_texts = list(payload.get("page_texts", []) or [])
    file_hash = payload.get("file_hash") or default_hash or ""
    return PdfOcrResult(
        full_text=payload.get("full_text", ""),
        page_texts=page_texts,
        page_count=int(payload.get("page_count", len(page_texts))),
        file_hash=file_hash,
        meta=dict(payload.get("meta", {}) or {}),
    )


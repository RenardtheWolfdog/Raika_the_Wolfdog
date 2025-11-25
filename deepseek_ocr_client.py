from __future__ import annotations

import base64
import bz2
import gzip
import logging
import os
import re
import zlib
from typing import Any, Optional

import httpx

from deepseek_ocr_types import PdfOcrResult, deserialize_pdf_ocr_result

_LOGGER = logging.getLogger(__name__)
_SERVER_URL = os.environ.get("DEEPSEEK_OCR_SERVER_URL", "").strip()
_DEFAULT_TIMEOUT = float(os.environ.get("DEEPSEEK_OCR_TIMEOUT_SECONDS", "600"))


def is_remote_available() -> bool:
    """DeepSeek OCR 전용 서버 URL이 설정되어 있는지 확인합니다."""
    return bool(_SERVER_URL)


def _build_endpoint(path: str) -> str:
    if not _SERVER_URL:
        raise RuntimeError("DEEPSEEK_OCR_SERVER_URL 환경 변수가 설정되어 있지 않습니다.")
    return f"{_SERVER_URL.rstrip('/')}{path}"


def _raise_for_status(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text
        raise RuntimeError(f"DeepSeek OCR 서버 응답 오류({exc.response.status_code}): {detail}") from exc


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]")
_BASE64_ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=-_")
_BINARY_PREFIXES = (
    b"\x89PNG",  # PNG
    b"%PDF",  # PDF
    b"PK\x03\x04",  # ZIP/Office
    b"\xff\xd8\xff",  # JPEG
    b"GIF8",  # GIF
    b"\x00\x00\x00\x18ftyp",  # MP4
)


def _is_likely_text_char(ch: str) -> bool:
    if ch.isspace():
        return True
    code = ord(ch)
    if 0x20 <= code <= 0x7E:
        return True  # Basic ASCII
    if 0xA0 <= code <= 0x17FF:
        return True  # Latin-1 Supplement, Latin Extended, etc.
    if 0x2000 <= code <= 0x2BFF:
        return True  # General punctuation, symbols
    if 0x3000 <= code <= 0x30FF:
        return True  # CJK punctuation, Katakana
    if 0x4E00 <= code <= 0x9FFF:
        return True  # CJK Unified Ideographs
    if 0xAC00 <= code <= 0xD7A3:
        return True  # Hangul
    return False


def _looks_like_text(text: str) -> bool:
    if not text:
        return False
    total = len(text)
    allowed = sum(1 for ch in text if _is_likely_text_char(ch))
    if allowed / total < 0.85:
        return False

    whitespace = sum(1 for ch in text if ch.isspace())
    if total > 200:
        cjk_ratio = sum(
            1 for ch in text if 0x4E00 <= ord(ch) <= 0x9FFF or 0xAC00 <= ord(ch) <= 0xD7A3
        ) / total
        if whitespace / total < 0.015 and cjk_ratio < 0.3:
            return False
    return True


def _looks_like_binary_bytes(raw: bytes) -> bool:
    if not raw:
        return False
    if any(raw.startswith(prefix) for prefix in _BINARY_PREFIXES):
        return True
    non_text = sum(1 for b in raw if b < 9 or 13 < b < 32 or b == 127)
    return (non_text / len(raw)) > 0.2


def _decode_binary_to_text(data: bytes) -> Optional[str]:
    if not data:
        return ""

    def _attempt(decoder) -> Optional[str]:
        try:
            candidate = decoder(data)
        except Exception:
            return None
        candidate = candidate.strip()
        if _looks_like_text(candidate):
            return candidate
        return None

    strategies = (
        lambda d: d.decode("utf-8"),
        lambda d: d.decode("utf-8", errors="replace"),
        lambda d: gzip.decompress(d).decode("utf-8"),
        lambda d: zlib.decompress(d).decode("utf-8"),
        lambda d: bz2.decompress(d).decode("utf-8"),
    )

    for strategy in strategies:
        decoded = _attempt(strategy)
        if decoded is not None:
            return decoded
    return None


def _maybe_decode_base64_text(text: str, logger: Optional[logging.Logger], context: str) -> tuple[Optional[str], bool]:
    stripped = "".join(text.split())
    if len(stripped) < 32:
        return None, False

    if any(ch not in _BASE64_ALPHABET for ch in stripped):
        return None, False

    padding = (-len(stripped)) % 4
    if padding:
        stripped += "=" * padding

    decode_attempts = (
        {"altchars": None},
        {"altchars": b"-_"},
    )
    raw = None
    for attempt in decode_attempts:
        try:
            raw = base64.b64decode(stripped, validate=True, **{k: v for k, v in attempt.items() if v is not None})
            break
        except Exception:
            continue

    if raw is None:
        return None, False

    decoded = _decode_binary_to_text(raw)
    if decoded is not None:
        if logger:
            logger.info(f"[OCR Client] base64 텍스트를 디코딩했습니다 ({context}, 길이={len(decoded)})")
        return decoded, False

    looks_binary = _looks_like_binary_bytes(raw)
    if looks_binary and logger:
        logger.warning(f"[OCR Client] base64 디코딩 결과가 바이너리로 추정됩니다 ({context})")
    return None, looks_binary


def _looks_like_binary_string(text: str) -> bool:
    if not text:
        return False
    if "\x00" in text:
        return True
    total = len(text)
    disallowed = sum(1 for ch in text if not _is_likely_text_char(ch))
    return (disallowed / total) > 0.15


def _sanitize_ocr_text(
    raw: Optional[str],
    logger: Optional[logging.Logger],
    context: str,
) -> tuple[str, bool]:
    if not raw:
        return "", False

    text = raw.strip()
    if not text:
        return "", False

    # Remove control characters and replacement chars
    cleaned = _CONTROL_CHAR_RE.sub("", text).replace("\ufffd", "").replace("�", "")

    if not cleaned:
        if logger:
            logger.warning(f"[OCR Client] 컨트롤 문자 제거 후 텍스트가 비었습니다 ({context})")
        return "", True

    decoded_text, binary_from_b64 = _maybe_decode_base64_text(cleaned, logger, context)
    if decoded_text is not None:
        cleaned = decoded_text
    elif binary_from_b64:
        return "", True

    if _looks_like_binary_string(cleaned):
        if logger:
            logger.warning(f"[OCR Client] 텍스트가 바이너리 데이터로 추정되어 무시합니다 ({context})")
        return "", True

    if len(cleaned) >= 120:
        if not _looks_like_text(cleaned):
            if logger:
                logger.warning(f"[OCR Client] 텍스트 패턴이 비정상적으로 감지되었습니다 ({context})")
            return "", True

        base64_like_count = sum(
            1 for ch in cleaned if ch in _BASE64_ALPHABET or ch in "\n\r"
        )
        if base64_like_count / len(cleaned) >= 0.94:
            if logger:
                logger.warning(f"[OCR Client] base64 패턴으로 추정되는 텍스트를 제거합니다 ({context})")
            return "", True

    return cleaned, False


def _postprocess_ocr_result(
    result: PdfOcrResult,
    logger: Optional[logging.Logger],
    context: str,
) -> PdfOcrResult:
    if not result:
        return result

    meta = dict(result.meta or {})

    sanitized_full, was_binary_full = _sanitize_ocr_text(result.full_text, logger, f"{context}:full_text")
    if was_binary_full:
        meta["sanitized_full_text"] = "1"
    result.full_text = sanitized_full

    sanitized_pages = []
    binary_pages = 0
    for idx, page in enumerate(result.page_texts or [], 1):
        sanitized_page, was_binary_page = _sanitize_ocr_text(page, logger, f"{context}:page_{idx}")
        if was_binary_page:
            binary_pages += 1
        sanitized_pages.append(sanitized_page)
    if binary_pages:
        meta["sanitized_page_texts"] = str(binary_pages)
    result.page_texts = sanitized_pages
    result.meta = meta
    return result


async def extract_pdf_text_async(
    pdf_bytes: bytes,
    *,
    session_id: str,
    filename: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    timeout: Optional[float] = None,
    redis_client: Optional[Any] = None,  # 로컬 폴백과 시그니처 호환을 위한 자리표시자
    redis_ttl: Optional[int] = None,
) -> PdfOcrResult:
    """
    DeepSeek OCR 전용 서버에 비동기 요청을 보내거나, 원격 서버가 없으면 로컬 파이프라인으로 폴백합니다.
    """
    if is_remote_available():
        request_timeout = timeout or _DEFAULT_TIMEOUT
        endpoint = _build_endpoint("/ocr/pdf")
        data = {"session_id": session_id}
        if filename:
            data["filename"] = filename

        async with httpx.AsyncClient(timeout=request_timeout) as client:
            if logger:
                logger.info(f"[OCR Client] 원격 OCR 서버 요청: {endpoint}, filename={filename}")
            
            response = await client.post(
                endpoint,
                data=data,
                files={"file": (filename or "document.pdf", pdf_bytes, "application/pdf")},
            )
            _raise_for_status(response)
            payload = response.json()
            
            # 디버그 모드: 전체 응답 페이로드 덤프
            if os.environ.get("DEEPSEEK_OCR_DEBUG", "").lower() in ("1", "true", "yes"):
                import json
                try:
                    dump_path = f"ocr_response_debug_{filename or 'unknown'}.json"
                    with open(dump_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2, ensure_ascii=False)
                    if logger:
                        logger.info(f"[OCR Client] 디버그: 응답 페이로드를 {dump_path}에 저장했습니다.")
                except Exception:
                    pass
            
            # 응답 페이로드 상세 로깅
            if logger:
                logger.info(
                    f"[OCR Client] 원격 서버 응답 수신: "
                    f"full_text 키={('full_text' in payload)}, "
                    f"page_texts 키={('page_texts' in payload)}, "
                    f"page_count={payload.get('page_count', 'N/A')}, "
                    f"file_hash={payload.get('file_hash', 'N/A')[:16] if payload.get('file_hash') else 'N/A'}..."
                )
                
                # full_text 상태 확인
                ft = payload.get('full_text', '')
                if ft:
                    logger.info(f"[OCR Client] full_text 존재: {len(ft)}자")
                else:
                    logger.warning(f"[OCR Client] full_text가 비어있거나 None")
                
                # page_texts 상태 확인
                pt = payload.get('page_texts', [])
                if pt:
                    valid_pt = [p for p in pt if p and isinstance(p, str) and p.strip()]
                    logger.info(
                        f"[OCR Client] page_texts 존재: 전체={len(pt)}, "
                        f"유효={len(valid_pt)}"
                    )
                    if valid_pt:
                        first_page_len = len(valid_pt[0])
                        logger.info(f"[OCR Client] 첫 페이지 텍스트 길이: {first_page_len}자")
                else:
                    logger.warning(f"[OCR Client] page_texts가 비어있거나 None")
            
            result = deserialize_pdf_ocr_result(payload, default_hash=payload.get("file_hash"))
            
            if logger:
                logger.info(
                    f"[OCR Client] deserialize 완료: "
                    f"full_text={len(result.full_text)}자, "
                    f"page_texts={len(result.page_texts)}개"
                )

            result = _postprocess_ocr_result(
                result,
                logger,
                filename or "pdf_document",
            )

            if logger:
                logger.info(
                    f"[OCR Client] 후처리 완료: full_text={len(result.full_text)}자, "
                    f"page_texts={len(result.page_texts)}개"
                )

            return result

    # 원격 서버가 없으면 로컬 파이프라인을 사용합니다.
    if logger is None:
        logger = _LOGGER
    try:
        from deepseek_pdf_pipeline import extract_text_from_pdf_bytes_cached
    except ImportError as exc:
        raise RuntimeError(
            "DeepSeek OCR 전용 서버 URL이 설정되지 않았고, 로컬 파이프라인도 사용할 수 없습니다."
        ) from exc

    return await extract_text_from_pdf_bytes_cached(
        pdf_bytes,
        session_id=session_id,
        redis_client=redis_client,
        redis_ttl=redis_ttl,
        filename=filename,
        logger=logger,
    )


def extract_pdf_text_sync(
    pdf_bytes: bytes,
    *,
    session_id: str,
    filename: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    timeout: Optional[float] = None,
) -> PdfOcrResult:
    """
    DeepSeek OCR 전용 서버를 동기 방식으로 호출합니다.
    원격 서버가 없으면 로컬 파이프라인으로 폴백합니다.
    """
    if is_remote_available():
        request_timeout = timeout or _DEFAULT_TIMEOUT
        endpoint = _build_endpoint("/ocr/pdf")
        data = {"session_id": session_id}
        if filename:
            data["filename"] = filename

        with httpx.Client(timeout=request_timeout) as client:
            if logger:
                logger.info(f"[OCR Client Sync] 원격 OCR 서버 요청: {endpoint}, filename={filename}")
            
            response = client.post(
                endpoint,
                data=data,
                files={"file": (filename or "document.pdf", pdf_bytes, "application/pdf")},
            )
            _raise_for_status(response)
            payload = response.json()
            
            # 디버그 모드: 전체 응답 페이로드 덤프
            if os.environ.get("DEEPSEEK_OCR_DEBUG", "").lower() in ("1", "true", "yes"):
                import json
                try:
                    dump_path = f"ocr_response_debug_sync_{filename or 'unknown'}.json"
                    with open(dump_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2, ensure_ascii=False)
                    if logger:
                        logger.info(f"[OCR Client Sync] 디버그: 응답 페이로드를 {dump_path}에 저장했습니다.")
                except Exception:
                    pass
            
            # 응답 페이로드 상세 로깅
            if logger:
                logger.info(
                    f"[OCR Client Sync] 원격 서버 응답 수신: "
                    f"full_text 키={('full_text' in payload)}, "
                    f"page_texts 키={('page_texts' in payload)}, "
                    f"page_count={payload.get('page_count', 'N/A')}, "
                    f"file_hash={payload.get('file_hash', 'N/A')[:16] if payload.get('file_hash') else 'N/A'}..."
                )
                
                ft = payload.get('full_text', '')
                if ft:
                    logger.info(f"[OCR Client Sync] full_text 존재: {len(ft)}자")
                else:
                    logger.warning(f"[OCR Client Sync] full_text가 비어있거나 None")
                
                pt = payload.get('page_texts', [])
                if pt:
                    valid_pt = [p for p in pt if p and isinstance(p, str) and p.strip()]
                    logger.info(
                        f"[OCR Client Sync] page_texts 존재: 전체={len(pt)}, "
                        f"유효={len(valid_pt)}"
                    )
                    if valid_pt:
                        first_page_len = len(valid_pt[0])
                        logger.info(f"[OCR Client Sync] 첫 페이지 텍스트 길이: {first_page_len}자")
                else:
                    logger.warning(f"[OCR Client Sync] page_texts가 비어있거나 None")
            
            result = deserialize_pdf_ocr_result(payload, default_hash=payload.get("file_hash"))
            
            if logger:
                logger.info(
                    f"[OCR Client Sync] deserialize 완료: "
                    f"full_text={len(result.full_text)}자, "
                    f"page_texts={len(result.page_texts)}개"
                )

            result = _postprocess_ocr_result(
                result,
                logger,
                filename or "pdf_document",
            )

            if logger:
                logger.info(
                    f"[OCR Client Sync] 후처리 완료: full_text={len(result.full_text)}자, "
                    f"page_texts={len(result.page_texts)}개"
                )

            return result

    if logger is None:
        logger = _LOGGER
    try:
        from deepseek_pdf_pipeline import extract_text_from_pdf_bytes
    except ImportError as exc:
        raise RuntimeError(
            "DeepSeek OCR 전용 서버 URL이 설정되지 않았고, 로컬 파이프라인도 사용할 수 없습니다."
        ) from exc

    result = extract_text_from_pdf_bytes(pdf_bytes, filename)
    result = _postprocess_ocr_result(
        result,
        logger,
        filename or "pdf_document",
    )
    result.meta = {
        **(result.meta or {}),
        "session_id": session_id,
    }
    return result


def extract_pdf_text_with_cache_sync(
    file_path: str,
    *,
    session_id: str,
    logger: Optional[logging.Logger] = None,
    timeout: Optional[float] = None,
    redis_client: Optional[Any] = None,
    redis_ttl: Optional[int] = None,
) -> PdfOcrResult:
    """
    파일 경로를 입력받아 DeepSeek OCR 처리를 수행합니다.
    """
    with open(file_path, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()

    return extract_pdf_text_sync(
        pdf_bytes,
        session_id=session_id,
        filename=os.path.basename(file_path),
        logger=logger,
        timeout=timeout,
    )


async def extract_pdf_text_with_cache_async(
    pdf_bytes: bytes,
    *,
    session_id: str,
    filename: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    timeout: Optional[float] = None,
    redis_client: Optional[Any] = None,
    redis_ttl: Optional[int] = None,
) -> PdfOcrResult:
    """
    비동기 환경에서 DeepSeek OCR 처리를 수행합니다.
    """
    return await extract_pdf_text_async(
        pdf_bytes,
        session_id=session_id,
        filename=filename,
        logger=logger,
        timeout=timeout,
        redis_client=redis_client,
        redis_ttl=redis_ttl,
    )




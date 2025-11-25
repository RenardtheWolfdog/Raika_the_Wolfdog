"""
DeepSeek-OCR 기반 PDF 전용 처리 파이프라인

이 모듈은 DeepSeek-OCR 모델을 활용하여 PDF(텍스트 + 이미지 포함)를
텍스트로 변환하는 기능을 제공합니다. 주요 특징은 다음과 같습니다.

1. 모델과 프로세서를 전역 싱글톤으로 관리하여 반복 로딩 비용을 최소화
2. PDF를 페이지 단위로 이미지 변환한 뒤 DeepSeek-OCR에 입력
3. 각 페이지별 추출 텍스트와 전체 텍스트를 모두 반환
4. Redis 등 외부 캐시 모듈에서 활용할 수 있도록 해시/메타데이터 생성 지원
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import asyncio
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from deepseek_ocr_types import (
    PdfOcrResult,
    deserialize_pdf_ocr_result,
    serialize_pdf_ocr_result,
)

import torch

try:
    from transformers import AutoModel, AutoTokenizer
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "transformers 라이브러리가 필요합니다. pip install transformers 로 설치해 주세요."
    ) from exc

try:
    from pdf2image import convert_from_bytes
except Exception:
    convert_from_bytes = None  # pdf2image 미설치 환경 대비

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Pillow 라이브러리가 필요합니다. pip install pillow 로 설치해 주세요."
    ) from exc

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_ID = os.environ.get("DEEPSEEK_OCR_MODEL_ID", "deepseek-ai/DeepSeek-OCR")
DEFAULT_DEVICE = os.environ.get("DEEPSEEK_OCR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
# 251108 - .pdf, OCR 문서 전용 처리 로직
DEFAULT_PROMPT = os.environ.get(
    "DEEPSEEK_OCR_PROMPT",
    "<image>\n<|grounding|>Convert the document to markdown.",
)

_ocr_model = None
_ocr_tokenizer = None
_effective_device_name: Optional[str] = None

# Redis/캐시 관련 기본 설정
PDF_OCR_CACHE_NAMESPACE = os.environ.get("DEEPSEEK_OCR_CACHE_NAMESPACE", "session")
PDF_OCR_CACHE_BUCKET = os.environ.get("DEEPSEEK_OCR_CACHE_BUCKET", "pdf_ocr")


def build_pdf_cache_key(session_id: str, file_hash: str) -> str:
    """
    Redis 등 외부 캐시에 사용할 PDF OCR 결과 키를 생성합니다.
    """
    sid = session_id or "default"
    return f"{PDF_OCR_CACHE_NAMESPACE}:{sid}:{PDF_OCR_CACHE_BUCKET}:{file_hash}"


async def _maybe_await(result: Any) -> Any:
    """
    동기/비동기 Redis 클라이언트 호환을 위한 헬퍼.
    """
    if inspect.isawaitable(result):
        return await result
    return result


def _decode_cache_payload(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _resolve_device_name() -> str:
    """
    환경 변수 및 실행 환경을 기반으로 사용할 장치를 결정합니다.
    CUDA가 요청되었지만 사용 불가하면 자동으로 CPU로 폴백합니다.
    """
    device_name = (DEFAULT_DEVICE or "").strip().lower() or "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning(
            "CUDA 장치가 요청되었지만 사용 불가합니다. CPU로 폴백합니다. "
            "PyTorch CUDA 빌드가 필요하면 CUDA 지원 버전의 torch를 설치하세요."
        )
        device_name = "cpu"
    if device_name.startswith("mps") and not torch.backends.mps.is_available():  # pragma: no cover (macOS 전용)
        LOGGER.warning("MPS 장치가 요청되었지만 사용 불가합니다. CPU로 폴백합니다.")
        device_name = "cpu"
    return device_name


def _current_device_name() -> str:
    """
    현재 로드된 모델이 사용하는 장치 이름을 반환합니다.
    """
    return _effective_device_name or _resolve_device_name()


def _ensure_model():
    """
    DeepSeek-OCR 모델과 프로세서를 전역 싱글톤으로 로드합니다.
    """
    global _ocr_model, _ocr_tokenizer, _effective_device_name

    if _ocr_model is not None and _ocr_tokenizer is not None:
        return

    LOGGER.info("DeepSeek-OCR 모델을 로드합니다: %s", DEFAULT_MODEL_ID)
    # 251108 - .pdf, OCR 문서 전용 처리 로직
    _ocr_tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_MODEL_ID,
        trust_remote_code=True,
    )
    _ocr_model = AutoModel.from_pretrained(
        DEFAULT_MODEL_ID,
        trust_remote_code=True,
        use_safetensors=True,
    )
    device_name = _resolve_device_name()
    _effective_device_name = device_name
    device = torch.device(device_name)
    if device.type == "cpu":
        _ocr_model.to(device=device)
    else:
        dtype = torch.bfloat16 if device.type == "cuda" else None
        if dtype is not None:
            _ocr_model.to(device=device, dtype=dtype)
        else:
            _ocr_model.to(device=device)
    _ocr_model.eval()


def _hash_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    PDF 원본 바이트로부터 SHA256 해시를 생성합니다.
    """
    return hashlib.sha256(pdf_bytes).hexdigest()


def _clean_ocr_artifacts(text: str) -> str:
    """
    OCR 결과를 완전히 텍스트화합니다.
    - 모든 특수 토큰 제거
    - 제어 문자/바이너리 바이트 제거
    - 사람이 읽을 수 있는 순수 텍스트만 반환
    """
    if not text:
        return text
    
    import re
    
    # 1. 모든 DeepSeek 특수 토큰 제거
    cleaned = text
    cleaned = re.sub(r'<\|[^>]+\|>', '', cleaned)  # <|토큰|>
    cleaned = re.sub(r'<\|/[^>]+\|>', '', cleaned)  # <|/토큰|>
    cleaned = re.sub(r'\[\[[\d\s,]+\]\]', '', cleaned)  # [[좌표]]
    
    # 2. 제어 문자 및 바이너리 바이트 완전 제거
    cleaned = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', cleaned)
    
    # 3. 유니코드 치환 문자 제거 (잘못된 인코딩)
    cleaned = cleaned.replace('\ufffd', '').replace('�', '')
    
    # 4. 과도한 특수 기호 연속 제거
    cleaned = re.sub(r'([^\w\s가-힣])\1{2,}', r'\1', cleaned)
    
    # 5. 빈 괄호 제거
    cleaned = re.sub(r'\(\s*\)|\[\s*\]|\{\s*\}', '', cleaned)
    
    # 6. 중복 공백/줄바꿈 정리
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    cleaned = re.sub(r'\t+', ' ', cleaned)
    
    # 7. 빈 줄 제거
    lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
    cleaned = '\n'.join(lines)
    
    return cleaned.strip()

def _normalize_infer_output(result: object) -> str:
    """
    DeepSeek-OCR infer 결과를 문자열로 정규화합니다.
    <|ref|>text<|/ref|> 형식을 파싱하고 특수 토큰을 제거합니다.
    """
    LOGGER.debug(f"[Normalize] input type: {type(result)}")
    
    if isinstance(result, str):
        import re
        
        # <|ref|>text<|/ref|> 형태에서 텍스트 추출
        if '<|ref|>' in result:
            matches = re.findall(r'<\|ref\|>(.*?)<\|/ref\|>', result, re.DOTALL)
            if matches:
                unique_texts = []
                for m in matches:
                    text_clean = _clean_ocr_artifacts(m.strip())
                    if text_clean and text_clean not in unique_texts:
                        unique_texts.append(text_clean)
                if unique_texts:
                    extracted = "\n\n".join(unique_texts)
                    LOGGER.info(f"[Normalize] ref 태그에서 {len(unique_texts)}개 블록 추출, 총 {len(extracted)}자")
                    return extracted
            
            # ref 태그가 있지만 추출 실패 시 태그만 제거
            result = re.sub(r'<\|ref\|>|<\|/ref\|>', '', result)
        
        # 특수 토큰 제거
        return _clean_ocr_artifacts(result)
    
    if isinstance(result, list):
        # 리스트의 문자열 요소만 결합
        texts = [_clean_ocr_artifacts(str(item)) for item in result if item]
        if texts:
            combined = "\n\n".join(t for t in texts if len(t) > 10)
            if combined:
                LOGGER.debug(f"[Normalize] 리스트에서 {len(texts)}개 추출")
                return combined
        return ""
    
    if isinstance(result, dict):
        # 일반적인 키 이름 확인
        for key in ["text", "response", "answer", "content", "output", "result"]:
            text = result.get(key)
            if text and isinstance(text, str):
                return _clean_ocr_artifacts(text)
        return ""
    
    # 기타: 문자열 변환
    return _clean_ocr_artifacts(str(result or ""))


def _collect_candidate_paths(obj: object, base_dir: str, candidates: List[str]) -> None:
    """
    infer 결과 객체에서 파일 경로 후보를 추출합니다.
    """
    if isinstance(obj, str):
        potential = obj if os.path.isabs(obj) else os.path.join(base_dir, obj)
        if os.path.exists(potential):
            candidates.append(potential)
    elif isinstance(obj, dict):
        for value in obj.values():
            _collect_candidate_paths(value, base_dir, candidates)
    elif isinstance(obj, (list, tuple, set)):
        for value in obj:
            _collect_candidate_paths(value, base_dir, candidates)


def _load_text_from_candidates(candidates: List[str]) -> str:
    """
    후보 경로 목록에서 유효한 텍스트를 찾아 반환합니다.
    가장 큰 파일을 우선 시도합니다.
    """
    def _is_probable_text_file(path: str) -> bool:
        # 이미지/압축 등 명백한 바이너리 확장자는 즉시 배제하고,
        # 텍스트 확장자는 우선적으로 허용한다.
        text_exts = {
            ".txt",
            ".md",
            ".mmd",
            ".markdown",
            ".json",
            ".csv",
            ".tsv",
            ".yaml",
            ".yml",
            ".html",
            ".htm",
            ".xml",
            ".log",
        }
        binary_exts = {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".webp",
            ".tif",
            ".tiff",
            ".pdf",
            ".zip",
            ".rar",
            ".gz",
            ".bz2",
            ".xz",
            ".tar",
            ".7z",
        }

        suffix = Path(path).suffix.lower()
        if suffix in text_exts:
            return True
        if suffix in binary_exts:
            return False

        # 확장자가 애매하면 파일 앞부분을 읽어 제어문자/바이너리 패턴을 검사한다.
        try:
            with open(path, "rb") as sample_file:
                sample = sample_file.read(4096)
        except Exception:
            return False

        if not sample:
            return False
        if b"\x00" in sample:
            return False

        # 제어문자 비율이 높으면 바이너리로 간주한다.
        non_text_bytes = sum(
            1
            for b in sample
            if (b < 9 or 13 < b < 32 or b == 127) and b not in (10, 13, 27)
        )
        if non_text_bytes / len(sample) > 0.15:
            return False

        # base64 형태의 긴 덩어리도 사실상 바이너리이므로 제거한다.
        base64_chars = sum(
            1 for b in sample if chr(b) in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=_-\r\n\t "
        )
        if len(sample) >= 128 and base64_chars / len(sample) > 0.9:
            return False

        return True

    if not candidates:
        return ""
    
    # 파일 크기 순으로 정렬 (큰 파일이 실제 내용일 가능성 높음)
    valid_files = []
    for path in candidates:
        if os.path.isdir(path):
            continue
        try:
            fsize = os.path.getsize(path)
            if fsize > 0:
                valid_files.append((path, fsize))
        except Exception:
            continue
    
    valid_files.sort(key=lambda x: x[1], reverse=True)  # 큰 파일 우선
    
    LOGGER.info(f"[LoadText] 유효 파일 {len(valid_files)}개, 크기 순 정렬 완료")
    
    # 각 파일 시도
    for path, fsize in valid_files:
        fname = os.path.basename(path)

        if not _is_probable_text_file(path):
            LOGGER.debug(f"[LoadText] 텍스트 파일로 판단되지 않아 건너뜀: {fname}")
            continue

        LOGGER.debug(f"[LoadText] 파일 시도: {fname} ({fsize} bytes)")
        
        # 다양한 인코딩으로 시도
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(path, "r", encoding=encoding) as file:
                    content = file.read()
                    if content and len(content.strip()) > 10:
                        # 특수 토큰만 제거
                        cleaned = _clean_ocr_artifacts(content)
                        if cleaned and len(cleaned.strip()) > 10:
                            LOGGER.info(
                                f"[LoadText] 성공: {fname} ({encoding}), "
                                f"원본={len(content)}자, 정제 후={len(cleaned)}자"
                            )
                            return cleaned
            except Exception:
                continue
    
    LOGGER.error(f"[LoadText] 모든 후보 파일에서 텍스트 추출 실패 (시도한 파일: {len(valid_files)}개)")
    return ""


def _run_infer_on_image_file(image_path: str, prompt: str, output_dir: str, page_index: int = 0) -> str:
    """
    단일 이미지 파일에 대해 DeepSeek-OCR 추론을 수행하고 텍스트를 반환합니다.
    
    Args:
        image_path: 이미지 파일 경로
        prompt: OCR 프롬프트
        output_dir: 결과 저장 디렉토리
        page_index: 페이지 번호 (1부터 시작, 로깅 및 파일 매칭용)
    """
    _ensure_model()

    page_num = os.path.basename(image_path)  # 로깅용
    
    try:
        LOGGER.debug(f"[Infer] 이미지 추론 시작: {page_num}")
        
        with torch.inference_mode():
            infer_result = _ocr_model.infer(
                _ocr_tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path=output_dir,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=True,
                test_compress=False,
            )
        
        # infer_result 타입 및 내용 로깅
        LOGGER.debug(
            f"[Infer] infer_result 타입: {type(infer_result)}, "
            f"내용(첫 200자): {str(infer_result)[:200] if infer_result else '(None)'}"
        )
        
        # 디버그 모드: infer_result 전체 덤프
        if os.environ.get("DEEPSEEK_OCR_DEBUG", "").lower() in ("1", "true", "yes"):
            try:
                import json
                dump_path = os.path.join(output_dir, f"infer_result_debug_page{page_index}.json")
                with open(dump_path, "w", encoding="utf-8") as f:
                    if isinstance(infer_result, (dict, list)):
                        json.dump(infer_result, f, indent=2, ensure_ascii=False, default=str)
                    else:
                        f.write(f"Type: {type(infer_result)}\n\n")
                        f.write(str(infer_result))
                LOGGER.info(f"[Infer] 디버그: infer_result를 {dump_path}에 저장")
            except Exception:
                pass
        
        # 1차: infer_result에서 직접 추출 시도
        direct_text = _normalize_infer_output(infer_result).strip()
        if direct_text and len(direct_text) >= 10:
            LOGGER.info(f"[Infer] 직접 추출 성공: {page_num}, 길이={len(direct_text)}자")
            return direct_text
        
        LOGGER.warning(f"[Infer] 직접 추출 실패 또는 너무 짧음 ({len(direct_text)}자), 파일 경로 탐색 시작: {page_num}")

        candidates: List[str] = []
        _collect_candidate_paths(infer_result, output_dir, candidates)
        LOGGER.debug(f"[Infer] 후보 경로 {len(candidates)}개 발견: {candidates[:3]}")
        
        fallback_text = _load_text_from_candidates(candidates)
        if fallback_text:
            LOGGER.info(f"[Infer] 후보 경로에서 추출 성공: {page_num}, 길이={len(fallback_text)}자")
            return fallback_text

        # 2차: output_dir 내 파일에서 텍스트 탐색 (적극적으로)
        LOGGER.info(f"[Infer] output_dir에서 저장된 파일 탐색 시작: {output_dir}")
        
        all_files_in_output = []
        try:
            all_files_in_output = os.listdir(output_dir)
            LOGGER.info(f"[Infer] output_dir 전체 파일 {len(all_files_in_output)}개: {all_files_in_output[:10]}")
        except Exception as list_err:
            LOGGER.error(f"[Infer] output_dir 조회 실패: {list_err}")
            return ""
        
        # 텍스트 파일 필터링 (확장자 제한 완화)
        text_files = [
            os.path.join(output_dir, name)
            for name in all_files_in_output
            if name.lower().endswith((".txt", ".md", ".markdown", ".text", ".out"))
        ]
        
        if not text_files:
            # 확장자가 없거나 이상한 파일도 시도 (DeepSeek가 이상한 이름으로 저장할 수 있음)
            LOGGER.warning(f"[Infer] 텍스트 파일 없음, 모든 파일을 후보로 시도")
            text_files = [
                os.path.join(output_dir, name)
                for name in all_files_in_output
                if os.path.isfile(os.path.join(output_dir, name))
            ]
        
        # 최신 파일 우선 정렬
        text_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        LOGGER.info(f"[Infer] 후보 파일 {len(text_files)}개 발견")
        
        # 각 후보 파일의 상세 정보 로깅
        for idx, tf in enumerate(text_files[:10], 1):
            try:
                fsize = os.path.getsize(tf)
                fname = os.path.basename(tf)
                mtime = os.path.getmtime(tf)
                LOGGER.info(f"[Infer] 후보 {idx}: {fname}, 크기={fsize}bytes, mtime={mtime}")
            except Exception:
                pass
        
        # 모든 후보 파일 시도
        final_text = _load_text_from_candidates(text_files)
        if final_text:
            LOGGER.info(f"[Infer] 파일에서 텍스트 추출 성공: {page_num}, 길이={len(final_text)}자")
            return final_text
        
        # 최후: output_dir 자체에 stdout capture된 텍스트가 있는지 확인
        LOGGER.error(
            f"[Infer] 모든 파일 탐색 실패: {page_num}. "
            f"output_dir={output_dir}, 전체 파일={len(all_files_in_output)}개"
        )
        
        # 디버그: 각 파일의 처음 100바이트라도 출력
        if os.environ.get("DEEPSEEK_OCR_DEBUG", "").lower() in ("1", "true", "yes"):
            for f in text_files[:5]:
                try:
                    with open(f, "rb") as raw_file:
                        raw_bytes = raw_file.read(100)
                        LOGGER.debug(f"[Infer Debug] {os.path.basename(f)} 처음 100bytes: {raw_bytes}")
                except Exception:
                    pass
        
        return ""
    except Exception as infer_error:
        LOGGER.error(
            f"[Infer] DeepSeek-OCR infer 예외 발생 ({page_num}): {infer_error}",
            exc_info=True
        )
        return ""


def extract_text_from_pdf_bytes(pdf_bytes: bytes, filename: str | None = None, timeout: int = 600) -> PdfOcrResult:
    """
    PDF 바이트 데이터를 입력받아 DeepSeek-OCR로 텍스트를 추출합니다.
    
    Args:
        pdf_bytes: PDF 파일 바이트
        filename: 파일명 (선택)
        timeout: OCR 처리 최대 시간(초), 기본값 600초
    """
    import time
    
    if not pdf_bytes:
        raise ValueError("PDF 바이트 데이터가 비어 있습니다.")

    if convert_from_bytes is None:
        raise RuntimeError(
            "pdf2image 라이브러리가 설치되어 있지 않습니다. "
            "pip install pdf2image Poppler 설치 후 다시 시도해 주세요."
        )

    start_time = time.time()
    page_texts: List[str] = []
    
    with tempfile.TemporaryDirectory(prefix="deepseek_ocr_") as tmp_dir:
        images = convert_from_bytes(pdf_bytes, fmt="png")
        if not images:
            raise RuntimeError("PDF -> 이미지 변환에 실패했습니다.")

        total_pages = len(images)
        LOGGER.info("PDF 변환 완료: %d페이지, 파일명: %s", total_pages, filename or "unknown")

        for index, img in enumerate(images, start=1):
            # 타임아웃 체크
            elapsed = time.time() - start_time
            if elapsed > timeout:
                LOGGER.warning("OCR 타임아웃(%d초 초과): %d/%d페이지 처리됨", timeout, index-1, total_pages)
                break
            
            image_path = os.path.join(tmp_dir, f"page_{index:04d}.png")
            img.save(image_path, format="PNG")
            img.close()

            # 진행 상황 로깅 (5페이지마다 또는 첫/마지막 페이지)
            if index == 1 or index == total_pages or index % 5 == 0:
                LOGGER.info("OCR 진행: %d/%d (%.1f%%)", index, total_pages, (index/total_pages*100))
            
            page_text = _run_infer_on_image_file(image_path, DEFAULT_PROMPT, tmp_dir, page_index=index)
            if not page_text:
                LOGGER.warning("DeepSeek-OCR 결과가 비어 있습니다 (페이지 %d)", index)
            else:
                LOGGER.info("DeepSeek-OCR 페이지 %d 추출 성공: %d자", index, len(page_text))
            page_texts.append(page_text)

    if not page_texts:
        raise RuntimeError("OCR 처리 결과가 하나도 없습니다.")
    
    full_text = "\n\n".join(page_texts)
    file_hash = _hash_pdf_bytes(pdf_bytes)

    processing_time = time.time() - start_time
    meta = {
        "filename": filename,
        "model_id": DEFAULT_MODEL_ID,
        "device": _current_device_name(),
        "processing_time_seconds": round(processing_time, 2),
        "pages_processed": len(page_texts),
        "total_pages": total_pages,
    }

    LOGGER.info("OCR 완료: %d/%d페이지, %.1f초 소요", len(page_texts), total_pages, processing_time)

    return PdfOcrResult(
        full_text=full_text,
        page_texts=page_texts,
        page_count=len(page_texts),
        file_hash=file_hash,
        meta=meta,
    )


async def extract_text_from_pdf_bytes_async(pdf_bytes: bytes, filename: str | None = None) -> PdfOcrResult:
    """
    비동기 환경에서 사용하기 위한 헬퍼. ThreadPool 실행을 통해 CPU/GPU 블로킹을 피합니다.
    """
    import asyncio

    return await asyncio.to_thread(extract_text_from_pdf_bytes, pdf_bytes, filename)


def extract_text_from_pdf_file(pdf_path: str) -> PdfOcrResult:
    """
    파일 경로를 입력받아 텍스트를 추출합니다.
    """
    with open(pdf_path, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()
    return extract_text_from_pdf_bytes(pdf_bytes, os.path.basename(pdf_path))


async def extract_text_from_pdf_bytes_cached(
    pdf_bytes: bytes,
    *,
    session_id: str,
    redis_client: Optional[Any],
    redis_ttl: Optional[int] = None,
    filename: str | None = None,
    logger: logging.Logger = LOGGER,
) -> PdfOcrResult:
    """
    Redis (또는 호환 캐시 클라이언트)를 사용하여 PDF OCR 결과를 캐시/재사용합니다.

    redis_client는 redis.asyncio.Redis와 같이 get/set 메서드를 제공해야 합니다.
    동기 클라이언트의 경우에도 동작하지만 이벤트 루프를 블로킹할 수 있으므로 권장하지 않습니다.
    """
    if not pdf_bytes:
        raise ValueError("PDF 바이트 데이터가 비어 있습니다.")

    pdf_hash = _hash_pdf_bytes(pdf_bytes)
    cache_key = build_pdf_cache_key(session_id, pdf_hash)

    if redis_client is not None:
        try:
            cached_raw = await _maybe_await(redis_client.get(cache_key))
            cached_payload = _decode_cache_payload(cached_raw)
            if cached_payload:
                cached_result = deserialize_pdf_ocr_result(cached_payload, pdf_hash)
                cached_result.meta = {
                    **(cached_result.meta or {}),
                    "session_id": session_id,
                    "filename": filename or cached_result.meta.get("filename"),
                    "device": (cached_result.meta or {}).get("device") or _current_device_name(),
                }
                if logger:
                    logger.info("DeepSeek-OCR 캐시 적중: %s", cache_key)
                return cached_result
        except Exception as cache_err:
            if logger:
                logger.warning("DeepSeek-OCR 캐시 조회 실패(%s): %s", cache_key, cache_err)

    if logger:
        logger.info("DeepSeek-OCR 캐시 미스 → 추론 실행: %s", cache_key)
    ocr_result = await extract_text_from_pdf_bytes_async(pdf_bytes, filename)
    ocr_result.meta = {
        **(ocr_result.meta or {}),
        "session_id": session_id,
        "filename": filename or (ocr_result.meta or {}).get("filename"),
        "device": (ocr_result.meta or {}).get("device") or _current_device_name(),
    }

    if redis_client is not None:
        try:
            payload = json.dumps(serialize_pdf_ocr_result(ocr_result), ensure_ascii=False)
            await _maybe_await(redis_client.set(cache_key, payload, ex=redis_ttl))
        except Exception as cache_err:
            if logger:
                logger.warning("DeepSeek-OCR 캐시 저장 실패(%s): %s", cache_key, cache_err)

    return ocr_result


async def extract_text_from_pdf_file_cached(
    pdf_path: str,
    *,
    session_id: str,
    redis_client: Optional[Any],
    redis_ttl: Optional[int] = None,
    logger: logging.Logger = LOGGER,
) -> PdfOcrResult:
    """
    파일 경로 기반 캐시 헬퍼.
    """
    with open(pdf_path, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()
    return await extract_text_from_pdf_bytes_cached(
        pdf_bytes,
        session_id=session_id,
        redis_client=redis_client,
        redis_ttl=redis_ttl,
        filename=os.path.basename(pdf_path),
        logger=logger,
    )


def calculate_pdf_hash(pdf_bytes: bytes) -> str:
    """
    외부 모듈에서 재사용할 수 있도록 PDF 해시 계산기를 제공합니다.
    """
    return _hash_pdf_bytes(pdf_bytes)


# 251108 - .pdf, OCR 문서 전용 처리 로직
def _get_default_pdf_path() -> str:
    """
    프로젝트 루트에 위치한 `Attention_is_all_you_need.pdf` 문서 경로를 반환합니다.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(project_root, "Attention_is_all_you_need.pdf")


# 251108 - .pdf, OCR 문서 전용 처리 로직
def _print_ocr_preview(result: PdfOcrResult, preview_chars: int = 800) -> None:
    """
    DeepSeek-OCR 결과를 간단히 확인할 수 있도록 요약 정보를 출력합니다.
    """
    print(f"[DeepSeek-OCR] 총 페이지 수: {result.page_count}")
    print(f"[DeepSeek-OCR] SHA256 해시: {result.file_hash}")
    if result.page_texts:
        print("[DeepSeek-OCR] 첫 페이지 추출 텍스트 미리보기:")
        first_page = result.page_texts[0][:preview_chars]
        print(first_page if first_page else "(비어 있음)")
    else:
        print("[DeepSeek-OCR] 추출된 텍스트가 없습니다.")


# 251108 - .pdf, OCR 문서 전용 처리 로직
def main():
    """
    deepseek_pdf_pipeline.py를 직접 실행했을 때,
    프로젝트 루트에 있는 Attention_is_all_you_need.pdf 논문을 OCR 처리합니다.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    pdf_path = _get_default_pdf_path()
    if not os.path.exists(pdf_path):
        print(f"[DeepSeek-OCR] 대상 PDF가 존재하지 않습니다: {pdf_path}")
        return

    print(f"[DeepSeek-OCR] PDF OCR을 시작합니다: {pdf_path}")

    try:
        result = extract_text_from_pdf_file(pdf_path)
        _print_ocr_preview(result)

        output_name = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_ocr.txt"
        output_path = os.path.join(os.path.dirname(pdf_path), output_name)
        with open(output_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(result.full_text)

        print(f"[DeepSeek-OCR] 전체 OCR 결과를 저장했습니다: {output_path}")
    except Exception as exc:
        logging.exception(f"[DeepSeek-OCR] OCR 처리 중 오류 발생: {exc}")


if __name__ == "__main__":
    main()



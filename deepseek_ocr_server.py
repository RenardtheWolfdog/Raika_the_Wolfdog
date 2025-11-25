from __future__ import annotations

import logging
import os
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

try:
    import redis.asyncio as redis
except Exception:  # pragma: no cover - redis는 선택 사항
    redis = None

from deepseek_ocr_types import serialize_pdf_ocr_result
from deepseek_pdf_pipeline import extract_text_from_pdf_bytes_cached

LOGGER = logging.getLogger("deepseek_ocr_server")

app = FastAPI(title="DeepSeek OCR Service", version="1.0.0")

_redis_client: Optional["redis.Redis"] = None
_redis_ttl: Optional[int] = None


async def _init_redis_client() -> Optional["redis.Redis"]:
    url = os.environ.get("DEEPSEEK_OCR_REDIS_URL")
    if not url or redis is None:
        return None

    client = redis.from_url(
        url,
        encoding="utf-8",
        decode_responses=False,
    )
    try:
        await client.ping()
    except Exception as exc:  # pragma: no cover - 시작 시 연결 실패 처리
        LOGGER.warning("Redis 연결 실패: %s", exc)
        await client.close()
        return None
    return client


@app.on_event("startup")
async def on_startup():
    global _redis_client, _redis_ttl

    logging.basicConfig(
        level=os.environ.get("DEEPSEEK_OCR_LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    LOGGER.info("DeepSeek OCR 서비스가 시작됩니다.")
    _redis_ttl_env = os.environ.get("DEEPSEEK_OCR_REDIS_TTL")
    _redis_ttl = int(_redis_ttl_env) if _redis_ttl_env else None
    _redis_client = await _init_redis_client()


@app.on_event("shutdown")
async def on_shutdown():
    if _redis_client is not None:
        await _redis_client.close()
        LOGGER.info("Redis 연결을 종료했습니다.")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/ocr/pdf")
async def run_pdf_ocr(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    filename: Optional[str] = Form(None),
):
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="PDF 데이터가 비어 있습니다.")

    resolved_filename = filename or getattr(file, "filename", None) or "document.pdf"

    try:
        result = await extract_text_from_pdf_bytes_cached(
            pdf_bytes,
            session_id=session_id,
            redis_client=_redis_client,
            redis_ttl=_redis_ttl,
            filename=resolved_filename,
            logger=LOGGER,
        )
    except Exception as exc:
        LOGGER.exception("DeepSeek OCR 처리 실패: %s", exc)
        raise HTTPException(status_code=500, detail=f"OCR 처리 실패: {exc}") from exc

    payload = serialize_pdf_ocr_result(result)
    payload["meta"] = {
        **(payload.get("meta") or {}),
        "session_id": session_id,
        "filename": resolved_filename,
    }
    return JSONResponse(content=payload)


def main():
    host = os.environ.get("DEEPSEEK_OCR_HOST", "0.0.0.0")
    port = int(os.environ.get("DEEPSEEK_OCR_PORT", "5600"))
    workers = int(os.environ.get("DEEPSEEK_OCR_WORKERS", "1"))

    if os.name == "nt":
        # Windows에서는 uvicorn workers>1 이 안정적이지 않아 단일 프로세스를 사용합니다.
        workers = 1

    uvicorn.run(
        "deepseek_ocr_server:app",
        host=host,
        port=port,
        workers=workers,
        reload=os.environ.get("DEEPSEEK_OCR_RELOAD", "0") not in {"0", "false", "no"},
    )


if __name__ == "__main__":
    main()


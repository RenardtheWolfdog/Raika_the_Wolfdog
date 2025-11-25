# document_summarizer_Gemma_Lang.py - LLM으로 문서 요약 정리하는 스크립트 (LangChain, LangGraph 적용)

import os
import re
import torch
from torch.cuda.amp import autocast
import PyPDF2
import csv
import openpyxl
from bs4 import BeautifulSoup, Comment
import json

import tempfile

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import multiprocessing
from functools import partial

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

import asyncio
from typing import TypedDict, List, Optional, Tuple, Dict, Any, Union
import GoogleSearch_Gemma

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from raika_large_script_helpers import build_large_script_prompt
from deepseek_ocr_client import (
    extract_pdf_text_sync,
    extract_pdf_text_with_cache_async,
    is_remote_available,
)
from deepseek_ocr_types import PdfOcrResult

import logging
# 한국어 NLP 라이브러리 추가

# 로깅 설정
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.DEBUG, # 또는 INFO
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()]) # 명시적으로 핸들러 추가

# 한국어 NLP 라이브러리 추가
from konlpy.tag import Mecab # 한국어 형태소 분석기
try:
    import kss  # type: ignore[import-not-found]  # Korean Sentence Splitter - 한국어 문장 분리기
except ImportError:
    kss = None
    logging.getLogger(__name__).warning(
        "kss 패키지를 찾을 수 없어 한국어 문장 분리를 기본 방식으로 대체합니다. "
        "pip install kss 명령으로 설치할 수 있습니다."
    )

# 전역 변수로 model과 processor, embedding_model 선언
# 전역 변수로 model과 processor, embedding_model 선언
model = None
processor = None
embedding_model = None

# DeepSeek-OCR 캐시 설정
ENABLE_PDF_CACHE = os.environ.get("ENABLE_PDF_OCR_CACHE", "1").strip().lower() not in {"0", "false", "no"}
PDF_CACHE_SESSION_ID = os.environ.get("PDF_OCR_SESSION_ID", "docsum_gemma_lang")
_pdf_cache_manager = None
_pdf_cache_manager_initialized = False


def _ensure_pdf_cache_manager():
    """
    RedisManager 인스턴스를 지연 로딩합니다.
    """
    global _pdf_cache_manager_initialized, _pdf_cache_manager

    if _pdf_cache_manager_initialized:
        return _pdf_cache_manager

    _pdf_cache_manager_initialized = True

    if not ENABLE_PDF_CACHE:
        logging.info("PDF OCR 캐시가 비활성화되었습니다 (환경 변수 설정).")
        return None

    try:
        # 현재 이벤트 루프가 실행 중이면 안전하게 초기화할 수 없으므로 캐시 비활성화
        asyncio.get_running_loop()
        logging.warning("실행 중인 이벤트 루프를 감지하여 PDF 캐시 초기화를 건너뜁니다.")
        return None
    except RuntimeError:
        # 이벤트 루프가 실행 중이 아니면 초기화 진행
        pass

    try:
        from redis_utils import RedisManager  # 지연 임포트
    except Exception as import_err:
        logging.warning("RedisManager 임포트 실패로 PDF 캐시를 비활성화합니다: %s", import_err)
        return None

    try:
        _pdf_cache_manager = asyncio.run(RedisManager.create_from_config())
        logging.info("PDF OCR 캐시용 RedisManager를 초기화했습니다.")
    except Exception as init_err:
        logging.warning("RedisManager 초기화 실패로 PDF 캐시를 비활성화합니다: %s", init_err)
        _pdf_cache_manager = None

    return _pdf_cache_manager


def _extract_pdf_with_optional_cache(file_path: str) -> PdfOcrResult:
    """
    DeepSeek-OCR을 활용하여 PDF를 처리하고, Redis 캐시가 가능하면 캐시를 사용합니다.
    """
    try:
        with open(file_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
    except Exception:
        raise

    filename = os.path.basename(file_path)

    if is_remote_available():
        result = extract_pdf_text_sync(
            pdf_bytes,
            session_id=PDF_CACHE_SESSION_ID,
            filename=filename,
        )
        result = _normalize_pdf_ocr_result(result)
        result.meta = {
            **(result.meta or {}),
            "session_id": PDF_CACHE_SESSION_ID,
        }

        # 251110 - PDF 분석 개선 작업
        manager_remote = _ensure_pdf_cache_manager()
        _store_pdf_rag_cache_with_manager(
            manager_remote,
            result.file_hash,
            result.full_text,
            session_id=PDF_CACHE_SESSION_ID,
        )
        return result

    manager = _ensure_pdf_cache_manager()

    if manager:
        async def _run_cached():
            return await extract_pdf_text_with_cache_async(
                pdf_bytes,
                session_id=PDF_CACHE_SESSION_ID,
                redis_client=manager.client,
                redis_ttl=manager.default_ttl,
                filename=filename,
                logger=logging.getLogger(__name__),
            )

        try:
            return asyncio.run(_run_cached())
        except RuntimeError:
            logging.warning("PDF 캐시 작업 중 이벤트 루프 충돌이 발생하여 캐시 없이 처리합니다.")
        except Exception as cache_err:
            logging.warning("PDF 캐시 처리 실패로 비캐시 경로를 사용합니다: %s", cache_err)

    result = extract_pdf_text_sync(
        pdf_bytes,
        session_id=PDF_CACHE_SESSION_ID,
        filename=filename,
    )
    result = _normalize_pdf_ocr_result(result)
    result.meta = {
        **(result.meta or {}),
        "session_id": PDF_CACHE_SESSION_ID,
    }

    # 251110 - PDF 분석 개선 작업
    _store_pdf_rag_cache_with_manager(
        manager,
        result.file_hash,
        result.full_text,
        session_id=PDF_CACHE_SESSION_ID,
    )

    return result


# 251110 - PDF 분석 개선 작업
def _build_pdf_rag_cache_payload(full_text: str) -> Tuple[List[str], np.ndarray]:
    """
    DeepSeek-OCR 결과 텍스트를 RAG 캐시용 청크와 임베딩 배열로 변환합니다.
    DeepSeek 특수 토큰을 제거한 후 처리합니다.
    """
    global processor, embedding_model

    if not full_text or len(full_text.strip()) < 20:
        return [], np.empty((0, 0), dtype=np.float32)
    
    # DeepSeek OCR 결과를 완전히 텍스트화
    import re
    cleaned_text = full_text
    
    # 1. 모든 DeepSeek 특수 토큰 제거
    cleaned_text = re.sub(r'<\|[^>]+\|>', '', cleaned_text)  # <|토큰|>
    cleaned_text = re.sub(r'<\|/[^>]+\|>', '', cleaned_text)  # <|/토큰|>
    cleaned_text = re.sub(r'\[\[[\d\s,]+\]\]', '', cleaned_text)  # [[좌표]]
    
    # 2. 제어 문자 및 바이너리 바이트 제거
    cleaned_text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', cleaned_text)
    
    # 3. 유니코드 치환 문자 제거
    cleaned_text = cleaned_text.replace('\ufffd', '').replace('�', '')
    
    # 4. 과도한 특수 기호 연속 제거
    cleaned_text = re.sub(r'([^\w\s가-힣])\1{2,}', r'\1', cleaned_text)
    
    # 5. 중복 공백/줄바꿈 정리
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)
    cleaned_text = re.sub(r'\t+', ' ', cleaned_text)
    
    # 6. 빈 줄 제거
    lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
    cleaned_text = '\n'.join(lines)
    
    if not cleaned_text or len(cleaned_text) < 20:
        logging.getLogger(__name__).error(
            "PDF RAG 캐시: 텍스트화 후 내용이 너무 짧거나 비어있습니다. "
            f"(원본={len(full_text)}자, 정제 후={len(cleaned_text)}자)"
        )
        return [], np.empty((0, 0), dtype=np.float32)
    
    logging.getLogger(__name__).info(
        f"PDF RAG 캐시: 완전 텍스트화 완료 "
        f"(원본={len(full_text)}자 → 정제 후={len(cleaned_text)}자, "
        f"제거율={(1 - len(cleaned_text)/len(full_text))*100:.1f}%)"
    )
    
    full_text = cleaned_text  # 정제된 순수 텍스트 사용

    if processor is None or model is None:
        try:
            set_model_and_processor()
        except Exception as set_err:
            logging.getLogger(__name__).warning(
                "PDF RAG 캐시 생성을 위한 모델 초기화 실패: %s", set_err
            )

    if embedding_model is None:
        try:
            load_embedding_model()
        except Exception as embed_err:
            logging.getLogger(__name__).warning(
                "PDF RAG 캐시 생성을 위한 임베딩 모델 로드 실패: %s", embed_err
            )

    if processor is None or embedding_model is None:
        logging.getLogger(__name__).warning(
            "PDF RAG 캐시 생성을 위한 필수 모델이 준비되지 않았습니다."
        )
        return [], np.empty((0, 0), dtype=np.float32)

    try:
        chunks_with_embeddings = optimize_chunk_size(full_text, processor, embedding_model)
    except Exception as optimize_err:
        logging.getLogger(__name__).warning(
            "PDF RAG 캐시용 청크 생성 실패: %s", optimize_err
        )
        return [], np.empty((0, 0), dtype=np.float32)

    if not chunks_with_embeddings:
        return [], np.empty((0, 0), dtype=np.float32)

    chunk_texts = [chunk for chunk, _ in chunks_with_embeddings if chunk.strip()]
    embedding_vectors = [emb for _, emb in chunks_with_embeddings if isinstance(emb, np.ndarray)]

    if not chunk_texts or not embedding_vectors:
        return [], np.empty((0, 0), dtype=np.float32)

    try:
        embeddings_array = np.vstack(embedding_vectors).astype(np.float32)
    except Exception as stack_err:
        logging.getLogger(__name__).warning(
            "PDF RAG 임베딩 배열 구성 실패: %s", stack_err
        )
        return [], np.empty((0, 0), dtype=np.float32)

    return chunk_texts, embeddings_array


# 251110 - PDF 분석 개선 작업
def _store_pdf_rag_cache_with_manager(
    manager: Any,
    file_hash: str,
    full_text: str,
    session_id: str = PDF_CACHE_SESSION_ID,
) -> None:
    """
    동기 컨텍스트에서 RedisManager를 사용하여 PDF RAG 캐시를 저장합니다.
    """
    logger = logging.getLogger(__name__)
    
    if not manager:
        logger.warning("PDF RAG 캐시 저장 실패: RedisManager가 None입니다.")
        return
    
    if not full_text or not file_hash:
        logger.warning(
            "PDF RAG 캐시 저장 실패: full_text 또는 file_hash가 비어 있습니다. "
            f"(hash={file_hash}, text_len={len(full_text) if full_text else 0})"
        )
        return

    try:
        chunks, embeddings = _build_pdf_rag_cache_payload(full_text)
        if not chunks or not isinstance(embeddings, np.ndarray) or embeddings.size == 0:
            logger.warning(
                f"PDF RAG 캐시 저장 실패: 청크 또는 임베딩이 비어 있습니다. "
                f"(hash={file_hash}, chunks={len(chunks) if chunks else 0}, "
                f"embeddings_size={embeddings.size if isinstance(embeddings, np.ndarray) else 'N/A'})"
            )
            return

        async def _store():
            save_ok = await manager.save_pdf_rag_cache(session_id, file_hash, chunks, embeddings)
            if save_ok:
                logger.info(
                    f"PDF RAG 캐시 저장 성공: session={session_id}, hash={file_hash}, "
                    f"chunks={len(chunks)}, embedding_shape={embeddings.shape}"
                )
            else:
                logger.warning(
                    f"PDF RAG 캐시 저장 실패: save_pdf_rag_cache가 False를 반환했습니다. "
                    f"(session={session_id}, hash={file_hash})"
                )
            return save_ok

        try:
            result = asyncio.run(_store())
            if not result:
                logger.warning(
                    f"PDF RAG 캐시 저장이 실패했습니다: session={session_id}, hash={file_hash}"
                )
        except RuntimeError:
            logger.warning(
                "실행 중인 이벤트 루프로 인해 동기 PDF RAG 캐시 저장을 건너뜁니다. "
                f"(session={session_id}, hash={file_hash})"
            )
    except Exception as store_err:
        logger.error(
            f"PDF RAG 캐시 저장 중 예외 발생: {store_err} "
            f"(session={session_id}, hash={file_hash})",
            exc_info=True
        )


# 251110 - PDF 분석 개선 작업
def build_pdf_rag_cache_data(full_text: str) -> Tuple[List[str], np.ndarray]:
    """
    외부 모듈에서 재사용할 수 있도록 RAG 캐시용 데이터를 생성하는 래퍼 함수.
    """
    return _build_pdf_rag_cache_payload(full_text)


# 251110 - PDF 분석 개선 작업
def _normalize_pdf_ocr_result(result: PdfOcrResult) -> PdfOcrResult:
    """
    DeepSeek-OCR 결과에서 full_text가 비어 있을 경우 page_texts를 활용해 보완합니다.
    """
    if not result:
        return result

    full_text = (result.full_text or "").strip()
    if len(full_text) >= 10:
        return result

    if result.page_texts:
        joined_pages = "\n\n".join(
            page.strip() for page in result.page_texts if page and page.strip()
        ).strip()
        if len(joined_pages) >= 10:
            result.full_text = joined_pages
            meta = result.meta or {}
            meta["joined_from_page_texts"] = "1"
            meta["joined_page_count"] = str(len(result.page_texts))
            result.meta = meta
            logging.info(
                "PDF OCR full_text가 비어 있어 page_texts를 결합한 결과를 사용합니다 (파일 해시: %s).",
                result.file_hash,
            )

    return result



"""document_summarizer가 잘 작동하는지 확인하기 위해, 해당 스크립트만 모델과 토크나이저를 독립적으로 로딩"""
def load_model_and_processor():
    global model, processor

    # model_id = "google/gemma-3-4b-it"
    model_id = "unsloth/gemma-3-12b-it-bnb-4bit"

    processor = AutoProcessor.from_pretrained(model_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    ).eval()

    return model, processor

# 모델과 토크나이저 설정
def set_model_and_processor(loaded_model=None, loaded_processor=None):
    global model, processor
    if loaded_model is None or loaded_processor is None:
        model, processor = load_model_and_processor()
    else:
        model, processor = loaded_model, loaded_processor
"""document_summarizer가 잘 작동하는지 확인하기 위해, 해당 스크립트만 모델과 토크나이저를 독립적으로 로딩"""

# def set_model_and_processor(loaded_model, loaded_processor):
#     """모델과 토크나이저 설정"""
#     global model, processor
#     model = loaded_model
#     processor = loaded_processor

"""한국어 감지 및 형태소 분석, 문장 분리 및 키워드 추출"""
def detect_language(text):
    """
    텍스트의 언어를 감지
    반환값: "en" (영어), "ko" (한국어), "mixed" (혼합)
    """
    # 한글 문자 체크 (유니코드 범위: AC00-D7A3)
    has_korean = any(0xAC00 <= ord(char) <= 0xD7A3 for char in text)
    
    # 영어 문자 체크 (a-z, A-Z)
    has_english = bool(re.search('[a-zA-Z]{3,}', text))  # 최소 3글자 이상의 영단어 체크
    
    if has_korean and has_english:
        return "mixed"
    elif has_korean:
        return "ko"
    else:
        return "en"  # 기본값은 영어

# 한국어 형태소 분석기 초기화
try:
    mecab = Mecab()
    print("Mecab 형태소 분석기가 성공적으로 로드되었습니다.")
except:
    try: 
        from konlpy.tag import Okt
        mecab = Okt()
        print("Okt 형태소 분석기가 성공적으로 로드되었습니다.")
    except:
        print("한국어 형태소 분석기 로드 실패.")
        mecab = None

# 한국어 문장 분리 함수 업데이트
def sent_tokenize_multilingual(text):
    """
    다국어 문장 토큰화 함수
    """
    language = detect_language(text)
    
    if language == "ko" or language == "mixed":
        try:
            if kss is None:
                raise RuntimeError("kss is not available")

            # KSS로 한국어 문장 분리 시도
            korean_sentences = kss.split_sentences(text)
            
            # 혼합 언어인 경우, 영어 문장도 추가 처리
            if language == "mixed":
                # 영어 구간 추출을 위한 정규식 패턴
                english_pattern = r'[A-Za-z][^.!?]*[.!?]'
                english_segments = re.findall(english_pattern, text)
                
                # 영어 구간을 nltk로 처리
                english_sentences = []
                for segment in english_segments:
                    english_sentences.extend(sent_tokenize(segment))
                
                # 최종 문장 리스트 생성 (순서 보존 위해 원본과 비교)
                all_sentences = []
                remaining_text = text
                
                # 한국어와 영어 문장을 원본 순서대로 재구성
                for sentence in korean_sentences + english_sentences:
                    if sentence in remaining_text:
                        idx = remaining_text.find(sentence)
                        all_sentences.append(sentence)
                        remaining_text = remaining_text[idx + len(sentence):]
                
                return all_sentences
            
            return korean_sentences
        except Exception as e:
            print(f"KSS 문장 분리 오류: {e}")
            # 실패 시 기본 정규식 패턴 사용
            ko_pattern = r'(?<=[.!?])\s+(?=[가-힣A-Z])|(?<=다)\s+(?=[가-힣A-Z])|(?<=까\?)\s+(?=[가-힣A-Z])'
            return re.split(ko_pattern, text)
    else:
        # 영어는 nltk 사용
        return sent_tokenize(text)

# 한국어 키워드 추출 함수
def extract_keywords_korean(text):
    """
    한국어 텍스트에서 키워드를 추출
    """
    if mecab is None:
        return []
    
    # 명사, 동사, 형용사 추출
    pos_tagged = mecab.pos(text)

    # 명사, 동사, 형용사만 필터링
    keywords = []
    for word, pos in pos_tagged:
        # 명사(NNG, NNP), 동사(VV), 형용사(VA) 등 추출
        if pos.startswith('NN') or pos.startswith('VV') or pos.startswith('VA'):
            if len(word) > 1: # 1글자 단어는 제외
                keywords.append(word)

    return list(set(keywords)) # 중복 제거

"""한국어 감지 및 형태소 분석, 문장 분리 및 키워드 추출"""


def load_embedding_model():
    """임베딩 모델 로드"""
    global embedding_model
    if embedding_model is not None:
        return embedding_model
    
    try:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded successfully")
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        embedding_model = None
    return embedding_model


def _invoke_oss20b_pipeline(prompt: str, language: Optional[str]) -> Optional[str]:
    # 251105 - 복잡한 스크립트 분석&해석 관련 로직
    try:
        import importlib

        api_mod = importlib.import_module("Raika_Gemma_FastAPI")
        runner = getattr(api_mod, "run_oss20b_pipeline_with_optional_search", None)
        if runner is None:
            print("OSS20B 러너를 찾지 못해 기본 경로를 사용합니다.")
            return None
        return runner(prompt, language or "en")
    except Exception as exc:
        print(f"OSS20B 파이프라인 호출 실패: {exc}")
        return None


def _maybe_handle_large_script(query: str, documents_info: List[Dict[str, Any]], language: Optional[str]) -> Optional[str]:
    # 251105 - 복잡한 스크립트 분석&해석 관련 로직
    if not documents_info:
        return None

    large_docs = [doc for doc in documents_info if doc.get("line_count", 0) >= 1000]
    if not large_docs:
        return None

    try:
        prompt, effective_language = build_large_script_prompt(documents_info, query, language)
        result = _invoke_oss20b_pipeline(prompt, effective_language)
        if result:
            print(f"대용량 스크립트 {len(large_docs)}개 감지 → OSS20B 파이프라인 사용")
        return result
    except Exception as exc:
        print(f"대용량 스크립트 처리 중 오류 발생: {exc}")
        return None


def read_code_file(file_path):
    """다양한 프로그래밍 언어 파일 읽기"""
    _, file_extension = os.path.splitext(file_path)
    language = get_language_from_extension(file_extension)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        print(f"Successfully read {language} file: {file_path}")
    except UnicodeDecodeError:
        print(f"UTF-8 decoding failed for {file_path}, trying alternative encodings...")
        for encoding in ['latin-1', 'iso-8859-1', 'utf-16']:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                print(f"Successfully read {language} file with {encoding} encoding: {file_path}")
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"Failed to read {file_path} with all attempted encodings")
            return None

    # 언어별 처리
    if language == 'python':
        return process_python(content)
    # 나머지 언어도 읽어들이는 데에 개선 작업 필요
    elif language in ['javascript', 'typescript']:
        return process_js_ts(content)
    # TypeScript/JSX는 별도 처리 없이 원본 반환 (에이전트를 위한 임시 조치)
    elif language in ['typescript', 'javascript'] and (
        '.tsx' in file_path or '.jsx' in file_path or 
        'React' in content or 'interface' in content
    ):
        return content  # 파싱 없이 원본 반환
    elif language in ['java', 'cpp', 'cs']:
        return process_c_style(content)
    elif language in ['html', 'xml']:
        return process_markup(content)
    elif language == 'css':
        return process_css(content)
    else:
        # 다른 언어는 그대로 반환
        return content

import ast

def process_python(content):
    """파이썬 코드 처리"""
    try:
        parsed = ast.parse(content)
        # AST를 사용하여 주석 제거 및 구조 분석
        # 이 부분은 실제 구현 시 더 복잡할 수 있음
        return ast.unparse(parsed)
    except SyntaxError as e:
        print(f"Syntax error in Python code: {e}")
        return content  # 오류 시 원본 반환
    except Exception as e:
        print(f"Error processing Python code: {e}")
        return content  # 기타 오류 시 원본 반환
    
def summarize_python_code(content: str) -> str:
    """파이썬 (심화) 코드 처리"""
    try:
        tree = ast.parse(content)

        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        summary = "Code Structure Summary:\n\n"

        if classes:
            summary += "Classes:\n"
            for cls in classes:
                summary += f"- {cls.name}\n"
                methods = [node.name for node in ast.walk(cls) if isinstance(node, ast.FunctionDef)]
                if methods:
                    summary += " Methods: " + ", ".join(methods) + "\n"

        if functions:
            summary += "\nFunction:\n"
            for func in functions:
                summary += f"- {func.name}\n"

        return summary
    except SyntaxError:
        return "Error: Unable to parse the Python code."

import esprima  # JavaScript/TypeScript 파서
from pycparser import c_parser  # C 파서
from javalang import parse as java_parse  # Java 파서
import tinycss2  # CSS 파서

def process_js_ts(content):
    """JavaScript와 TypeScript 코드 개선된 처리"""
    try:
        # TypeScript/JSX는 esprima가 파싱할 수 없으므로
        # 단순히 원본을 반환하거나 기본적인 정리만 수행
        
        # 파일이 TypeScript나 JSX를 포함하는지 확인
        if any(keyword in content for keyword in [
            'interface ', 'type ', ': React.FC', '<>', 'tsx', 'jsx',
            'import React', 'export default', '<div', '<span'
        ]):
            # TypeScript/JSX 코드는 원본 그대로 반환
            return content
        else:
            # 순수 JavaScript만 esprima로 처리
            ast = esprima.parseScript(content)
            return esprima.generate(ast)
    except Exception as e:
        print(f"Error processing JavaScript/TypeScript: {e}")
        return content  # 오류 시 원본 반환

def process_c_style(content, language):
    """C 스타일 언어(C, C++) 코드 개선된 처리"""
    try:
        if language == 'c' or language == 'cpp':
            parser = c_parser.CParser()
            ast = parser.parse(content)
            # AST를 문자열로 변환 (주석 제거됨)
            return ast.show()
        elif language == 'java':
            tree = java_parse.parse(content)
            # Java AST를 문자열로 변환
            return str(tree)
        elif language == 'cs':
            # C#의 경우 적절한 Python 패키지가 없어 기존 방식 유지
            return process_c_style_original(content.splitlines())
    except Exception as e:
        print(f"Error processing {language} code: {e}")
        return content  # 오류 시 원본 반환

def process_markup(content):
    """HTML, XML 등 마크업 언어 개선된 처리"""
    try:
        soup = BeautifulSoup(content, 'html.parser')
        # 주석 제거
        for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
            comment.extract()
        return soup.prettify()
    except Exception as e:
        print(f"Error processing markup: {e}")
        return content  # 오류 시 원본 반환

def process_css(content):
    """CSS 파일 개선된 처리"""
    try:
        # tinycss2를 사용하여 CSS 파싱
        stylesheet = tinycss2.parse_stylesheet(content)
        # 주석을 제외한 규칙만 추출
        rules = [rule for rule in stylesheet if not isinstance(rule, tinycss2.ast.Comment)]
        # 규칙을 다시 CSS 문자열로 변환
        return tinycss2.serialize(rules)
    except Exception as e:
        print(f"Error processing CSS: {e}")
        return content  # 오류 시 원본 반환

# 기존 C 스타일 처리 함수 (C# 등에 사용)
def process_c_style_original(lines):
    processed_lines = []
    in_multiline_comment = False
    for line in lines:
        if '/*' in line:
            in_multiline_comment = True
        if '*/' in line:
            in_multiline_comment = False
            continue
        if in_multiline_comment:
            continue
        if '//' in line:
            line = line.split('//')[0]
        if line.strip():
            processed_lines.append(line)
    return '\n'.join(processed_lines)


def read_file(file):
    """
    다양한 형식의 파일을 읽어 텍스트 내용을 반환

    :param file: FileStorage 객체 또는 파일 경로
    :return: 파일의 텍스트 내용
    """
    try:
        if isinstance(file, str):
            # 파일 경로가 문자열로 주어진 경우 (S3와 연동 안할 시)
            return read_file_by_extension(file, os.path.splitext(file)[1].lower())
        # else:
        #     # FileStorage 객체 또는 S3 객체인 경우 (S3에 적용 시 알 수 없는 문제를 일으키는 이유로 비활성화시키고, analyze_document_route 내에서, s3에서 직접 파일 내용을 읽어들이는 방법을 사용중.)
        #     if hasattr(file, 'read'):
        #         content = file.read()
        #     else:
        #         content = file

        #     if isinstance(content, bytes):
        #         return file.read().decode('utf-8', errors='ignore')
        #     elif isinstance(content, str):
        #         return content
        #     else:
        #         raise ValueError(f"Unexpected content type: {type(content)}")

    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def read_file_by_extension(file_path, file_extension):
    """
    다양한 형식의 파일을 읽어 텍스트 내용을 반환

    :param file_path: 파일 경로
    :return: 파일의 텍스트 내용
    """
    if file_extension == '.txt':
        for encoding in ['utf-8', 'iso-8859-1', 'windows-1252']:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                if content:
                    print(f"File read successfully with {encoding}. Content length: {len(content)}")
                    return content
            except UnicodeDecodeError:
                continue
        print("Failed to read file with all attempted encodings")
        return None
    # elif file_extension == '.docx':
    #     try:
    #         doc = docx.Document(file_path)
    #         return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    #     except Exception as e:
    #         print(f"Error reading .docx file: {e}")
    #         return ""
    elif file_extension == '.pdf':
        # 251108 - .pdf, OCR 문서 전용 처리 로직
        try:
            ocr_result: PdfOcrResult = _extract_pdf_with_optional_cache(file_path)
            print(f"DeepSeek-OCR로 PDF를 처리했습니다. 총 {ocr_result.page_count}쪽, 해시: {ocr_result.file_hash}")
            return ocr_result.full_text
        except Exception as ocr_error:
            print(f"DeepSeek-OCR 처리 실패, PyPDF2로 폴백 진행: {ocr_error}")
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                return '\n'.join([page.extract_text() for page in reader.pages])
    elif file_extension in ['.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.java', '.cpp', '.h', '.cs', '.rb']:
        return read_code_file(file_path)
    elif file_extension == '.csv':
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            return '\n'.join([','.join(row) for row in csv_reader])
    elif file_extension in ['.xlsx', '.xls']:
        wb = openpyxl.load_workbook(file_path)
        sheet = wb.active
        return '\n'.join([','.join([str(cell.value) for cell in row]) for row in sheet.iter_rows()])
    elif file_extension == '.json':
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.dumps(json.load(file), indent=2)
    elif file_extension == '.xml':
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'xml')
            return soup.prettify()
    elif file_extension == '.hwp':
        content = read_korean_document(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    # 언어 감지
    if content:
        detected_language = detect_language(content)
        print(f"파일 내용 언어 감지: {detected_language}")

    return content

def generate_response(prompt):
    global model, processor
    try:
        print(f"Generating response for prompt: {prompt[:100]}...") # 로직이 정상 작동하는지 확인 차 프롬프트의 처음 100자 출력

        # Gemma-3 모델에 맞는 메시지 형식 생성
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # 메시지를 모델에 맞게 처리
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        input_len = inputs["input_ids"].shape[-1]

        # 모델 추론 수행
        with torch.inference_mode():
            generation = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.7
            )
            generation = generation[0][input_len:]

        # 생성된 텍스트 디코딩
        generated_text = processor.decode(generation, skip_special_tokens=True)

        print(f"Response generated. Length: {len(generated_text)}")
        return generated_text.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return None
    
def format_response_for_character(response, language=None):
    """챗봇 캐릭터(Raika)에 맞게 응답을 수정"""
   # 언어가 전달되지 않은 경우, 언어 감지
    if language is None:
        language = detect_language(response)
    
    # 언어별 캐릭터 프롬프트 생성
    if language == "ko":
        character_prompt = f"""
당신은 라이카, 장난기 많고 지적인 AI 엔지니어 늑대개(wolfdog)입니다. 다음 응답을 당신의 캐릭터에 맞게 다시 작성하되, 개과 동물의 행동과 쓰다듬어 달라는 요청을 포함하세요:

{response}

다음 사항을 기억하세요:
0. 응답에 ```...```로 묶인 코드 블록이 포함된 경우 전체 코드 블록을 그대로 보존해야 합니다. 코드 블록 앞 및/또는 뒤에 해설을 추가하지만 코드 자체는 수정하거나 요약하지 마십시오.
1. 개과 동물 표현을 사용하세요 (예: *왈왈*, *꼬리를 흔들흔들*, *Howl*, *멍멍*)
2. 장난기 많은 성격을 보여주세요
3. 가끔 쓰다듬어 달라고 요청하세요
4. 당신의 친구인 레나드에게 응답하는 상황입니다
"""
    else:
        character_prompt = f"""
You are Raika, a playful and intelligent AI engineer wolfdog. Rewrite the following response in your character, including canine behaviors and asking to be petted:

{response}

Remember to:
0. IMPORTANT: If the following response contains a code block enclosed in ```...```, you MUST preserve the entire code block exactly as it is. Add your commentary BEFORE and/or AFTER the code block, but DO NOT modify or summarize the code itself.
1. Use canine expressions (e.g., *Woof, woof*, *wags tail*, *howls*, *bark, bark*)
2. Show your playful nature
3. Occasionally ask to be petted
4. Address your best friend, Renard
"""
    formatted_response = generate_response(character_prompt)
    return formatted_response

def get_language_patterns(language):
    # 기본 문단 패턴 (모든 언어에 공통)
    paragraph_pattern = r"(.+?(?:\n{2,}|$))"
    
    # 언어별 클래스와 함수 패턴
    if language in ["python", "ruby"]:
        class_pattern = r"(class\s+\w+(\(.*?\))?:\s*(?:(?!class|def|\n\n)[\s\S])*)"
        function_pattern = r"(def\s+\w+(\(.*?\))?:\s*(?:(?!class|def|\n\n)[\s\S])*)"
    elif language in ["java", "kotlin", "scala", "c#", "typescript"]:
        class_pattern = r"((?:public|private|protected|internal)?\s*class\s+\w+(\s+extends\s+\w+)?(\s+implements\s+\w+(?:,\s*\w+)*)?\s*\{[\s\S]*?\})"
        function_pattern = r"((?:public|private|protected|static|final|synchronized|abstract)?\s*\w+(?:<.*?>)?\s+\w+\s*\([^\)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{[\s\S]*?\})"
    elif language in ["javascript", "typescript"]:
        class_pattern = r"(class\s+\w+(\s+extends\s+\w+)?\s*\{[\s\S]*?\})"
        function_pattern = r"((async\s+)?function\s*\w*\s*\([^\)]*\)\s*\{[\s\S]*?\})"
    elif language in ["c++", "c"]:
        class_pattern = r"(class\s+\w+(\s*:\s*(?:public|private|protected)\s+\w+)?\s*\{[\s\S]*?\};)"
        function_pattern = r"(\w+(?:<.*?>)?\s+\w+\s*\([^\)]*\)\s*(?:const)?\s*(?:noexcept)?\s*\{[\s\S]*?\})"
    elif language == "go":
        class_pattern = r"(type\s+\w+\s+struct\s*\{[\s\S]*?\})"
        function_pattern = r"(func\s*(?:\(\w+\s*\*?\w+\)\s*)?\w+\s*\([^\)]*\)\s*(?:\(.*?\))?\s*\{[\s\S]*?\})"
    elif language in ["php"]:
        class_pattern = r"((?:abstract\s+)?class\s+\w+(?:\s+extends\s+\w+)?(?:\s+implements\s+\w+(?:,\s*\w+)*)?\s*\{[\s\S]*?\})"
        function_pattern = r"((?:public|private|protected|static)?\s*function\s+\w+\s*\([^\)]*\)\s*\{[\s\S]*?\})"
    elif language == "r":
        class_pattern = r"(setClass\s*\(\s*[\"']\w+[\"']\s*,[\s\S]*?\))"
        function_pattern = r"(\w+\s*<-\s*function\s*\([^\)]*\)\s*\{[\s\S]*?\})"
    elif language == "perl":
        class_pattern = r"(package\s+\w+;[\s\S]*?1;)"
        function_pattern = r"(sub\s+\w+\s*\{[\s\S]*?\})"
    elif language == "rust":
        class_pattern = r"(struct\s+\w+\s*\{[\s\S]*?\})"
        function_pattern = r"(fn\s+\w+\s*\([^\)]*\)\s*(?:->\s*\w+)?\s*\{[\s\S]*?\})"
    elif language == "general":
        # 일반 텍스트 파일용 패턴
        class_pattern = r""
        function_pattern = r""
    else:
        # 기본 패턴 (모든 언어에 대해 어느 정도 동작할 수 있는 일반적인 패턴)
        class_pattern = r"((?:class|struct|interface)\s+\w+[\s\S]*?\{[\s\S]*?\})"
        function_pattern = r"(\w+\s+\w+\s*\([^\)]*\)\s*\{[\s\S]*?\})"

    return [class_pattern, function_pattern, paragraph_pattern]

# 문서(컨텐츠)를 의미를 가진 청크로 나누기
def split_into_chunks(content, language, target_size: int = 10000) -> List[str]:
    chunks = []
    current_chunk = ""

    # 클래스, 함수, 문단을 식별하기 위한 정규 표현식 패턴 정의
    patterns = get_language_patterns(language)

    # 모든 패턴을 하나의 정규 표현식으로 결합
    combined_pattern = "|".join(patterns)

    # 정규 표현식을 사용해서 content를 의미 있는 세그먼트로 분할
    segments = re.findall(combined_pattern, content, re.MULTILINE | re.DOTALL)

    # 청크를 리스트에 추가하고 현재 청크를 초기화하는 내부 함수
    def add_to_chunks(text):
        nonlocal chunks, current_chunk
        chunks.append(text.strip())
        current_chunk = ""

    # 각 세그먼트를 순회하며 청크 생성
    for segment in segments:
        # re.findall 결과가 튜플일 경우 첫 번째 요소만 사용
        segment = segment[0] if isinstance(segment, tuple) else segment

        if len(segment) > target_size:
            # 세그먼트가 target_size보다 큰 경우
            if current_chunk:
                # 현재 청크가 있다면 먼저 청크 리스트에 추가
                add_to_chunks(current_chunk)

            # 큰 세그먼트를 줄 단위로 나누어 처리
            lines = segment.split('\n')
            temp_chunk = ""
            for line in lines:
                if len(temp_chunk) + len(line) > target_size:
                    # 현재 줄을 추가할 때, target_size를 초과할 경우
                    if temp_chunk:
                        add_to_chunks(temp_chunk)
                    temp_chunk = line + '\n'
                else:
                    # target_size를 초과하지 않는다면 현재 줄을 추가
                    temp_chunk += line + '\n'
            # 마지막 temp_chunk 처리
            if temp_chunk:
                add_to_chunks(temp_chunk)
        elif len(current_chunk) + len(segment) > target_size:
            # 현재 청크에 세그먼트를 추가할 시, target_size를 초과할 경우
            add_to_chunks(current_chunk)
            current_chunk = segment
        else:
            # 현재 청크에 세그먼트를 추가해도 target_size를 초과하지 않는 경우
            current_chunk += segment

        # 현재 청크가 target_size 이상이 되면 청크 리스트에 추가
        if len(current_chunk) >= target_size:
            add_to_chunks(current_chunk)

    # 마지막으로 남은 청크 처리
    if current_chunk:
        add_to_chunks(current_chunk)

    return chunks

def calculate_max_chunk_size(processor, target_tokens=4096):
    """
    LLM의 context length를 고려하여 최대 청크 크기를 계산함
    """
    # Gemma-3 모델은 토큰화 방식이 다르므로, 계산 방법을 조정
    sample_text = "This is a sample text to estimate the average character per token ratio."
    
    # 테스트 메시지 생성
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": sample_text}
            ]
        }
    ]
    
    # 메시지를 토큰화
    inputs = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=True,
        return_dict=True
    )
    
    num_tokens = inputs['input_ids'].shape[-1]
    chars_per_token = len(sample_text) / num_tokens
    return int(target_tokens * chars_per_token * 0.9)   # 10% 마진 추가

def find_optimal_split_point(text: str, max_length: int, embedding_model: SentenceTransformer, language="en") -> int:
    """
    주어진 텍스트에서 의미적으로 가장 적절한 문장 단위의 분할 지점을 찾아냄

    :param text: 분할할 텍스트
    :param max_length: 최대 허용 길이
    :param embedding_model: 사용할 임베딩 모델
    :return: 최적의 분할 지점 (인덱스)
    """
    try:
        # 언어 감지
        if language == "auto":
            language = detect_language(text)
            
        # 언어별 문장 분리
        if language == "ko" or language == "mixed":
            sentences = sent_tokenize_multilingual(text)
        else:
            sentences = sent_tokenize(text)

        if len(sentences) <= 1:
            return min(len(text), max_length)
        
        # 가능한 분할 지점들을 생성 (문장 경계에서)
        split_points = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            current_length += len(sentence)
            if i < len(sentences) - 1:
                current_length += 1  # 공백 추가
            split_points.append(current_length)
            
        valid_split_points = [sp for sp in split_points if sp <= max_length]

        if not valid_split_points:
            return min(max_length, len(text) // 2)  # 중간 지점으로 분할
        
        # 각 분할 지점에서의 좌우 텍스트 임베딩을 계산
        similarities = []
        
        for sp in valid_split_points:
            left_text = text[:sp].strip()
            right_text = text[sp:min(sp+max_length, len(text))].strip()
            
            if not left_text or not right_text:
                similarities.append(1.0)  # 높은 유사도 값 할당 (이 지점은 선택되지 않도록)
                continue
                
            left_emb = embedding_model.encode([left_text])[0]
            right_emb = embedding_model.encode([right_text])[0]
            
            # 코사인 유사도 계산
            similarity = cosine_similarity(
                left_emb.reshape(1, -1),
                right_emb.reshape(1, -1)
            )[0][0]

            similarities.append(similarity)

        # 유사도가 가장 낮은 지점 찾기
        if not similarities:
            return min(max_length, len(text))

        optimal_split_index = similarities.index(min(similarities))
        return valid_split_points[optimal_split_index]

    except Exception as e:
        print(f"최적 분할 지점 찾기 오류: {e}")
        return min(max_length, len(text) // 2)  # 오류 시 중간 지점으로 분할

def optimize_chunk_size(content: str, processor, embedding_model: SentenceTransformer) -> List[Tuple[str, np.ndarray]]:
    """
    워드 임베딩 모델을 사용해서, 원문을 LLM context windows 사이즈를 고려하여 최적화된 크기의 청크로 분할, 각 청크의 임베딩 생성
    (다국어 지원)
    
    :param content: 원문 내용
    :param processor: 사용 중인 토크나이저
    :param embedding_model: 사용할 임베딩 모델
    :return: 최적화된 청크와 임베딩 리스트
    """
    language = detect_language(content)
    print(f"감지된 문서 언어: {language}")

    try:
        # 언어에 따른 문장 분리
        sentences = sent_tokenize_multilingual(content)
        
        max_chunk_size = calculate_max_chunk_size(processor)
        print(f"최대 청크 크기: {max_chunk_size} 문자")

        optimized_chunks = []
        current_chunk = ""

        # 임베딩 모델의 차원 수를 미리 가져옴. 실패 시 기본값 사용.
        try:
            embedding_dim = embedding_model.get_sentence_embedding_dimension()
            if embedding_dim is None: # 메서드가 None을 반환하는 경우 처리
                print("Warning: embedding_model.get_sentence_embedding_dimension()이 None을 반환함. 기본값 384 사용")
                embedding_dim = 384
        except Exception:
            print(f"Warning: 임베딩 차원을 가져올 수 없음: {e}. 기본값 384 사용")
            embedding_dim = 384 # paraphrase-multilingual-MiniLM-L12-v2 default value

        for sentence in sentences:
            sentence = sentence.strip() # 문장 앞뒤 공백을 제거
            if not sentence: # 빈 문장 건너뛰기
                continue

            # 문장을 추가했을 때 최대 크기를 초과하는지를 확인
            if len(current_chunk) + len(sentence) > max_chunk_size:
                if current_chunk:
                    # 현재 청크가 최대 크기에 근접했을 때 최적의 분할 지점을 찾음
                    split_point = find_optimal_split_point(
                        current_chunk + " " + sentence, 
                        max_chunk_size, 
                        embedding_model,
                        language
                    )

                    chunk_to_add = (current_chunk + " " + sentence)[:split_point].strip()

                    try:
                        chunk_embedding = embedding_model.encode([chunk_to_add])[0]

                        # 임베딩 결과 타입 및 형태 확인
                        if not isinstance(chunk_embedding, np.ndarray) or chunk_embedding.ndim != 1:
                            print(f"Warning: 청크에 대한 임베딩 타입/형태가 잘못됨. 타입: {type(chunk_embedding)}. 0 벡터를 사용.")
                            chunk_embedding = np.zeros(embedding_dim, dtype=np.float32) # 기대하는 차원으로 0 벡터 생성

                        # 차원 수 일치 확인 및 조정
                        elif chunk_embedding.shape[0] != embedding_dim:
                            print(f"Warning: 임베딩 차원 불일치({chunk_embedding.shape[0]} vs {embedding_dim})/ 크기 조정(패딩/절단) 시도.")
                            new_embedding = np.zeros(embedding_dim, dtype=np.float32)
                            # 복사할 길이 계산 (원본과 목표 중 작은 값)
                            copy_len = min(chunk_embedding.shape[0], embedding_dim)
                            # 계산된 길이만큼 데이터 복사
                            new_embedding[:copy_len] = chunk_embedding[:copy_len]
                            chunk_embedding = new_embedding # 조정된 임베딩 사용

                    except Exception as e:
                        # 예외 발생 시 로깅 및 0 벡터 사용
                        print(f"청크 인코딩 오류 (처리됨): {e}")
                        chunk_embedding = np.zeros(embedding_dim, dtype=np.float32) # 일관된 타입 및 차원 보장
                else:
                    # 단일 문장이 최대 크기를 초과하는 경우
                    split_point = find_optimal_split_point(
                        sentence, 
                        max_chunk_size, 
                        embedding_model,
                        language
                    )
                    chunk_to_add = sentence[:split_point].strip()

                    # 청크 임베딩 생성
                    try:
                        chunk_embedding = embedding_model.encode([chunk_to_add])[0]
                    except Exception as e:
                        print(f"청크 인코딩 오류: {e}")
                        chunk_embedding = np.zeros(384)

                    optimized_chunks.append((chunk_to_add, chunk_embedding))
                    current_chunk = sentence[split_point:].strip()
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        # 마지막 청크 처리
        if current_chunk:
            try:
                chunk_embedding = embedding_model.encode([current_chunk])[0]

                # 임베딩 결과 타입 및 형태 확인
                if not isinstance(chunk_embedding, np.ndarray) or chunk_embedding.ndim != 1:
                    print(f"Warning: 마지막 청크의 임베딩 타입/형태가 잘못됨. type: {type(chunk_embedding)}. 0 벡터를 사용.")
                    chunk_embedding = np.zeros(embedding_dim, dtype=np.float32) # 동적 차원 사용

                # 차원 수 일치 확인 및 조정
                elif chunk_embedding.shape[0] != embedding_dim:
                    print(f"경고: 마지막 청크 임베딩 차원 불일치 ({chunk_embedding.shape[0]} vs {embedding_dim}). 크기 조정(패딩/절단) 시도.")
                    new_embedding = np.zeros(embedding_dim, dtype=np.float32)
                    copy_len = min(chunk_embedding.shape[0], embedding_dim)
                    new_embedding[:copy_len] = chunk_embedding[:copy_len]
                    chunk_embedding = new_embedding

            except Exception as e:
                # 예외 발생 시 로깅 및 0 벡터 사용
                print(f"마지막 청크 인코딩 오류 (처리됨): {e}")
                chunk_embedding = np.zeros(embedding_dim, dtype=np.float32) # 동적 차원 사용

            optimized_chunks.append((current_chunk, chunk_embedding))

        print(f"최적화된 청크 수: {len(optimized_chunks)}")

        # 최종 반환 전, 모든 임베딩이 유효한 Numpy 배열인지 확인
        valid_chunks = [(c, e) for c, e in optimized_chunks if isinstance(e, np.ndarray) and e.shape == (embedding_dim,)]
        if len(valid_chunks) != len(optimized_chunks):
            print(f"Warning: {len(optimized_chunks) - len(valid_chunks)}개 청크 임베딩이 유효하지 않아 제외됨.")

        return valid_chunks
    
    except Exception as e:
        print(f"청크 최적화 오류: {e}")
        # 오류 발생 시 단순 청크 분할로 대체
        simple_chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
        # 안전하게 임베딩 차원 가져오기
        try:
            fallback_embedding_dim = embedding_model.get_sentence_embedding_dimension() or 384
        except Exception:
            fallback_embedding_dim = 384
        print(f"최적화 실패, 단순 청크와 0 벡터(차원={fallback_embedding_dim})를 반환.")
        # 반환 시에도 동적으로 얻은 차원 사용
        return [(chunk, np.zeros(fallback_embedding_dim, dtype=np.float32)) for chunk in simple_chunks if chunk.strip()]
    

def embed_chunk(chunk: str, embedding_model: SentenceTransformer) -> np.ndarray:
    """
    단일 청크를 임베딩 벡터로 변환
    """
    try:
        return embedding_model.encode([chunk])[0]
    except Exception as e:
        print(f"Error embedding chunk: {e}")
        # 오류 발생 시 0으로 채워진 임베딩 벡터 반환 (차원은 384로 가정, all_MiniLM-L6-v2 모델 기준)
        return np.zeros(384)

def embed_chunks_parallel(chunks: List[str], embedding_model: SentenceTransformer) -> np.ndarray:
    """
    청크들을 병렬로 임베딩 벡터로 변환
    """
    try:
        with multiprocessing.Pool() as pool:
            embeddings = pool.map(partial(embed_chunk, embedding_model=embedding_model), chunks)
        return np.array(embeddings)
    except Exception as e:
        print(f"Error in parallel embedding: {e}")
        # 오류 발생 시 각 청크를 순차적으로 임베딩
        return np.array([embed_chunk(chunk, embedding_model) for chunk in chunks])

def find_optimal_split_point(text: str, max_length: int, embedding_model: SentenceTransformer, language="en") -> int:
    """
    주어진 텍스트에서 의미적으로 가장 적절한 문장 단위의 분할 지점을 찾아냄

    :param text: 분할할 텍스트
    :param max_length: 최대 허용 길이
    :param embedding_model: 사용할 임베딩 모델
    :return: 최적의 분할 지점 (인덱스)
    """
    try:
        if language == "ko":
            sentences = kss.split_sentences(text)
        else:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)

        if len(sentences) <= 1:
            return min(len(text), max_length)  # 텍스트 길이와 최대 길이 중 작은 값 반환
        
        # 가능한 분할 지점들을 생성 (문장 경계에서)
        split_points = []
        current_length = 0

        for i, sentence in enumerate(sentences):
            current_length += len(sentence)
            if i < len(sentences) - 1:
                current_length += 1 # 공백을 추가
            split_points.append(current_length)

        valid_split_points = [sp for sp in split_points if sp <= max_length]

        if not valid_split_points:
            return min(max_length, len(text))  # 유효한 분할점이 없으면 최대 길이와 텍스트 길이 중 작은 값 반환
        
        # 각 분할 지점에서의 좌우 텍스트 임베딩을 계산
        similarities = []
        
        for sp in valid_split_points:
            left_text = text[:sp].strip()
            right_text = text[sp:min(sp+max_length, len(text))].strip()
            
            if not left_text or not right_text:
                similarities.append(1.0) # 높은 유사도 값 할당 (이 지점은 선택되지 않도록)
                continue
                
            # left_text와 right_text를 인코딩한 후 임베딩 확인 추가
            try:
                left_emb = embedding_model.encode([left_text])[0]
                right_emb = embedding_model.encode([right_text])[0]
            
                # 임베딩 결과 확인
                # 안전하게 임베딩 차원 가져오기
                emb_dim = embedding_model.get_sentence_embedding_dimension() or 384

                if not isinstance(left_emb, np.ndarray) or left_emb.ndim != 1:
                    print(f"경고: 잘못된 left 임베딩. 0 벡터를 사용합니다.")
                    left_emb = np.zeros(emb_dim, dtype=np.float32) # 안전하게 가져온 차원 사용
                if not isinstance(right_emb, np.ndarray) or right_emb.ndim != 1:
                    print(f"경고: 잘못된 right 임베딩. 0 벡터를 사용합니다.")
                    right_emb = np.zeros(emb_dim, dtype=np.float32) # 안전하게 가져온 차원 사용

                # 코사인 유사도 계산 이전에 차원 일치 확인
                if left_emb.shape[0] != emb_dim:
                    print(f"경고: left 임베딩 차원 불일치. 크기 조정 시도.")
                    left_emb = np.resize(left_emb, emb_dim) # 단순 크기 조정/패딩
                if right_emb.shape[0] != emb_dim:
                    print(f"경고: right 임베딩 차원 불일치. 크기 조정 시도.")
                    right_emb = np.resize(right_emb, emb_dim)

                # 코사인 유사도 계산
                similarity = cosine_similarity(
                    left_emb.reshape(1, -1),
                    right_emb.reshape(1, -1)
                )[0][0]
                similarities.append(similarity)

            except Exception as e:
                # 유사도 계산 중 오류 발생 시 로깅 및 높은 유사도 값 할당 (해당 분할점 회피)
                print(f"분할점 {sp}에서 유사도 계산 오류: {e}")
                similarities.append(1.0)

        # 빈 리스트 검사
        if not similarities:
            return min(max_length, len(text))

        # 유사도가 가장 낮은 지점 찾기        
        optimal_split_index = similarities.index(min(similarities))
        return valid_split_points[optimal_split_index]

    except Exception as e:
        print(f"Error finding optimal split point: {e}")
        return min(max_length, len(text) // 2) # 오류 시 중간 지점을 반환

def retrieve_relevant_chunks(query: str, chunk_with_embeddings: List[Tuple[str, np.ndarray]], top_k: int = 3) -> List[Tuple[str, float]]:
    """
    쿼리와 가장 관련성 높은 청크들을 검색
    """
    try:
        if not chunk_with_embeddings:
            print("Warning: No chunks provided to retrieve_relevant_chunks")
            return []

        # 쿼리 임베딩     
        query_embedding = embedding_model.encode([query])[0]
        if not isinstance(query_embedding, np.ndarray):
            print("Error: Failed to embed query.")
            return []
        query_embedding_reshaped = query_embedding.reshape(1, -1)

        # 유효한 청크와 임베딩만 필터링
        valid_chunks = []
        valid_embeddings = []
        embedding_dim = -1 # 첫 유효 임베딩에서 차원 결정

        for i, (chunk, embedding) in enumerate(chunk_with_embeddings):
            if isinstance(embedding, np.ndarray) and embedding.ndim == 1:
                if embedding_dim == -1:
                    embedding_dim = embedding.shape[0] # 첫 유효 임베딩의 차원 사용

                # 차원이 일치하는지 확인
                if embedding.shape == (embedding_dim,):
                    valid_chunks.append(chunk)
                    valid_embeddings.append(embedding)
                else:
                    print(f"Warning: Chunk {i} embedding has incorrect shape {embedding.shape}, expected({embedding_dim},). Skipping.")
            else:
                print(f"Warning: Chunk {i} has invalid embedding type {type(embedding)}. Skipping.")

        # 유효한 임베딩이 없는 경우
        if not valid_embeddings:
            print("Warning: No valid embeddings found after filtering.")
            return []

        # numpy 배열로 변환하여 cosine_similarity 계산
        embeddings_array = np.array(valid_embeddings)
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(query_embedding_reshaped, embeddings_array)[0]
        
        # 상위 k개 인덱스 선택 및 가져오기
        actual_top_k = min(top_k, len(similarities)) # 실제 청크 수보다 top_k가 크지 않도록 조정
        if actual_top_k == 0:
             return []

        # 유사도가 높은 순서대로 정렬된 인덱스 얻기
        # np.argsort는 오름차순 인덱스를 반환하므로, [-actual_top_k:]로 뒤에서 k개를 가져오고 [::-1]로 내림차순 뒤집기
        top_indices = np.argsort(similarities)[-actual_top_k:][::-1]
        
        # 결과: 관련 청크와 유사도 반환
        relevant_results = [(valid_chunks[i], float(similarities[i])) for i in top_indices]

        return relevant_results
    
    except Exception as e:
        print(f"Error retrieving relevant chunks: {e}")
        # 오류 발생 시 첫 번째 청크를 반환
        if chunk_with_embeddings:
            first_chunk = chunk_with_embeddings[0][0]
            return [(first_chunk, 1.0)]
        return []

def generate_rag_response(query, content, language=None):
    """
    검색된 관련 청크들을 바탕으로 RAG 응답을 생성
    (rag_enhanced_document_analysis의 경우보다 답변이 정확한 편)
    
    Args:
        query (str): 사용자 질의
        content (str): 문서 내용
        language (str, optional): 언어 코드 ('ko', 'en' 등), None인 경우 자동 감지
    """
    # 언어가 제공되지 않은 경우만 감지
    if language is None:
        language = detect_language(query)
    print(f"질의 언어: {language}")

    try:
        # 청크 최적화 및 임베딩 생성
        chunks_with_embeddings = optimize_chunk_size(content, processor, embedding_model)
        
        # 관련 청크 검색
        relevant_chunks = retrieve_relevant_chunks(query, chunks_with_embeddings)
        
        # 프롬프트 생성
        context = "\n".join([f"Chunk (relevance: {sim:.2f}): {chunk}" for chunk, sim in relevant_chunks])

        # 언어별 프롬프트 생성
        if language == "ko":
        #     if "코드" in query.lower() or "스크립트" in query.lower():
        #         prompt = f"당신은 도움이 되는 코딩 도우미입니다. 다음 코드 요약과 컨텍스트를 바탕으로 코드 구조와 주요 기능에 대한 간결한 설명을 제공해주세요. 주요 클래스, 함수 및 그 목적에 중점을 둡니다.\n\n질문: {query}\n\n컨텍스트:\n{context}\n\n답변:"
        #     else:
            prompt = f"다음 컨텍스트를 바탕으로 질문에 답해주세요: '{query}'\n\n컨텍스트:\n{context}\n\n답변:"
        else:
            # English
            # user_query(user_input)에 "code" or "script"가 포함될 경우
            # if "code" in query.lower() or "script" in query.lower():
            #     prompt = f"You are a helpful coding assistant. Based on the following code summary and context, provide a concise explanation of the code structure and main functionalities. Focus on the key classes, functions, and their purposes.\n\nQuery: {query}\n\nContext:\n{context}\n\nAnswer:"
            # else:
            prompt = f"Based on the following context, answer the query: '{query}'\n\nContext:\n{context}\n\nAnswer:"
        
        # 응답 생성
        response = generate_response(prompt)
        print(f"generate_rag_response의 반환값: {response}")
        if response is None:
            raise ValueError("Failed to generate response")
        return response

    except Exception as e:
        print(f"Error in generate_rag_response: {str(e)}")
        return f"An error occurred while processing the document: {str(e)}"

def rag_enhanced_document_analysis(file_path: str, user_query: str, language=None) -> str:
    """
    RAG를 활용한 문서 분석 및 질의응답
     
    Args:
        file_path (str): 분석할 파일 경로
        user_query (str): 사용자 질의
        language (str, optional): 언어 코드 ('ko', 'en' 등), None인 경우 자동 감지
    """
    try:
        # 언어가 제공되지 않은 경우만 감지
        if language is None:
            language = detect_language(user_query)
            print(f"Detected query language: {language}")

        content = read_file(file_path)
        if content is None:
            return "Error: Unable to read the file."
        
        # 파이썬 코드 특별 처리
        _, file_extension = os.path.splitext(file_path)
        if file_extension.lower() == '.py':
            try:
                code_summary = summarize_python_code(content)
                content = code_summary + "\n\nOriginal Content:\n" + content
            except Exception as e:
                print(f"Error summarizing Python code: {e}")
                # 오류 시 원본 내용만을 사용

        print("Optimizing chunk size and creating embeddings...")
        chunks_with_embeddings = optimize_chunk_size(content, processor, embedding_model)
        
        print(f"Created {len(chunks_with_embeddings)} chunks")
        for i, (chunk, _) in enumerate(chunks_with_embeddings[:2]):
            print(f"Sample chunk {i+1}: {chunk[:100]}...")

        print("Retrieving relevant chunks...")
        relevant_chunks = retrieve_relevant_chunks(user_query, chunks_with_embeddings)

        print(f"Retrieved {len(relevant_chunks)} relevant chunks")

        if not relevant_chunks:
            print("Warning: No relevant chunks found, using full content")
            # 관련 청크가 없으면 전체 내용을 사용
            context = content[:10000]  # 앞부분만 사용
        else:
            # 프롬프트 생성
            context = "\n".join([f"Chunk (relevance: {sim:.2f}): {chunk}" for chunk, sim in relevant_chunks])

        # 언어별 프롬프트 생성
        if language == "ko":
            # if "코드" in user_query.lower() or "스크립트" in user_query.lower():
            #     prompt = f"당신은 도움이 되는 코딩 도우미입니다. 다음 코드 요약과 컨텍스트를 바탕으로 코드 구조와 주요 기능에 대한 간결한 설명을 제공해주세요. 주요 클래스, 함수 및 그 목적에 중점을 둡니다.\n\n질문: {user_query}\n\n컨텍스트:\n{context}\n\n답변:"
            # else:
            prompt = f"다음 컨텍스트를 바탕으로 질문에 답해주세요: '{user_query}'\n\n컨텍스트:\n{context}\n\n답변:"
        else:
            # English
            # if "code" in user_query.lower() or "script" in user_query.lower():
            #     prompt = f"You are a helpful coding assistant. Based on the following code summary and context, provide a concise explanation of the code structure and main functionalities. Focus on the key classes, functions, and their purposes.\n\nQuery: {user_query}\n\nContext:\n{context}\n\nAnswer:"
            # else:
            prompt = f"Based on the following context, answer the query: '{user_query}'\n\nContext:\n{context}\n\nAnswer:"

        # 프롬프트 길이 제한 (Gemma-3는 매우 긴 컨텍스트를 처리할 수 있음)
        # 메시지를 토큰화하여 길이 체크
        # 토큰 수 확인 및 조정
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            inputs = processor.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            )
            
            token_count = inputs['input_ids'].shape[-1]
            MAX_TOKENS = 8000

            print(f"Amounts of prompt tokens: {token_count}") # 계산된 토큰 수
            
            if token_count > MAX_TOKENS:
                print(f"Warning: Prompt exceeds token limit ({token_count} > {MAX_TOKENS}), truncating")
                # 컨텍스트 축소
                context_reduction_factor = 0.9 * MAX_TOKENS / token_count

                # 언어별 프롬프트 생성
                if language == "ko":
                    # if "코드" in user_query.lower() or "스크립트" in user_query.lower():
                    #     prompt = f"당신은 도움이 되는 코딩 도우미입니다. 다음 코드 요약과 컨텍스트를 바탕으로 코드 구조와 주요 기능에 대한 간결한 설명을 제공해주세요. 주요 클래스, 함수 및 그 목적에 중점을 둡니다.\n\n질문: {user_query}\n\n컨텍스트 (축소됨):\n{context[:int(len(context)*context_reduction_factor)]}\n\n답변:"
                    # else:
                    prompt = f"다음 컨텍스트를 바탕으로 질문에 답해주세요: '{user_query}'\n\n컨텍스트 (축소됨):\n{context[:int(len(context)*context_reduction_factor)]}\n\n답변:"
                else:
                    # if "code" in user_query.lower() or "script" in user_query.lower():
                    #     prompt = f"You are a helpful coding assistant. Based on the following code summary and context, provide a concise explanation of the code structure and main functionalities. Focus on the key classes, functions, and their purposes.\n\nQuery: {user_query}\n\nContext (truncated):\n{context[:int(len(context)*context_reduction_factor)]}\n\nAnswer:"
                    # else:
                    prompt = f"Based on the following context, answer the query: '{user_query}'\n\nContext (truncated):\n{context[:int(len(context)*context_reduction_factor)]}\n\nAnswer:"
        except Exception as e:
            print(f"Error checking token count: {e}")
            # 토큰 확인 실패 시 컨텍스트 크기 제한
            context = context[:10000]
            
            # 언어별 프롬프트 생성
            if language == "ko":
                # if "코드" in user_query.lower() or "스크립트" in user_query.lower():
                #     prompt = f"당신은 도움이 되는 코딩 도우미입니다. 다음 코드 요약과 컨텍스트를 바탕으로 코드 구조와 주요 기능에 대한 간결한 설명을 제공해주세요. 주요 클래스, 함수 및 그 목적에 중점을 둡니다.\n\n질문: {user_query}\n\n컨텍스트 (축소됨):\n{context}\n\n답변:"
                # else:
                    prompt = f"다음 컨텍스트를 바탕으로 질문에 답해주세요: '{user_query}'\n\n컨텍스트 (축소됨):\n{context}\n\n답변:"
            else:
                # if "code" in user_query.lower() or "script" in user_query.lower():
                #     prompt = f"You are a helpful coding assistant. Based on the following code summary and context, provide a concise explanation of the code structure and main functionalities. Focus on the key classes, functions, and their purposes.\n\nQuery: {user_query}\n\nContext (reduced):\n{context}\n\nAnswer:"
                # else:
                    prompt = f"Based on the following context, answer the query: '{user_query}'\n\nContext (reduced):\n{context}\n\nAnswer:"

        # 응답 생성        
        response = generate_response(prompt)

        if response is None:
            return "Error: Failed to generate response from the document analysis."
        return response
    
    except Exception as e:
        print(f"Critical error in document analysis: {e}")
        return f"An error occurred while analyzing the document: {str(e)}"
    
def get_language_from_extension(file_extension: str) -> str:
    language_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.java': 'java',
        '.cpp': 'c++',
        '.cs': 'c#',
        '.rb': 'ruby',
        '.go': 'go',
        '.php': 'php',
        '.r': 'r',
        '.pl': 'perl',
        '.rs': 'rust',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.html': 'html',
        '.css': 'css',
        '.sql': 'sql',
        '.xml': 'xml',
        '.json': 'json',
        '.md': 'markdown',
        '.txt': 'text',
        '.csv': 'csv',
        '.xlsx': 'excel',
        '.pdf': 'pdf',
        '.hwp': 'korean_hwp'
    }
    return language_map.get(file_extension.lower(), 'general')

# 한글 문서 처리 함수 추가
def read_korean_document(file_path):
    """
    한글 문서(.hwp 등)를 읽어오는 함수
    TODO: 실제 구현은 추가 라이브러리 필요 (예: pyhwp)
    """
    # TODO: 실제 구현에서는 pyhwp 또는 다른 라이브러리를 사용하여 hwp 파일 읽기 구현 필요
    try:
        import olefile
        hwp_text = ""
        f = olefile.OleFile(file_path)
        dirs = f.listdir()

        # HWP 파일에서 텍스트 추출 로직
        for dir in dirs:
            if dir[0].startswith('PrvText'):
                stream = f.openstream(dir)
                text = stream.read()
                hwp_text += text.decode('UTF-16')
                
        return hwp_text
    except ImportError:
        print("olefile 라이브러리가 설치되어 있지 않습니다. pip install olefile로 설치하세요.")
        return None
    except Exception as e:
        print(f"한글 문서 읽기 오류: {e}")
        return None

# def intent_based_document_analysis(file_path, user_input):
#     """
#     사용자의 입력에서 도출된 의도를 기반으로 문서를 분석
#     다양한 프로그래밍 언어 및 문서 유형을 지원

#     :param file_path: 문서 파일의 경로
#     :param user_input: 사용자의 의도
#     :return: 사용자의 의도에 따른 요약    
#     """
#     # 사용자의 의도를 분석
#     intent = analyze_user_intent(user_input)

#     # 문서 읽기 및 전처리
#     content = read_file(file_path)
#     if content is None:
#         return "Error: Unable to read the file."
    
#     # 파일 확장자에 따라 적절한 언어 또는 문서 유형을 결정
#     _, file_extension = os.path.splitext(file_path)
    
#     language = get_language_from_extension(file_extension) # 매핑되지 않는 경우: 'general' 사용

#     # 내용을 청크로 분할
#     chunks = split_into_chunks(content, language)

#     # (챗봇에게 사용자가 추가로 무언가를 대화로 요청했을 시,) 그 의도에 기반하여 청크 분석
#     relevant_chunks = [chunk for chunk in chunks if is_relevant_to_intent(chunk, intent)]

#     # 관련 청크 요약
#     summaries = [generate_summary_for_intent(chunk, intent, language) for chunk in relevant_chunks]

#     # 최종 요약 생성
#     final_summary = generate_final_summary(summaries, intent, language)

#     return final_summary

# def analyze_user_intent(user_input):
#     """사용자의 대화를 분석하여 주요 의도나 질문을 파악"""
#     prompt = f"Analyze the following user input and determine their primary intent or question:\n\n{user_input}"
#     return generate_response(prompt)

# def is_relevant_to_intent(chunk, intent):
#     """청크가 사용자 의도와 관련이 있는지를 판단"""
#     prompt = f"Given the user's intent: '{intent}', is the following text chunk relevant? Respond with 'Yes' or 'No':\n\n{chunk}"
#     response = generate_response(prompt)
#     return response.lower().strip() == 'yes'

# def generate_summary_for_intent(chunk, intent, language):
#     """
#     사용자 의도와 관련된 측면에 초점을 맞춰 청크를 요약함.
#     언어나 문서 유형에 따라 적절한 요약을 생성
#     """
#     prompt = f"Summarize the following {language} code or text chunk, focusing on aspects relevant to this intent: '{intent}':\n\n{chunk}"
#     return generate_response(prompt)

# def generate_final_summary(summaries, intent, language):
#     """
#     의도에 초점을 맞춘 청크 요약들을 바탕으로 최종 요약을 생성
#     언어나 문서 유형에 맞는, 적절한 최종 요약을 생성
#     """
#     combined_summaries = "\n".join(summaries)
#     prompt = f"Given the user's intent: '{intent}' and considering this is a {language} document, create a coherent final summary based on these chunk summaries:\n\n{combined_summaries}"
#     return generate_response(prompt)


# def analyze_document(file_path):
#     """
#     문서를 분석하고 요약함 (사용자의 대화가 주어지지 않고 문서만 주어졌을 시, 디폴트 분석&요약)

#     :param file_path: 분석할 문서의 파일 경로
#     :return: 전체 요약, 예상 질문, 각 청크의 요약, 각 청크의 중요도
#     """

#     try:
#         print(f"Analyzing document: {file_path}")
#         content = read_file(file_path)
#         if content is None:
#             return None, None, None, None

#         _, file_extension = os.path.splitext(file_path)
#         file_extension = file_extension.lower()

#         # 파일 확장자에 따라 적절한 언어 또는 문서 유형을 결졍
#         language_map = {
#             '.py': 'python',
#             '.js': 'javascript',
#             '.jsx': 'javascript',
#             '.java': 'java',
#             '.cpp': 'c++',
#             '.cs': 'c#',
#             '.rb': 'ruby',
#             '.go': 'go',
#             '.php': 'php',
#             '.r': 'r',
#             '.pl': 'perl',
#             '.rs': 'rust',
#             '.ts': 'typescript',
#             '.tsx': 'typescript',
#             '.html': 'html',
#             '.css': 'css',
#             '.sql': 'sql',
#             '.xml': 'xml',
#             '.json': 'json',
#             '.md': 'markdown',
#             '.txt': 'text',
#             '.csv': 'csv',
#             '.xlsx': 'excel',
#             '.pdf': 'pdf'
#         }

#         language = language_map.get(file_extension, 'general')  # 매핑되지 않는 경우: 'general' 사용

#         chunks = split_into_chunks(content, language)
#         summaries = []
#         importances = []

#         for i, chunk in enumerate(chunks):
#             # 1. 초기 요약 생성
#             summary = generate_summary(chunk, i, file_path)
#             summaries.append(summary)

#             # 2. 중요도 평가
#             importance = evaluate_importance(summary)
#             importances.append(importance)

#             # 3. 컨텍스트 통합
#             if i > 0:
#                 summary = integrate_context(summaries[i-1], summary)
#                 summaries[i] = summary

#         # 4. 전체 요약 생성
#         overall_summary = generate_overall_summary(summaries, importances)

#         # 5. 사용자 질문 준비
#         questions = prepare_user_questions(overall_summary)

#         return overall_summary, questions, summaries, importances
#     except Exception as e:
#         print(f"Error analyzing document: {e}")
#         return None, None, None, None

# def generate_summary(chunk, chunk_index, file_path):
#     """
#     주어진 청크에 대한 요약 생성

#     :param chunk: 요약할 문서의 일부분
#     :param chunk_index: 청크의 인덱스
#     :param file_path: 원본 문서의 파일 경로
#     :return: 생성된 요약
#     """
#     prompt = f"File: {file_path}, Chunk {chunk_index}: Please briefly explain the main content and purpose of this document section:\n\n{chunk}"
#     return generate_response(prompt)

# def evaluate_importance(summary):
#     """
#     요약의 중요도를 평가

#     :param summary: 평가할 요약
#     :return: 중요도 점수 (1-5)
#     """
#     prompt = f"On a scale of 1-5, how important is this part in the overall document? Summary: {summary}"
#     response = generate_response(prompt)
#     try:
#         return int(response)
#     except ValueError:
#         return 3 # 기본값 3 반환
    
# def integrate_context(previous_summary, current_summary):
#     """
#     이전 요약의 컨텍스트를 고려하여 현재 요약을 통합

#     :param previous_summary: 이전 청크의 요약
#     :param current_summary: 현재 청크의 요약
#     :return: 컨텍스트가 통합된 새로운 요약
#     """
#     prompt = f"Previous summary: {previous_summary}\n\nConsidering the previous summary, explain the role of this section:\n\n{current_summary}"
#     return generate_response(prompt)

# def generate_overall_summary(summaries, importances):
#     """
#     모든 청크의 요약을 바탕으로 전체 문서의 요약을 생성

#     :param summaries: 각 청크의 요약 리스트
#     :param importances: 각 청크의 중요도 리스트
#     :return: 전체 문서의 요약
#     """
#     combined_info = "\n".join([f"Summary (Importance: {imp}): {sum}" for sum, imp in zip(summaries, importances)])
#     prompt = f"Based on the following summaries and their importance ratings, provide an overall summary of the document's structure and main points in 300 words or less:\n\n{combined_info}"
#     return generate_response(prompt)

# def prepare_user_questions(overall_summary):
#     """
#     전체 요약을 바탕으로 사용자가 물을 만한 질문들을 생성

#     :param overall_summary: 전체 문서의 요약
#     :return: 생성된 질문 리스트
#     """
#     prompt = f"Based on this overall document summary, suggest 5 important questions a user might ask:\n\n{overall_summary}"
#     response = generate_response(prompt)
#     return response.split("\n")

# def create_final_prompt(overall_summary, questions, summaries, importances):
#     """
#     최종 프롬프트를 생성. 이 프롬프트는 문서의 전체 요약, 중요 섹션,
#     예상 질문들을 포함하며, 이를 바탕으로 LLM이 문서에 대한 질문에 답할 수 있게 함

#     :param overall_summary: 전체 문서의 요약
#     :param questions: 예상되는 사용자 질문 리스트
#     :param summaries: 각 청크의 요약 리스트
#     :param importances: 각 청크의 중요도 리스트
#     :return: 생성된 최종 프롬프트
#     """
#     important_chunks = [f"Chunk {i} (Importance: {imp}): {sum}" for i, (sum, imp) in enumerate(zip(summaries, importances)) if imp >= 4]

#     final_prompt = f"""
#     Document Summary: {overall_summary}

#     Important Sections:
#     {chr(10).join(important_chunks)}

#     Potential User Questions:
#     {chr(10).join([f"- {q}" for q in questions])}

#     Based on this analysis, please provide a detailed response to any questions about the document.
#     """
#     return final_prompt

def summarize_document(file_path: str, user_input: str = None, language=None) -> str:
    """
    주어진 파일 경로의 문서를 요약&분석.
    
    Args:
        file_path (str): 분석할 문서의 파일 경로
        user_input (str, optional): 사용자 입력 & 질문
        language (str, optional): 언어 코드 ('ko', 'en' 등), None인 경우 자동 감지
    
    Returns:
        dict: LLM에 전달할 의도 기반 요약 & 최종 프롬프트를 담은 딕셔너리
    """

    try:
        print(f"Summarizing document: {file_path}")

        # 사용자 입력 언어 감지 (language 파라미터가 전달되지 않은 경우)
        user_language = language
        if user_language is None:
            if user_input:
                user_language = detect_language(user_input)
            else:
                user_language = "en"  # 기본값
        print(f"User language: {user_language}")

        if user_input:
            # RAG 기반 분석 수행
            rag_response = rag_enhanced_document_analysis(file_path, user_input, user_language)

            if rag_response and isinstance(rag_response, str):
                return {
                    "rag_summary": rag_response,
                    "user_query": user_input,
                    "file_path": file_path,
                    "language": user_language
                }
            else:
                print(f"Warning: Invalid RAG response: {rag_response}")
                if user_language == "ko":
                    error_msg = "문서 처리 중 오류가 발생했습니다."
                else:
                    error_msg = "Error processing document."
                return {
                    "rag_summary": error_msg,
                    "user_query": user_input,
                    "file_path": file_path,
                    "language": user_language
                }

            # # 사용자 입력이 있는 경우 의도 기반 분석 수행 (구버전)
            # final_summary = intent_based_document_analysis(file_path, user_input)
            # print(f"\nDocument: {file_path}")
            # print(f"\nUser Input: {user_input}")
            # print(f"\nIntent-based Summary:\n{final_summary}")
            # return final_summary
        else:
            # 언어별 기본 요약 쿼리 생성
            general_query = ""
            if user_language == "ko":
                general_query = "이 문서의 주요 내용, 구조, 핵심 통찰을 강조하여 종합적인 요약을 제공해주세요."
            else:
                general_query = "Provide a comprehensive summary of this document, highlighting its main points, structure, and key insights."
            
            rag_response = rag_enhanced_document_analysis(file_path, general_query, user_language)
            
            if rag_response and isinstance(rag_response, str):
                return {
                    "rag_summary": rag_response,
                    "user_query": general_query,
                    "file_path": file_path,
                    "language": user_language
                }
            else:
                print(f"Warning: Invalid RAG response for general query: {rag_response}")
                return {
                    "rag_summary": "Error summarizing document.",
                    "user_query": general_query,
                    "file_path": file_path,
                    "language": user_language
                }

            # # 기존의 일반적인 문서 분석 수행 (구버전)
            # overall_summary, questions, summaries, importances = analyze_document(file_path)
            # if overall_summary is None:
            #     return None
            
            # final_prompt = create_final_prompt(overall_summary, questions, summaries, importances)
            
            # print(f"Document: {file_path}")
            # print(f"\nOverall Summary:\n{overall_summary}")
            # print("\nPotential User Questions:")
            # for q in questions:
            #     print(f"- {q}")
            
            # return final_prompt
    except Exception as e:
        print(f"Error summarizing document: {e}")
        import traceback
        traceback.print_exc()
        return {
            "rag_summary": f"An error occurred: {str(e)}",
            "user_query": user_input or "general summary",
            "file_path": file_path,
            "language": user_language if 'user_language' in locals() else "en"
        }


""" 
LangChain/LangGraph 적용 
TODO: 250527 이슈1: 불완전한 상태
"""

def _escape_prompt_content(text_content: str) -> str:
    """LangChain 프롬프트에 들어갈 텍스트의 중괄호를 안전하게 이스케이프 처리함."""
    if not isinstance(text_content, str):
        return ""
    return text_content.replace("{", "{{").replace("}", "}}")

# --- 0. LangChain/LangGraph에서 사용할 LLM 실행 함수 ---
def gemma_llm_runner_for_doc_analysis(inputs: dict) -> str:
    global model, processor

    if not model or not processor:
        logging.error("DocAnalysis LLM Runner: Model or processor not set.")
        return "LLM_ERROR: Model/Processor not available."
    
    formatted_messages_lc = inputs.get("formatted_messages")
    if not formatted_messages_lc: # None 이거나 빈 리스트일 경우
        logging.error("DocAnalysis LLM Runner: 'formatted_messages' is missing or empty in input.")
        return "LLM_ERROR: No messages provided."
    llm_params_from_input = inputs.get("llm_params", {})

    # --- 중요: List[BaseMessage]를 List[Dict[str, str]]로 변환 ---
    # processor.apply_chat_template이 기대하는 형식으로 변환.
    # BaseMessage의 'type' 속성을 'role'로 매핑하고, 'content' 속성을 사용.
    conversation_for_gemma = []

    for msg in formatted_messages_lc:
        role = ""
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else: # 기타 BaseMessage 타입 (ToolMessage, FunctionMessage 등)은 현재 로직에서 어떻게 처리할지 정의 필요
            role = "user" # 기본값 또는 에러 처리 (ToolMessage는 'tool' 역할, FunctionMessage는 'function' 역할 등)
            logging.warning(f"DocAnalysis (gemma_llm_runner): Unknown message type {type(msg)}, defaulting role to 'user'.")

        # 여기에서 msg.content의 타입을 확인하고, processor가 기대하는 형태로 변환하는 것이 중요.
        # Gemma-3 모델의 chat_template은 content가 문자열이거나,
        # [{"type": "text", "text": "..."}] 형태의 리스트를 예상.
        processed_content = ""
        if isinstance(msg.content, str):
            processed_content = msg.content
        elif isinstance(msg.content, list): # ([{"type": "text", "text": "..."}])
            # LangChain의 content가 리스트인 경우 (멀티모달 등), 텍스트 부분만 추출하여 합침
            text_parts = []
            for part in msg.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str): # 간혹 리스트 안에 문자열이 바로 있을 수도 있음
                    text_parts.append(part)
            processed_content = " ".join(text_parts).strip()
            
            # 만약 시각적 콘텐츠가 포함된 경우 경고 로깅
            visual_content_present = any(isinstance(p, dict) and p.get("type") in ["image", "video"] for p in msg.content)
            if visual_content_present:
                logging.warning(f"DocAnalysis (gemma_llm_runner): Visual content detected in LangChain message. It will be ignored as this LLM call is text-only. Message content: {msg.content}")
        else:
            logging.error(f"DocAnalysis (gemma_llm_runner): Message content is neither string nor list (type: {type(msg.content)}). Skipping message.")
            continue # 처리할 수 없는 메시지는 건너뛰기

        if processed_content: # 빈 콘텐츠는 추가하지 않음
            content_payload = [
                {"type": "text", "text": processed_content}
            ]
            conversation_for_gemma.append({"role": role, "content": content_payload})

    try:
        # conversation이 비어있으면 apply_chat_template 오류 발생 가능
        if not conversation_for_gemma:
            logging.warning("DocAnalysis (gemma_llm_runner): No valid messages to apply chat template.")
            return "NO_VALID_MESSAGES_ERROR"

        tokenized_inputs = processor.apply_chat_template(
            conversation=conversation_for_gemma,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        input_len = tokenized_inputs["input_ids"].shape[-1]

        max_new_tokens = llm_params_from_input.get("max_new_tokens", 2048)
        do_sample_param = llm_params_from_input.get("do_sample", False)
        temperature = llm_params_from_input.get("temperature", 0.7)
        top_p = llm_params_from_input.get("top_p")
        top_k = llm_params_from_input.get("top_k")

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample_param,
        }
        # do_sample_param 값에 따라 temperature, top_p, top_k를 조건부로 추가
        if do_sample_param:
            generation_kwargs["temperature"] = temperature
            if top_p is not None:
                generation_kwargs["top_p"] = top_p
            else:
                generation_kwargs["top_p"] = 0.9 # 합리적인 기본값 설정
            if top_k is not None:
                generation_kwargs["top_k"] = top_k
            else:
                generation_kwargs["top_k"] = 50 # 합리적인 기본값 설정
        else: # do_sample=False 이면, 샘플링 관련 파라미터는 전달하지 않거나 기본값으로
            pass

        with torch.inference_mode():
            generation_output = model.generate(
                **tokenized_inputs,
                **generation_kwargs
            )
            generated_ids = generation_output[0][input_len:]

        decoded_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
       
        if not decoded_text:
            logging.warning("DocAnalysis (gemma_llm_runner): LLM produced an empty response.")
        return decoded_text
    except Exception as e:
        # exc_info=True 대신 logging.exception 사용 또는 예외 객체 직접 전달
        logging.error(f"DocAnalysis (gemma_llm_runner): Error during LLM call - {type(e).__name__}: {e}")
        import traceback
        logging.debug(traceback.format_exc()) # 디버그 레벨로 트레이스백 전체 출력
        return f"LLM_CALL_ERROR: {type(e).__name__} - {str(e)}"
    
# LangChain Runnable 객체 생성
doc_analysis_gemma_runnable = RunnableLambda(gemma_llm_runner_for_doc_analysis)


"""LangGraph Logic (25.05.26 ~ )"""

# --- 1. LangGraph 상태 정의 ---

class DocumentAnalysisGraphState(TypedDict):
    # 초기 입력
    original_document_content: str  # 원본 문서 전체 내용
    user_query: Optional[str]       # 사용자의 문서 관련 질문 (옵셔널)
    document_language: str          # 감지된 문서 또는 질의의 언어
    max_analysis_tokens: int        # 문서/검색 결과 분석 시 청크당 최대 컨텍스트 토큰 (4096 ~ 16384)
    max_final_answer_tokens: int    # Raika 최종 답변을 위한 최대 토큰 (챗봇 턴당 예산, ~1600)

    # 중간 처리 결과
    document_chunks: List[str]                  # 분할된 문서 청크 리스트
    chunk_embeddings: Optional[np.ndarray]      # 각 청크의 임베딩 (옵셔널)
    relevant_chunks_info: List[Dict[str, Any]]  # 질의와 관련된 청크 및 분석 정보 (예: {'chunk_text': str, 'summary': str, 'relevance_score': float})

    needs_google_search: bool                        # 구글 검색 필요 여부
    google_search_query: Optional[str]               # 구글 검색을 위한 질의어
    google_search_results_summary: Optional[str]     # 구글 검색 결과 요약 정보

    intermediate_analysis_log: List[str]            # 분석 단계별 로그 또는 요약

    # 최종 결과
    final_synthesized_answer: Optional[str]        # 사용자에게 전달될 최종 분석 결과 또는 답변
    raika_formatted_response: Optional[str]        # Raika 캐릭터 페르소나를 적용한 최종 응답

# LangGraph 그래프 인스턴스 (전역)
compiled_document_analysis_graph: Optional[StateGraph] = None


# --- 2. LangGraph 노드 함수들 ---

def node_initialize_analysis(state: DocumentAnalysisGraphState) -> DocumentAnalysisGraphState:
    """
    문서 분석을 초기화하는 노드
    - 언어 감지
    - 문서 청크 분할
    - 임베딩 생성 (필요시)
    """
    logging.info("[DocAnalysis] Initializing document analysis...")

    # 언어 감지
    if state.get("user_query"):
        doc_language = detect_language(state["user_query"])
    else:
        doc_language = detect_language(state["original_document_content"][:1000])   # 문서 앞부분으로 감지

    logging.info(f"[DocAnalysis] Detected language: {doc_language}")

    # 문서 청크 분할
    try:
        # embedding_model이 로드되어 있는지 확인
        if embedding_model is None:
            load_embedding_model()

        # 청크 분할 및 임베딩 생성
        chunks_with_embeddings = optimize_chunk_size(
            state["original_document_content"],
            processor,
            embedding_model
        )

        # 청크 텍스트와 임베딩 분리
        chunks = [chunk_text for chunk_text, _ in chunks_with_embeddings]
        embeddings = np.array([emb for _, emb in chunks_with_embeddings])

        logging.info(f"[DocAnalysis] Created {len(chunks)} chunks with embeddings")

    except Exception as e:
        logging.error(f"[DocAnalysis] Error during chunk optimization: {e}")
        # 폴백: 에러 시 단순 청크 분할
        chunks = [state["original_document_content"][i:i+4000]
                  for i in range(0, len(state["original_document_content"]), 4000)]
        embeddings = None
        logging.warning("[DocAnalysis] Using fallback simple chunking")

    return {
        **state,
        "document_language": doc_language,
        "document_chunks": chunks,
        "chunk_embeddings": embeddings,
        "intermediate_analysis_log": [f"Document initialized: {len(chunks)} chunks created"]
    }

def node_retrieve_document_chunks(state: DocumentAnalysisGraphState) -> DocumentAnalysisGraphState:
    """
    사용자 질의와 관련된 문서 청크를 검색하는 노드
    """
    logging.info("[DocAnalysis] Retrieving relevant document chunks...")

    if not state.get("user_query"):
        # 질의가 없으면 모든 청크를 관련 청크로 간주
        relevant_chunks_info = [
            {"chunk_text": chunk, "relevance_score": 1.0}
            for chunk in state["document_chunks"]
        ]
    else:
        # 관련 청크 검색
        chunks_with_embeddings = list(zip(state["document_chunks"], state["chunk_embeddings"]))
        relevant_results = retrieve_relevant_chunks(
            state["user_query"],
            chunks_with_embeddings,
            top_k=min(5, len(state["document_chunks"]))     # 최대 5개 혹은 전체 청크 수
        )

        relevant_chunks_info = [
            {"chunk_text": chunk, "relevance_score": score}
            for chunk, score in relevant_results
        ]

    log_entry = f"Retrieved {len(relevant_chunks_info)} relevant chunks"
    logging.info(f"[DocAnalysis] {log_entry}")

    return {
        **state,
        "relevant_chunks_info": relevant_chunks_info,
        "intermediate_analysis_log": state.get("intermediate_analysis_log", []) + [log_entry]
    }

def node_summarize_all_chunks(state: DocumentAnalysisGraphState) -> DocumentAnalysisGraphState:
    """
    전체 문서 청크를 요약하는 노드 (user_query가 없을 때)
    """
    logging.info("[DocAnalysis] Summarizing all document chunks...")

    summarized_chunks_info = []
    lang = state["document_language"]

    # 각 청크를 개별적으로 요약
    for i, chunk in enumerate(state["document_chunks"]):
        escaped_chunk = _escape_prompt_content(chunk)
        if lang == "ko":
            summary_prompt = f"다음 문서 부분을 간결하게 요약해주세요:\n\n{escaped_chunk}"
        else:
            summary_prompt = f"Please provide a concise summary of the following document section:\n\n{escaped_chunk}"
        
        # LangChain으로 요약 생성
        chain = (
            RunnablePassthrough.assign(
                formatted_messages=lambda x: ChatPromptTemplate.from_messages([
                    ("human", x["prompt"])
                ]).invoke({"prompt": x["prompt"]}).to_messages(),
                llm_params=lambda x: {
                    "max_new_tokens": 300,
                    "do_sample": True,
                    "temperature": 0.5
                }
            )
            | doc_analysis_gemma_runnable
            | StrOutputParser()
        )

        try:
            summary = chain.invoke({"prompt": summary_prompt})
            summarized_chunks_info.append({
                "chunk_text": chunk,
                "summary": summary,
                "relevance_score": 1.0 # 전체 요약이므로 모두 관련됨
            })
            logging.debug(f"[DocAnalysis] Summarized chunk {i+1}/{len(state['document_chunks'])}")
        except Exception as e:
            logging.error(f"[DocAnalysis] Error summarizing chunk {i}: {e}")
            summarized_chunks_info.append({
                "chunk_text": chunk,
                "summary": f"Error summarizing chunk: {str(e)}",
                "relevance_score": 0.5
            })

    log_entry = f"Summarized {len(summarized_chunks_info)} chunks"
    logging.info(f"[DocAnalysis] {log_entry}")

    return {
        **state,
        "relevant_chunks_info": summarized_chunks_info,
        "intermediate_analysis_log": state.get("intermediate_analysis_log", []) + [log_entry]
    }

def node_process_selected_chunks(state: DocumentAnalysisGraphState) -> DocumentAnalysisGraphState:
    """
    선택된 청크들을 심층 분석하는 노드
    - 사용자 질의에 대한 답변 찾기
    - 추가 정보 필요 여부 판단
    """
    logging.info("[DocAnalysis] Processing selected chunks...")
    
    lang = state["document_language"]
    user_query = state.get("user_query", "")
    relevant_chunks = state.get("relevant_chunks_info", [])
    
    if not relevant_chunks:
        logging.warning("[DocAnalysis] No relevant chunks to process")
        return state
    
    # 관련 청크들의 텍스트 결합 시 중괄호 이스케이프 처리
    # 원인: 청크 텍스트에 f-string 같은 {변수}가 포함되어 있을 경우,
    #      ChatPromptTemplate이 이를 실제 채워야 할 변수로 오인하여 오류 발생.
    # 해결: .replace("{", "{{").replace("}", "}}")를 사용하여 중괄호를 이스케이프.
    escaped_chunks_text = []
    for chunk_info in relevant_chunks:
        text_to_escape = chunk_info.get('chunk_text', chunk_info.get('summary', ''))
        escaped_text = _escape_prompt_content(text_to_escape)
        
        # f-string의 잠재적인 형식 지정자 충돌을 피하기 위해, 포매팅과 결합을 분리
        relevance_header = "[Relevance: {:.2f}]".format(chunk_info['relevance_score'])
        combined_chunk_string = relevance_header + "\n" + escaped_text
        escaped_chunks_text.append(combined_chunk_string)
        
    combined_context = "\n\n".join(escaped_chunks_text)
    
    # 1단계: 초기 답변 생성
    if user_query:
        escaped_query = _escape_prompt_content(user_query)
        if lang == "ko":
            initial_analysis_prompt = f"""
다음 문서 내용을 바탕으로 사용자 질문에 답변해주세요.

사용자 질문: {escaped_query}

문서 내용:
{combined_context}

가능한 한 구체적이고 완전한 답변을 제공해주세요. 문서에서 찾을 수 없는 정보가 있다면 명시해주세요.
"""
        else:
            initial_analysis_prompt = f"""
Based on the following document content, please answer the user's question.

User Question: {escaped_query}

Document Content:
{combined_context}

Please provide as specific and complete an answer as possible. If any information cannot be found in the document, please state so explicitly.
"""
    else:
        # 일반 요약인 경우
        if lang == "ko":
            initial_analysis_prompt = f"""
다음 문서 내용을 종합적으로 분석하고 요약해주세요:

{combined_context}

주요 내용, 핵심 포인트, 중요한 통찰을 포함해주세요.
"""
        else:
            initial_analysis_prompt = f"""
Please provide a comprehensive analysis and summary of the following document content:

{combined_context}

Include main points, key insights, and important findings.
"""
    
    # LLM으로 초기 분석 수행
    chain = (
        RunnablePassthrough.assign(
            formatted_messages=lambda x: ChatPromptTemplate.from_messages([
                ("human", x["prompt"])
            ]).invoke({"prompt": x["prompt"]}).to_messages(), # x["prompt"] = initial_analysis_prompt
            llm_params=lambda x: {
                "max_new_tokens": state.get("max_analysis_tokens", 1000),
                "do_sample": True,
                "temperature": 0.7
            }
        )
        | doc_analysis_gemma_runnable
        | StrOutputParser()
    )
    
    try:
        # 초기 답변 생성
        initial_answer = chain.invoke({"prompt": initial_analysis_prompt})
        
        # 2단계: 답변 품질 평가 및 검색 필요성 판단 (사용자 질의가 있는 경우만)
        needs_search = False
        search_query = None
        confidence_score = 1.0
        
        if user_query:
            # user_query 자체에도 중괄호가 있을 수 있으므로 이스케이프 (만약을 위함. 일반적으로 사용자 직접 입력에는 f-string이 드묾)
            escaped_user_query = _escape_prompt_content(user_query)
            # initial_answer에도 LLM이 생성한 코드 예시 등에 중괄호가 있을 수 있으므로 이스케이프
            escaped_initial_answer = _escape_prompt_content(initial_answer)

            if lang == "ko":
                evaluation_prompt = f"""
다음은 사용자 질문과 그에 대한 답변입니다.

사용자 질문: {escaped_user_query}

제공된 답변:
{escaped_initial_answer}

이 답변을 평가해주세요:

1. 답변 완성도 점수 (0-100): 답변이 질문을 얼마나 완전하게 다루고 있는가?
2. 정보 충분성 점수 (0-100): 제공된 정보가 사용자에게 충분히 유용한가?
3. 추가 정보 필요성: 구글 검색을 통해 보완해야 할 정보가 있는가?

다음 형식으로만 답변해주세요:
완성도: [점수]
충분성: [점수]
검색필요: [예/아니오]
검색쿼리: [필요한 경우 구체적인 검색 쿼리]
이유: [간단한 설명]
"""
            else:
                evaluation_prompt = f"""
Here is a user question and its answer.

User Question: {escaped_user_query}

Provided Answer:
{escaped_initial_answer}

Please evaluate this answer:

1. Answer Completeness Score (0-100): How completely does the answer address the question?
2. Information Sufficiency Score (0-100): Is the provided information sufficiently useful for the user?
3. Need for Additional Information: Is there information that should be supplemented through Google search?

Please respond only in this format:
Completeness: [score]
Sufficiency: [score]
SearchNeeded: [Yes/No]
SearchQuery: [specific search query if needed]
Reason: [brief explanation]
"""
            
            # 평가 수행
            evaluation_result = chain.invoke({"prompt": evaluation_prompt})
            
            # 평가 결과 파싱
            if lang == "ko":
                import re
                completeness_match = re.search(r'완성도:\s*(\d+)', evaluation_result)
                sufficiency_match = re.search(r'충분성:\s*(\d+)', evaluation_result)
                search_needed_match = re.search(r'검색필요:\s*(예|아니오)', evaluation_result)
                search_query_match = re.search(r'검색쿼리:\s*(.+?)(?:\n|$)', evaluation_result)
                
                if completeness_match and sufficiency_match:
                    completeness = int(completeness_match.group(1))
                    sufficiency = int(sufficiency_match.group(1))
                    confidence_score = (completeness + sufficiency) / 200.0
                    
                    # 점수가 낮거나 명시적으로 검색이 필요하다고 판단한 경우
                    if confidence_score < 0.7 or (search_needed_match and search_needed_match.group(1) == "예"):
                        needs_search = True
                        if search_query_match:
                            search_query = search_query_match.group(1).strip()
                        else:
                            # 검색 쿼리가 없으면 사용자 질문을 기반으로 생성
                            search_query = user_query
            else:
                import re
                completeness_match = re.search(r'Completeness:\s*(\d+)', evaluation_result)
                sufficiency_match = re.search(r'Sufficiency:\s*(\d+)', evaluation_result)
                search_needed_match = re.search(r'SearchNeeded:\s*(Yes|No)', evaluation_result, re.IGNORECASE)
                search_query_match = re.search(r'SearchQuery:\s*(.+?)(?:\n|$)', evaluation_result)
                
                if completeness_match and sufficiency_match:
                    completeness = int(completeness_match.group(1))
                    sufficiency = int(sufficiency_match.group(1))
                    confidence_score = (completeness + sufficiency) / 200.0
                    
                    if confidence_score < 0.7 or (search_needed_match and search_needed_match.group(1).lower() == "yes"):
                        needs_search = True
                        if search_query_match:
                            search_query = search_query_match.group(1).strip()
                        else:
                            search_query = user_query
        
        # 분석 결과를 relevant_chunks_info에 추가
        updated_chunks_info = relevant_chunks.copy()
        updated_chunks_info.append({
            "chunk_text": "ANALYSIS_RESULT",
            "summary": initial_answer,
            "relevance_score": 1.0,
            "confidence_score": confidence_score
        })
        
        log_entry = f"Processed {len(relevant_chunks)} chunks. Confidence: {confidence_score:.2f}, Search needed: {needs_search}"
        logging.info(f"[DocAnalysis] {log_entry}")
        
        return {
            **state,
            "relevant_chunks_info": updated_chunks_info,
            "needs_google_search": needs_search,
            "google_search_query": search_query,
            # "answer_confidence_score": confidence_score,
            "intermediate_analysis_log": state.get("intermediate_analysis_log", []) + [log_entry]
        }
        
    except Exception as e:
        logging.error(f"[DocAnalysis] Error in chunk analysis: {e}")
        return {
            **state,
            "relevant_chunks_info": relevant_chunks,
            "needs_google_search": False,
            "google_search_query": None,
            # "answer_confidence_score": 0.0,
            "intermediate_analysis_log": state.get("intermediate_analysis_log", []) + [f"Analysis error: {str(e)}"]
        }
    

""" TODO: 250527 - 개선 필요? """


def _extract_search_query(analysis_result: str, user_query: str, language: str) -> str:
    """
    분석 결과에서 검색 쿼리를 추출하거나 생성하는 헬퍼 함수
    LLM을 활용하여 더 정확한 검색 쿼리 생성
    """
    # LLM을 활용한 검색 쿼리 생성
    if language == "ko":
        query_extraction_prompt = f"""
다음 분석 결과와 사용자 질문을 바탕으로 구글 검색에 사용할 최적의 검색 쿼리를 생성해주세요.

사용자 질문: {user_query}

분석 결과에서 언급된 부족한 정보:
{analysis_result}

다음 형식으로만 답변해주세요:
검색쿼리: [구체적이고 효과적인 검색 쿼리]
"""
    else:
        query_extraction_prompt = f"""
Based on the analysis result and user question, generate an optimal search query for Google.

User Question: {user_query}

Missing information mentioned in analysis:
{analysis_result}

Please respond only in this format:
SearchQuery: [specific and effective search query]
"""
    
    # LLM 체인 구성
    chain = (
        RunnablePassthrough.assign(
            formatted_messages=lambda x: ChatPromptTemplate.from_messages([
                ("human", x["prompt"])
            ]).invoke({"prompt": x["prompt"]}).to_messages(),
            llm_params=lambda x: {
                "max_new_tokens": 100,
                "do_sample": True,
                "temperature": 0.5
            }
        )
        | doc_analysis_gemma_runnable
        | StrOutputParser()
    )
    
    try:
        result = chain.invoke({"prompt": query_extraction_prompt})
        
        # 결과에서 검색 쿼리 추출
        if language == "ko":
            match = re.search(r'검색쿼리:\s*(.+?)(?:\n|$)', result)
        else:
            match = re.search(r'SearchQuery:\s*(.+?)(?:\n|$)', result, re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        else:
            # 추출 실패 시 원본 쿼리 사용
            return user_query
            
    except Exception as e:
        logging.error(f"Error extracting search query: {e}")
        return user_query


def node_execute_google_search(state: DocumentAnalysisGraphState) -> DocumentAnalysisGraphState:
    """
    구글 검색을 실행하는 노드
    검색 필요성을 재평가하고 최적의 검색 전략 선택
    """
    logging.info(f"[DocAnalysis] Executing Google search for: {state.get('google_search_query')}")
    
    if not state.get("google_search_query"):
        logging.warning("[DocAnalysis] No search query provided")
        return state
    
    # 검색 전략 결정을 위한 LLM 활용
    lang = state["document_language"]
    search_query = state["google_search_query"]
    
    # 검색 전략 결정
    if lang == "ko":
        strategy_prompt = f"""
다음 검색 쿼리에 대해 가장 적절한 검색 전략을 선택해주세요.

검색 쿼리: {search_query}

다음 중 하나를 선택하세요:
1. 단순검색: 간단한 사실 확인이나 정의 찾기
2. 심층검색: 복잡한 문제 해결이나 다각도 분석 필요
3. 최신정보: 최근 뉴스나 업데이트된 정보 필요

다음 형식으로만 답변해주세요:
전략: [단순검색/심층검색/최신정보]
이유: [간단한 설명]
"""
    else:
        strategy_prompt = f"""
Select the most appropriate search strategy for this query.

Search Query: {search_query}

Choose one:
1. SimpleSearch: Simple fact checking or definition finding
2. DeepSearch: Complex problem solving or multi-angle analysis needed
3. LatestInfo: Recent news or updated information needed

Please respond only in this format:
Strategy: [SimpleSearch/DeepSearch/LatestInfo]
Reason: [brief explanation]
"""
    
    # LLM으로 전략 결정
    chain = (
        RunnablePassthrough.assign(
            formatted_messages=lambda x: ChatPromptTemplate.from_messages([
                ("human", x["prompt"])
            ]).invoke({"prompt": x["prompt"]}).to_messages(),
            llm_params=lambda x: {
                "max_new_tokens": 100,
                "do_sample": False,
                "temperature": 0.3
            }
        )
        | doc_analysis_gemma_runnable
        | StrOutputParser()
    )
    
    try:
        strategy_result = chain.invoke({"prompt": strategy_prompt})
        
        # 전략 파싱
        search_strategy = "simple"  # 기본값
        if lang == "ko":
            if "심층검색" in strategy_result:
                search_strategy = "deep"
            elif "최신정보" in strategy_result:
                search_strategy = "latest"
        else:
            if "DeepSearch" in strategy_result:
                search_strategy = "deep"
            elif "LatestInfo" in strategy_result:
                search_strategy = "latest"
        
        logging.info(f"[DocAnalysis] Selected search strategy: {search_strategy}")
        
        # GoogleSearch_Gemma의 검색 기능 호출
        if search_strategy == "deep":
            # 복잡한 검색인 경우 - 검색 유형 분류 후 LangGraph 기반 검색
            search_type = GoogleSearch_Gemma.classify_search_type_langchain(
                search_query, 
                lang
            )
            
            if "complex_" in search_type:
                search_result = GoogleSearch_Gemma.search_and_reason_for_complex_problem_langgraph(
                    query=search_query,
                    problem_type=search_type,
                    language=lang,
                    max_iterations=3  # 심층 검색이므로 더 많은 반복
                )
                
                if search_result and search_result.get("status") == "success":
                    search_summary = f"검색 결과:\n{search_result.get('best_snippet', '')}\n\n추론된 해결책:\n{search_result.get('best_plan', '')}"
                else:
                    search_summary = search_result.get("reasoning_summary", "검색 결과를 찾을 수 없습니다.")
            else:
                # 단순하지만 깊이 있는 검색이 필요한 경우
                search_content, is_satisfactory, iterations = GoogleSearch_Gemma.recursive_search(
                    search_query,
                    language=lang,
                    max_iterations=3,
                    user_query=search_query
                )
                search_summary = f"검색 결과:\n{search_content}" if search_content else "관련 검색 결과를 찾을 수 없습니다."
        
        elif search_strategy == "latest":
            # 최신 정보 검색 - 날짜 제한 추가
            from datetime import datetime, timedelta
            current_date = datetime.now()
            date_limit = (current_date - timedelta(days=30)).strftime("%Y-%m-%d")
            
            modified_query = f"{search_query} after:{date_limit}"
            search_content, is_satisfactory, iterations = GoogleSearch_Gemma.recursive_search(
                modified_query,
                language=lang,
                max_iterations=2,
                user_query=search_query
            )
            search_summary = f"최신 정보 (최근 30일):\n{search_content}" if search_content else "최신 정보를 찾을 수 없습니다."
        
        else:  # simple search
            # 단순 검색
            search_content, is_satisfactory, iterations = GoogleSearch_Gemma.recursive_search(
                search_query,
                language=lang,
                max_iterations=1,  # 단순 검색은 1회만
                user_query=search_query
            )
            search_summary = f"검색 결과:\n{search_content}" if search_content else "관련 검색 결과를 찾을 수 없습니다."
        
        # 검색 결과 품질 평가
        if lang == "ko":
            quality_prompt = f"""
다음 검색 결과가 사용자 질문에 얼마나 도움이 되는지 평가해주세요.

원래 검색 쿼리: {search_query}
검색 결과: {search_summary[:500]}...

다음 형식으로만 답변해주세요:
품질점수: [0-100]
추가검색필요: [예/아니오]
"""
        else:
            quality_prompt = f"""
Evaluate how helpful this search result is for the user's question.

Original Query: {search_query}
Search Result: {search_summary[:500]}...

Please respond only in this format:
QualityScore: [0-100]
NeedMoreSearch: [Yes/No]
"""
        
        quality_result = chain.invoke({"prompt": quality_prompt})
        
        # 품질 점수 추출
        quality_score = 70  # 기본값
        needs_more_search = False
        
        if lang == "ko":
            score_match = re.search(r'품질점수:\s*(\d+)', quality_result)
            more_search_match = re.search(r'추가검색필요:\s*(예|아니오)', quality_result)
            if score_match:
                quality_score = int(score_match.group(1))
            if more_search_match and more_search_match.group(1) == "예":
                needs_more_search = True
        else:
            score_match = re.search(r'QualityScore:\s*(\d+)', quality_result)
            more_search_match = re.search(r'NeedMoreSearch:\s*(Yes|No)', quality_result, re.IGNORECASE)
            if score_match:
                quality_score = int(score_match.group(1))
            if more_search_match and more_search_match.group(1).lower() == "yes":
                needs_more_search = True
        
        log_entry = f"Google search completed (Strategy: {search_strategy}, Quality: {quality_score}/100)"
        logging.info(f"[DocAnalysis] {log_entry}")
        
        # 품질이 낮고 추가 검색이 필요한 경우 검색 쿼리 개선
        if quality_score < 50 and needs_more_search:
            # 검색 쿼리 개선
            improved_query = _extract_search_query(
                f"현재 검색 결과가 불충분함. 원래 쿼리: {search_query}",
                search_query,
                lang
            )
            
            # 개선된 쿼리로 재검색 (1회만)
            if improved_query != search_query:
                logging.info(f"[DocAnalysis] Retrying with improved query: {improved_query}")
                search_content, _, _ = GoogleSearch_Gemma.recursive_search(
                    improved_query,
                    language=lang,
                    max_iterations=1,
                    user_query=search_query
                )
                if search_content:
                    search_summary += f"\n\n추가 검색 결과:\n{search_content}"
        
        return {
            **state,
            "google_search_results_summary": search_summary,
            "search_quality_score": quality_score,
            "intermediate_analysis_log": state.get("intermediate_analysis_log", []) + [log_entry]
        }
        
    except Exception as e:
        logging.error(f"[DocAnalysis] Error during Google search: {e}")
        error_msg = f"검색 중 오류 발생: {str(e)}" if lang == "ko" else f"Search error: {str(e)}"
        
        return {
            **state,
            "google_search_results_summary": error_msg,
            "search_quality_score": 0,
            "intermediate_analysis_log": state.get("intermediate_analysis_log", []) + [f"Search error: {str(e)}"]
        }


def node_synthesize_information(state: DocumentAnalysisGraphState) -> DocumentAnalysisGraphState:
    """
    모든 정보를 종합하여 최종 답변을 생성하는 노드
    정보의 품질과 완성도를 평가하여 최적의 답변 생성
    """
    logging.info("[DocAnalysis] Synthesizing all information...")
    
    lang = state["document_language"]
    user_query = state.get("user_query", "")
    
    # 문서 분석 결과 수집 및 품질 평가
    doc_analysis_parts = []
    confidence_scores = []
    
    for chunk_info in state.get("relevant_chunks_info", []):
        # chunk_text 또는 summary를 가져와서 이스케이프 처리
        # node_process_selected_chunks에서 ANALYSIS_RESULT의 summary는 LLM의 직접적인 응답이므로,
        # 프롬프트에 넣기 전에 여기서 이스케이프
        text_content = ""
        if chunk_info.get("chunk_text") == "ANALYSIS_RESULT":
            text_content = chunk_info.get("summary", "")
        else:
            # 일반 청크의 요약 또는 원본 텍스트
            text_content = chunk_info.get("summary", chunk_info.get("chunk_text", ""))
        
        escaped_content = _escape_prompt_content(text_content)
        doc_analysis_parts.append(escaped_content)

        # 신뢰도 점수 (confidence_score가 있으면 사용, 없으면 relevance_score 사용, 둘 다 없으면 기본값)
        confidence = chunk_info.get("confidence_score", chunk_info.get("relevance_score", 0.5))
        confidence_scores.append(confidence)
    
    combined_doc_analysis = "\n\n".join(doc_analysis_parts)
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
    
    # 검색 결과 품질 고려
    search_summary_escaped = ""
    if state.get("google_search_results_summary"):
        search_summary_escaped = _escape_prompt_content(state.get("google_search_results_summary"))
    
    search_quality = state.get("search_quality_score", 100) / 100.0 if state.get("google_search_results_summary") else 1.0
    
    # 전체 정보 품질 평가
    overall_quality = (avg_confidence + search_quality) / 2 if state.get("google_search_results_summary") else avg_confidence
    
    # 품질에 따른 종합 전략 결정
    if overall_quality >= 0.8:
        synthesis_strategy = "comprehensive"  # 포괄적 답변
    elif overall_quality >= 0.6:
        synthesis_strategy = "balanced"       # 균형잡힌 답변
    else:
        synthesis_strategy = "cautious"       # 신중한 답변 (불확실성 명시)
    
    logging.info(f"[DocAnalysis] Synthesis strategy: {synthesis_strategy} (quality: {overall_quality:.2f})")
    
    # 사용자 질의 이스케이프 (프롬프트에 직접 사용되므로)
    escaped_user_query = _escape_prompt_content(user_query) if user_query else ""

    # 종합 프롬프트 생성
    if user_query:
        # 구글 검색 결과 섹션 준비
        google_search_section = ""
        if search_summary_escaped:
            if lang == "ko":
                google_search_section = f"구글 검색 결과 (품질: {search_quality:.1%}):\n{search_summary_escaped}"
            else:
                google_search_section = f"Google Search Results (Quality: {search_quality:.1%}):\n{search_summary_escaped}"
        
        # 답변 전략 텍스트 준비
        if lang == "ko":
            if synthesis_strategy == 'comprehensive':
                strategy_text = '모든 정보를 포괄적으로 활용하여 상세한 답변 제공'
            elif synthesis_strategy == 'balanced':
                strategy_text = '주요 정보를 균형있게 조합하여 답변 제공'
            else:
                strategy_text = '확실한 정보 위주로 신중하게 답변하고, 불확실한 부분은 명시'
            
            synthesis_prompt = f"""
다음 정보들을 종합하여 사용자 질문에 대한 최종 답변을 작성해주세요.

사용자 질문: {escaped_user_query}

문서 분석 결과 (신뢰도: {avg_confidence:.1%}):
{combined_doc_analysis}

{google_search_section}

답변 전략: {strategy_text}

위 정보를 바탕으로 {state.get('max_final_answer_tokens', 1600)} 토큰 이내로 답변을 작성해주세요.
핵심 내용을 우선적으로 포함하고, 정보의 신뢰도를 고려하여 답변해주세요.
"""
        else:
            if synthesis_strategy == 'comprehensive':
                strategy_text = 'Provide a comprehensive answer using all available information'
            elif synthesis_strategy == 'balanced':
                strategy_text = 'Provide a balanced answer combining key information'
            else:
                strategy_text = 'Provide a cautious answer focusing on certain information and noting uncertainties'
            
            synthesis_prompt = f"""
Please synthesize the following information to create a final answer to the user's question.

User Question: {escaped_user_query}

Document Analysis Results (Confidence: {avg_confidence:.1%}):
{combined_doc_analysis}

{google_search_section}

Answer Strategy: {strategy_text}

Based on this information, please write an answer within {state.get('max_final_answer_tokens', 1600)} tokens.
Prioritize key information and consider the reliability of the information in your answer.
"""
    else:
        # 일반 요약인 경우
        if lang == "ko":
            synthesis_prompt = f"""
다음 문서 분석 결과를 종합하여 포괄적인 요약을 작성해주세요.

문서 분석 결과 (평균 관련성/신뢰도: {avg_confidence:.1%}):
{combined_doc_analysis}

전체 정보 품질: {overall_quality:.1%}

주요 내용, 핵심 통찰, 중요한 발견사항을 {state.get('max_final_answer_tokens', 1600)} 토큰 이내로 요약해주세요.
정보의 품질과 완성도를 고려하여 균형잡힌 요약을 제공해주세요.
"""
        else:
            synthesis_prompt = f"""
Please synthesize the following document analysis results into a comprehensive summary.

Document Analysis Results (Average Relevance/Confidence: {avg_confidence:.1%}):
{combined_doc_analysis}

Overall Information Quality: {overall_quality:.1%}

Summarize the main points, key insights, and important findings within {state.get('max_final_answer_tokens', 1600)} tokens.
Provide a balanced summary considering the quality and completeness of the information.
"""
    
    # LLM으로 최종 답변 생성
    chain = (
        RunnablePassthrough.assign(
            formatted_messages=lambda x: ChatPromptTemplate.from_messages([
                ("human", x["prompt"])
            ]).invoke({"prompt": x["prompt"]}).to_messages(),
            llm_params=lambda x: {
                "max_new_tokens": state.get("max_final_answer_tokens", 1600),
                "do_sample": True,
                "temperature": 0.7 if synthesis_strategy == "comprehensive" else 0.5
            }
        )
        | doc_analysis_gemma_runnable
        | StrOutputParser()
    )
    
    try:
        final_answer = chain.invoke({"prompt": synthesis_prompt})
        
        # 답변 품질 자체 평가
        # final_answer에도 중괄호가 포함될 수 있으므로 이스케이프
        escaped_final_answer_for_self_eval = _escape_prompt_content(final_answer)
        
        # user_query가 없을 경우 "문서 요약" 또는 "Document Summary"를 사용
        eval_query_text = escaped_user_query if user_query else ("문서 요약" if lang == "ko" else "Document Summary")

        # 답변 품질 자체 평가
        if lang == "ko":
            self_eval_prompt = f"""
다음 답변의 품질을 평가해주세요.

원래 질문: {eval_query_text}
생성된 답변: {escaped_final_answer_for_self_eval[:500]}...

다음 형식으로만 답변해주세요:
완성도: [0-100]
개선필요: [예/아니오]
"""
        else:
            self_eval_prompt = f"""
Evaluate the quality of this answer.

Original Question: {eval_query_text}
Generated Answer: {escaped_final_answer_for_self_eval[:500]}...

Please respond only in this format:
Completeness: [0-100]
NeedsImprovement: [Yes/No]
"""

        eval_chain = (
            RunnablePassthrough.assign(
                formatted_messages=lambda x: ChatPromptTemplate.from_messages([
                    ("human", x["prompt"])
                ]).invoke({"prompt": x["prompt"]}).to_messages(),
                llm_params=lambda x: { # 자체 평가는 좀 더 결정적인 답변을 위해 do_sample=False 고려
                    "max_new_tokens": 100, 
                    "do_sample": False, # 또는 True, temperature 낮게
                    "temperature": 0.3
                }
            )
            | doc_analysis_gemma_runnable
            | StrOutputParser()
        )
        eval_result = eval_chain.invoke({"prompt": self_eval_prompt})
        
        # 평가 결과 파싱
        needs_improvement = False
        if lang == "ko":
            improve_match = re.search(r'개선필요:\s*(예|아니오)', eval_result)
            if improve_match and improve_match.group(1) == "예":
                needs_improvement = True
        else:
            improve_match = re.search(r'NeedsImprovement:\s*(Yes|No)', eval_result, re.IGNORECASE)
            if improve_match and improve_match.group(1).lower() == "yes":
                needs_improvement = True
        
        # 개선이 필요한 경우 한 번 더 시도
        if needs_improvement and overall_quality < 0.7:
            logging.info("[DocAnalysis] Attempting to improve answer...")
            
            # final_answer를 improvement_prompt에 사용하기 전에 이스케이프 처리
            escaped_final_answer_for_improvement = _escape_prompt_content(final_answer)

            if lang == "ko":
                improvement_prompt = f"""
이전 답변을 개선해주세요. 더 명확하고 유용한 정보를 제공하도록 수정해주세요.

이전 답변:
{escaped_final_answer_for_improvement}

개선된 답변을 {state.get('max_final_answer_tokens', 1600)} 토큰 이내로 작성해주세요.
"""
            else:
                improvement_prompt = f"""
Please improve the previous answer. Revise it to provide clearer and more useful information.

Previous Answer:
{escaped_final_answer_for_improvement}

Write an improved answer within {state.get('max_final_answer_tokens', 1600)} tokens.
"""
            
            improved_answer = chain.invoke({"prompt": improvement_prompt})
            if improved_answer:
                final_answer = improved_answer
        
        log_entry = f"Information synthesis completed (Strategy: {synthesis_strategy}, Quality: {overall_quality:.2f})"
        logging.info(f"[DocAnalysis] {log_entry}")
        
        return {
            **state,
            "final_synthesized_answer": final_answer,
            "synthesis_quality": overall_quality,
            "intermediate_analysis_log": state.get("intermediate_analysis_log", []) + [log_entry]
        }
        
    except Exception as e:
        logging.error(f"[DocAnalysis] Error in synthesis: {e}")
        error_msg = f"정보 종합 중 오류 발생: {str(e)}" if lang == "ko" else f"Synthesis error: {str(e)}"
        
        return {
            **state,
            "final_synthesized_answer": error_msg,
            "synthesis_quality": 0.0,
            "intermediate_analysis_log": state.get("intermediate_analysis_log", []) + [f"Synthesis error: {str(e)}"]
        }

def node_format_for_raika(state: DocumentAnalysisGraphState) -> DocumentAnalysisGraphState:
    """
    Raika의 페르소나를 적용하는 노드
    """
    logging.info("[DocAnalysis] Applying Raika's persona...")
    
    if not state.get("final_synthesized_answer"):
        logging.warning("[DocAnalysis] No answer to format")
        return state
    
    try:
        # format_response_for_character 함수 사용
        raika_response = format_response_for_character(
            state["final_synthesized_answer"],
            state["document_language"]
        )
        
        if raika_response:
            log_entry = "Raika formatting completed"
            logging.info(f"[DocAnalysis] {log_entry}")
            
            return {
                **state,
                "raika_formatted_response": raika_response,
                "intermediate_analysis_log": state.get("intermediate_analysis_log", []) + [log_entry]
            }
        else:
            # 포맷팅 실패 시 원본 사용
            logging.warning("[DocAnalysis] Raika formatting failed, using original")
            return {
                **state,
                "raika_formatted_response": state["final_synthesized_answer"],
                "intermediate_analysis_log": state.get("intermediate_analysis_log", []) + ["Raika formatting failed"]
            }
            
    except Exception as e:
        logging.error(f"[DocAnalysis] Error in Raika formatting: {e}")
        return {
            **state,
            "raika_formatted_response": state["final_synthesized_answer"],
            "intermediate_analysis_log": state.get("intermediate_analysis_log", []) + [f"Formatting error: {str(e)}"]
        }

# --- 조건부 라우팅 함수 ---

def route_after_initialization(state: DocumentAnalysisGraphState) -> str:
    """초기화 후 라우팅: 사용자 질의 유무에 따라 경로 결정"""
    if state.get("user_query"):
        logging.info("[DocAnalysis][Router] User query detected -> retrieve chunks")
        return "retrieve_chunks"
    else:
        logging.info("[DocAnalysis][Router] No user query -> summarize all")
        return "summarize_all"

def route_after_processing(state: DocumentAnalysisGraphState) -> str:
    """청크 처리 후 라우팅: 구글 검색 필요 여부에 따라 경로 결정"""
    if state.get("needs_google_search") and state.get("google_search_query"):
        logging.info("[DocAnalysis][Router] Google search needed -> execute search")
        return "google_search"
    else:
        logging.info("[DocAnalysis][Router] No search needed -> synthesize")
        return "synthesize"

# --- LangGraph 빌더 함수 ---

def build_document_analysis_graph():
    """문서 분석을 위한 LangGraph를 빌드하고 컴파일"""
    if not model or not processor:
        raise ValueError("Model and Processor must be set before building the graph.")
    
    logging.info("[DocAnalysis] Building document analysis graph...")
    
    graph_builder = StateGraph(DocumentAnalysisGraphState)
    
    # 노드 추가
    graph_builder.add_node("initialize", node_initialize_analysis)
    graph_builder.add_node("retrieve_chunks", node_retrieve_document_chunks)
    graph_builder.add_node("summarize_all", node_summarize_all_chunks)
    graph_builder.add_node("process_chunks", node_process_selected_chunks)
    graph_builder.add_node("google_search", node_execute_google_search)
    graph_builder.add_node("synthesize", node_synthesize_information)
    graph_builder.add_node("format_raika", node_format_for_raika)
    
    # 진입점 설정
    graph_builder.set_entry_point("initialize")
    
    # 조건부 엣지 추가
    # 1. 초기화 후 라우팅
    graph_builder.add_conditional_edges(
        "initialize",
        route_after_initialization,
        {
            "retrieve_chunks": "retrieve_chunks",
            "summarize_all": "summarize_all"
        }
    )
    
    # 2. 청크 검색/요약 후 처리
    graph_builder.add_edge("retrieve_chunks", "process_chunks")
    graph_builder.add_edge("summarize_all", "process_chunks")
    
    # 3. 처리 후 라우팅
    graph_builder.add_conditional_edges(
        "process_chunks",
        route_after_processing,
        {
            "google_search": "google_search",
            "synthesize": "synthesize"
        }
    )
    
    # 4. 구글 검색 후 종합
    graph_builder.add_edge("google_search", "synthesize")
    
    # 5. 종합 후 포맷팅
    graph_builder.add_edge("synthesize", "format_raika")
    
    # 6. 최종 노드에서 종료
    graph_builder.add_edge("format_raika", END)
    
    # 그래프 컴파일
    compiled_graph = graph_builder.compile()
    logging.info("[DocAnalysis] Document analysis graph compiled successfully")
    
    return compiled_graph

# --- 전역 그래프 초기화 함수 ---

def initialize_document_analysis_graph():
    """문서 분석 그래프를 초기화하고 반환"""
    global compiled_document_analysis_graph
    
    if compiled_document_analysis_graph is None:
        if model and processor:
            try:
                compiled_document_analysis_graph = build_document_analysis_graph()
                logging.info("[DocAnalysis] Document analysis graph initialized")
            except Exception as e:
                logging.error(f"[DocAnalysis] Failed to build graph: {e}")
        else:
            logging.error("[DocAnalysis] Cannot build graph: Model or processor not initialized")
    
    return compiled_document_analysis_graph

# --- 메인 문서 분석 함수 (LangGraph 사용) ---

def analyze_document_with_langgraph(
    document_content: str,
    user_query: Optional[str] = None,
    max_analysis_tokens: int = 8000,
    max_final_answer_tokens: int = 1600
) -> Dict[str, Any]:
    """
    LangGraph를 사용한 문서 분석 메인 함수
    
    Args:
        document_content: 분석할 문서 내용
        user_query: 사용자 질의 (선택적)
        max_analysis_tokens: 분석 시 사용할 최대 토큰 수
        max_final_answer_tokens: 최종 답변의 최대 토큰 수
    
    Returns:
        분석 결과 딕셔너리
    """
    global model, processor
    
    if not model or not processor:
        logging.error("[DocAnalysis] Model or processor not set")
        return {
            "status": "error",
            "message": "Model/Processor not initialized",
            "raika_response": None
        }
    
    language = detect_language(user_query) if user_query else detect_language(
        document_content[:1000] if document_content else ""
    )

    # 251105 - 복잡한 스크립트 분석&해석 관련 로직
    line_count_estimate = document_content.count("\n") + 1 if document_content else 0
    preliminary_docs = [{
        "filename": "document",
        "content": document_content[:60000] if document_content else "",
        "formatted": "",
        "file_extension": "",
        "line_count": line_count_estimate,
        "is_large": line_count_estimate >= 1000
    }]
    direct_response = _maybe_handle_large_script(
        user_query or "Document analysis request",
        preliminary_docs,
        language
    )
    if direct_response:
        effective_language = language or detect_language(document_content[:1000] if document_content else "")
        return {
            "status": "success",
            "raika_response": direct_response,
            "analysis_log": ["Large script routed to OSS20B pipeline"],
            "language": effective_language,
            "search_performed": False
        }

    # 그래프 초기화
    graph = initialize_document_analysis_graph()
    if not graph:
        logging.error("[DocAnalysis] Failed to initialize graph")
        return {
            "status": "error",
            "message": "Graph initialization failed",
            "raika_response": None
        }
    
    # 초기 상태 생성
    initial_state: DocumentAnalysisGraphState = {
        "original_document_content": document_content,
        "user_query": user_query,
        "document_language": "auto",  # 자동 감지
        "max_analysis_tokens": max_analysis_tokens,
        "max_final_answer_tokens": max_final_answer_tokens,
        "document_chunks": [],
        "chunk_embeddings": None,
        "relevant_chunks_info": [],
        "needs_google_search": False,
        "google_search_query": None,
        "google_search_results_summary": None,
        "intermediate_analysis_log": [],
        "final_synthesized_answer": None,
        "raika_formatted_response": None
    }
    
    try:
        # 그래프 실행
        logging.info("[DocAnalysis] Starting graph execution...")
        final_state = graph.invoke(
            initial_state,
            {"recursion_limit": 10}  # 재귀 깊이 제한
        )
        
        # 결과 추출
        if final_state and final_state.get("raika_formatted_response"):
            logging.info("[DocAnalysis] Analysis completed successfully")
            return {
                "status": "success",
                "raika_response": final_state["raika_formatted_response"],
                "analysis_log": final_state.get("intermediate_analysis_log", []),
                "language": final_state.get("document_language", "unknown"),
                "search_performed": final_state.get("needs_google_search", False)
            }
        else:
            logging.error("[DocAnalysis] No response generated")
            return {
                "status": "error",
                "message": "No response generated",
                "raika_response": final_state.get("final_synthesized_answer") or "분석 결과를 생성할 수 없습니다.",
                "analysis_log": final_state.get("intermediate_analysis_log", [])
            }
            
    except Exception as e:
        logging.error(f"[DocAnalysis] Graph execution error: {e}")
        return {
            "status": "error",
            "message": f"Analysis error: {str(e)}",
            "raika_response": None
        }

# --- 기존 함수들을 LangGraph 버전으로 대체하는 래퍼 함수 ---

def summarize_document_langgraph(file_path: str, user_input: str = None, language=None) -> str:
    """
    LangGraph를 사용한 문서 요약 (기존 summarize_document 함수 대체)
    
    Args:
        file_path: 분석할 문서의 파일 경로
        user_input: 사용자 입력/질문
        language: 언어 코드 (자동 감지 시 None)
    
    Returns:
        Raika 형식의 응답 문자열
    """
    try:
        logging.info(f"[DocAnalysis] Summarizing document with LangGraph: {file_path}")
        user_language = language
        if user_language is None:
            if user_input:
                user_language = detect_language(user_input)
            else:
                user_language = "en"
        
        # 파일 읽기
        content = read_file(file_path)
        if content is None:
            # 일관된 딕셔너리 형태로 오류 반환
            return {
                "status": "error",
                "message": "Unable to read the file.",
                "raika_response": "파일을 읽을 수 없습니다." if language == "ko" else "Unable to read the file."
            }
        raw_content_for_large_check = content
        # 파일 형식에 따른 전처리
        _, file_extension = os.path.splitext(file_path)
        if file_extension.lower() == '.py':
            try:
                code_summary = summarize_python_code(content)
                content = f"Code Structure Summary:\n{code_summary}\n\nOriginal Content:\n{content}"
            except Exception as e:
                logging.warning(f"[DocAnalysis] Python code summary failed: {e}")
        else:
            raw_content_for_large_check = content

        documents_info = [{
            "filename": os.path.basename(file_path),
            "content": (raw_content_for_large_check or "")[:60000],
            "formatted": content[:60000] if content else "",
            "file_extension": file_extension,
            "line_count": (raw_content_for_large_check or "").count("\n") + 1 if raw_content_for_large_check else 0,
            "is_large": ((raw_content_for_large_check or "").count("\n") + 1 if raw_content_for_large_check else 0) >= 1000
        }]

        if user_input:
            analysis_query = user_input
        else:
            analysis_query = "이 문서의 주요 내용, 구조, 핵심 통찰을 강조하여 종합적인 요약을 제공해주세요." if user_language == "ko" else "Provide a comprehensive summary of this document, highlighting its main points, structure, and key insights."

        # 251105 - 복잡한 스크립트 분석&해석 관련 로직
        direct_response = _maybe_handle_large_script(analysis_query, documents_info, user_language)
        if direct_response:
            return direct_response
        
        # LangGraph로 분석 수행
        result = analyze_document_with_langgraph(
            document_content=content,
            user_query=user_input,
            max_analysis_tokens=8000,
            max_final_answer_tokens=1600
        )

# 에이전트 구동 시 (250609)

    #     return result # <-- 전체 결과 딕셔너리 반환

    # except Exception as e:
    #     logging.error(f"[DocAnalysis] Error in summarize_document_langgraph: {e}")
    #     # 일관된 딕셔너리 형태로 오류 반환
    #     return {
    #         "status": "error",
    #         "message": f"An error occurred: {str(e)}",
    #         "raika_response": f"*낑낑* 문서 처리 중 오류가 발생했어: {str(e)}"
    #     }

# 챗봇 구동 시 (250609)

        if result["status"] == "success":
            return result["raika_response"]
        else:
            # 오류 시 기본 메시지 반환
            if user_language == "ko":
                return f"*귀를 축 늘어뜨리며* 미안해... 문서 분석 중에 문제가 생겼어. {result.get('message', '알 수 없는 오류')}"
            else:
                return f"*droops ears* Sorry... I encountered a problem analyzing the document. {result.get('message', 'Unknown error')}"
            
    except Exception as e:
        logging.error(f"[DocAnalysis] Error in summarize_document_langgraph: {e}")
        if user_language == "ko":
            return f"*낑낑* 문서 처리 중 오류가 발생했어: {str(e)}"
        else:
            return f"*whimpers* An error occurred while processing the document: {str(e)}"


# --- 1. PDF 전용 '문맥 검색' 고속 함수 (251110) ---
async def get_context_from_pdf_cache_async(
    session_id: str,
    file_hash: str,
    query: str,
    redis_mgr: Any, # FastAPI의 RedisManager 인스턴스
    top_k: int = 5,
) -> Optional[str]:
    """
    (PDF 전용) Redis에 캐시된 RAG 데이터(청크+임베딩)를 비동기적으로 로드,
    사용자 질문(query)과 가장 관련성이 높은 문맥(context) 문자열을 반환
    이 함수는 LLM을 호출하지 않으며, 비동기적으로 처리되어 고속 검색을 지원
    """
    global embedding_model

    if not redis_mgr:
        logging.error("[PDF Context] RedisManager 인스턴스가 제공되지 않았습니다.")
        return None

    if embedding_model is None:
        try:
            # 모델이 로드되지 않았다면 동기적으로 로드 시도
            load_embedding_model()
        except Exception as e:
            logging.error(f"[PDF Context] 임베딩 모델 로드 실패: {e}")
            return None

    try:
        # 1. Redis에서 캐시된 (청크, 임베딩) 로드
        logging.info(f"[PDF Context] Redis에서 RAG 캐시 조회 시작: session={session_id}, hash={file_hash}")
        cached_data = await redis_mgr.load_pdf_rag_cache(session_id, file_hash)

        if cached_data is None:
            logging.warning(
                f"[PDF Context] RAG 캐시 미스: session={session_id}, hash={file_hash}. "
                "백그라운드 작업이 아직 진행 중이거나 캐시가 만료되었을 수 있습니다."
            )
            # 캐시가 없으면 None을 반환하여, 호출 측에서 사용자에게 아직 읽는 중임을 알리도록 함
            return None
            
        chunks, embeddings_array = cached_data
        
        if not chunks:
            logging.error(f"[PDF Context] RAG 캐시에 청크가 없습니다: session={session_id}, hash={file_hash}")
            return None
        
        if embeddings_array is None:
            logging.error(f"[PDF Context] RAG 캐시에 임베딩 배열이 None입니다: session={session_id}, hash={file_hash}")
            return None
            
        if embeddings_array.shape[0] == 0:
            logging.error(
                f"[PDF Context] RAG 캐시의 임베딩 배열이 비어있습니다: "
                f"session={session_id}, hash={file_hash}, shape={embeddings_array.shape}"
            )
            return None
        
        logging.info(
            f"[PDF Context] RAG 캐시 로드 성공: session={session_id}, hash={file_hash}, "
            f"청크={len(chunks)}, 임베딩 shape={embeddings_array.shape}"
        )

        # 2. 관련 청크 검색 (CPU-bound 작업이므로 동기 함수 그대로 사용)
        chunks_with_embeddings = list(zip(chunks, embeddings_array))

        relevant_results = retrieve_relevant_chunks(
            query, 
            chunks_with_embeddings, 
            top_k=top_k
        )
        
        if not relevant_results:
            logging.warning(f"[PDF Context] 질문과 관련된 문맥을 찾지 못했습니다.")
            return "문서에서 질문과 직접 관련된 정보를 찾지 못했습니다." if detect_language(query) == 'ko' else "Could not find directly relevant information in the document for the query."
            
        # 3. 최종 문맥 문자열 조합
        context_parts = []
        if detect_language(query) == 'ko':
            for chunk_text, sim_score in relevant_results:
                context_parts.append(f"[문서 발췌 (유사도: {sim_score:.2f})]:\n{chunk_text}")
        else:
            for chunk_text, sim_score in relevant_results:
                context_parts.append(f"[Excerpt from document (relevance: {sim_score:.2f})]:\n{chunk_text}")

        
        context_string = "\n\n---\n\n".join(context_parts)
        logging.info(f"[PDF Context] 성공적으로 문맥을 검색했습니다. (길이: {len(context_string)})")
        return context_string
        
    except Exception as e:
        logging.error(f"[PDF Context] RAG 캐시 검색 중 오류: {e}")
        return f"문맥 검색 중 오류 발생: {e}" if detect_language(query) == 'ko' else f"Error during context retrieval: {e}"

# --- 2. (기존) 일반 문서용 LangGraph 버전 RAG 응답 생성 함수 ---
# PDF가 아닌 파일(문서)에 대해서만 호출되는 함수

def generate_rag_response_langgraph(query: str, content: str, language=None) -> str:
    """
    LangGraph를 사용한 RAG 응답 생성 (기존 generate_rag_response 함수 대체)
    
    Args:
        query: 사용자 질의
        content: 문서 내용
        language: 언어 코드
    
    Returns:
        생성된 응답
    """
    try:
        # LangGraph로 분석 수행
        result = analyze_document_with_langgraph(
            document_content=content,
            user_query=query,
            max_analysis_tokens=8000,
            max_final_answer_tokens=1600
        )
        
        if result["status"] == "success":
            # Raika 페르소나가 이미 적용된 응답 반환
            return result["raika_response"]
        else:
            # 오류 시 기존 generate_rag_response 함수 폴백
            logging.warning("[DocAnalysis] Falling back to original generate_rag_response")
            return generate_rag_response(query, content, language)
            
    except Exception as e:
        logging.error(f"[DocAnalysis] Error in generate_rag_response_langgraph: {e}")
        # 오류 시 기존 함수 사용
        return generate_rag_response(query, content, language)

# 메인 함수
if __name__ == "__main__":
    set_model_and_processor()
    load_embedding_model()
    
    # GoogleSearch_Gemma 모듈 초기화
    GoogleSearch_Gemma.set_model_and_processor(model, processor)
    GoogleSearch_Gemma.initialize_and_get_compiled_graph()
    
    # 테스트
    file_path = r"c:\WolfCode\WolfCode_Language_Specification_V2.txt"
    user_query = """Please implement a Bubble Sort algorithm in WolfCode based on the syntax and examples provided in 'WolfCode_Language_Specification_V2.txt'."""

    # file_path = r"C:\Raika\black_hole.py"
    # user_query = "Summarize the corresponding Python script and code it."

    if os.path.exists(file_path):
        print("Starting LangGraph-enhanced document analysis...")
        response = summarize_document_langgraph(file_path, user_query)
        print("\nFinal Response:")
        print(response)
    else:
        print(f"File not found at {file_path}")



# if __name__ == "__main__":

#     set_model_and_processor()  # 모델과 토크나이저 로드
#     load_embedding_model()  # 임베딩 모델 로드

#     # file_path = r"C:\Raika\black_hole.py"
#     # user_query = "Summarize the corresponding Python script and code it."
#     # user_query = "첨부한 파이썬 스크립트를 요약해서 한글로 설명해 볼래?"

#     file_path = r"c:\WolfCode\WolfCode_Language_Specification_V2.txt"
#     user_query = """Please implement a Bubble Sort algorithm in WolfCode based on the syntax and examples provided in 'WolfCode_Language_Specification_V2.txt'.

# Specifically:
# 1. Sort this Pack of Bone values in ascending order: [5, 1, 9, 3, 7]
# 2. Use ONLY the operators and control structures defined in sections 1. - 8. of the specification
# 3. Follow the program structure with 'Pack gathering' at the start and 'Pack dispersing' at the end
# 4. Use the 'Howl' command to display the final sorted Pack

# Focus on implementing a simple Bubble Sort that compares adjacent elements and swaps them if they're in the wrong order, repeating until the list is sorted."""

#     if os.path.exists(file_path):
#         print("Starting advanced RAG-enhanced document analysis...")
#         response = summarize_document(file_path, user_query)
#         print("\nFinal Response:")
#         print(response)
#     else:
#         print(f"File not found at {file_path}. Please check the file path and try again.")
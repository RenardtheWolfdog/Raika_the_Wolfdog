"""
Redis 연동 유틸리티

이 모듈은 다음 역할을 수행합니다.
- config.ini 또는 환경변수로부터 Redis 접속 정보를 읽어 비동기 클라이언트를 생성
- 세션별 대화 상태(특히 '답변 계속' 상태) 저장/로딩/삭제
- 세션별 첨부 미디어/문서 메타데이터 캐시 (최근 N개 유지)


이 모듈은 한편으로 다음 역할도 수행합니다. (Vector DB 확장판)
1. Redis 접속 및 비동기 클라이언트 생성
2. RediSearch를 이용한 Vector Index 자동 생성 (Schema 정의)
3. 대화 턴(Turn)의 벡터/키워드 데이터 저장 및 검색
- 목적: Hybrid Memory-Aware Retrieval System의 기반 데이터소스로 활용

이 모듈은 Redis를 단순 캐시가 아닌 '벡터 데이터베이스(Vector DB)'로 활용하기 위한 핵심 기능을 제공합니다.
RediSearch 모듈을 사용하여 텍스트 임베딩(Embedding)을 저장하고, KNN(K-Nearest Neighbors) 알고리즘을 통해
의미적으로 가장 유사한 과거 대화 내용을 고속으로 검색합니다.

[주요 기능]
1. Redis 비동기 클라이언트 설정 및 연결 관리
2. Hybrid Search를 위한 인덱스(Index) 자동 생성 (Vector + Tag + Text)
3. 대화 내용의 벡터화 저장 및 유사도 검색 (KNN)


설정 방법(config.ini 예시):

  [REDIS]
  host = 127.0.0.1
  port = 6379
  db = 0
  username = 
  password = YOUR_REDIS_PASSWORD
  ssl = false
  ttl_seconds = 86400 <- 24시간 뒤에 캐시 삭제

환경변수로도 오버라이드 가능:
  REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_USERNAME, REDIS_PASSWORD, REDIS_SSL

의존성: redis>=5 (redis.asyncio 사용)
"""

from __future__ import annotations

import os
import json
import asyncio
from typing import Any, Dict, List, Optional, Union

import configparser
from redis import asyncio as aioredis
from redis.commands.search.field import TextField, VectorField, TagField, NumericField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import numpy as np
from typing import Tuple

# 환경 변수 로드 핼퍼
def _get_cfg_bool(value: str | None, default: bool = False) -> bool:
    """환경변수 값을 bool로 변환하는 함수"""
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


class RedisManager:
    """Redis 관리 클래스 (비동기).

    Redis를 Vector DB 및 세션 저장소로 활용하는 매니저 클래스
    RediSearch 기능을 통해 Hybrid Search(Vector + Metadata)를 지원

    - 키 스키마
      session:{sid}:continuation        -> JSON (메인 LLM의 '답변 계속' 상태)
      session:{sid}:media_list          -> LIST(JSON) (최근 업로드 미디어 메타데이터)
      session:{sid}:doc_list            -> LIST(JSON) (최근 업로드 문서 메타데이터)
      session:{sid}:conv_turns          -> LIST(JSON) (대화 턴 캐시: user+assistant 한 쌍)
    """

    def __init__(self, client: aioredis.Redis, default_ttl: int = 86400):
        self.client = client
        self.default_ttl = default_ttl
        self.vector_dim = 384 # Sentence-BERT (MiniLM-L6-V2 등) 임베딩 차원 수
        self.index_name = "idx:raika_memory" # 생성할 Redis 검색 인덱스 이름

    @classmethod
    async def create_from_config(cls) -> "RedisManager":
        """config.ini 또는 환경변수에서 접속 정보를 읽어 인스턴스 생성 및 인덱스 초기화"""
        cfg = configparser.ConfigParser()
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini")
        if os.path.exists(config_path):
            try:
                cfg.read(config_path, encoding="utf-8")
            except Exception:
                cfg.read(config_path)

        host = os.environ.get("REDIS_HOST") or cfg.get("REDIS", "host", fallback="127.0.0.1")
        port = int(os.environ.get("REDIS_PORT") or cfg.get("REDIS", "port", fallback="6379"))
        db = int(os.environ.get("REDIS_DB") or cfg.get("REDIS", "db", fallback="0"))
        username = os.environ.get("REDIS_USERNAME") or cfg.get("REDIS", "username", fallback=None)
        password = os.environ.get("REDIS_PASSWORD") or cfg.get("REDIS", "password", fallback=None)
        ssl_enabled = _get_cfg_bool(os.environ.get("REDIS_SSL") or cfg.get("REDIS", "ssl", fallback="false"))
        ttl = int(os.environ.get("REDIS_TTL") or cfg.get("REDIS", "ttl_seconds", fallback="86400"))

        client = aioredis.Redis(
            host=host,
            port=port,
            db=db,
            username=username or None,
            password=password or None,
            ssl=ssl_enabled,
            decode_responses=True,  # 문자열 JSON 저장/로드 용이
        )

        manager = cls(client, ttl)

        # [중요] 벡터 검색 인덱스가 없으면 생성 (비동기 초기화)
        try:
            await manager._create_index_if_not_exists()
        except Exception as e:
            print(f"[RedisManager] 인덱스 초기화 경고 (RediSearch 모듈 확인 필요): {e}")

        return manager

    async def _create_index_if_not_exists(self):
        """
        RediSearch Vector Index 생성
        이미 존재할 경우 생성을 건너 뜀
        """
        try:
            await self.client.ft(self.index_name).info()
            # print(f"[RedisManager] 인덱스 이미 존재: {self.index_name}")
        except Exception:
            # 인덱스가 없으면 생성 (Unknown Index Error 발생할 시)
            print(f"[RedisManager] 인덱스 생성 시작: {self.index_name}")

            schema = (
                TagField("session_id"),           # 세션별 필터링을 위한 태그
                TagField("role"),                 # 화자 (user/assistant)
                TextField("keywords"),            # 키워드 매칭용 (Sparse Search)
                NumericField("timestamp"),        # 시간 감쇠(Decay) 적용용
                VectorField("embedding",          # 임베딩 벡터 (Dense Search)
                    "FLAT", {                     # HNSW가 더 빠르지만, 데이터 적을땐 FLAT도 무방
                        "TYPE": "FLOAT32",              # 데이터 타입
                        "DIM": self.vector_dim,         # 벡터 차원 (384)
                        "DISTANCE_METRIC": "COSINE"     # 거리 계산 방법 (코사인 유사도)
                    }
                )
            )

            # 'mem:' 접두사가 붙은 Hash 키들을 자동으로 인덱싱하도록 정의
            definition = IndexDefinition(prefix=["mem:"], index_type=IndexType.HASH) # 세션별 태그 검색 가능하도록 설정
            await self.client.ft(self.index_name).create_index(schema, definition=definition)
            print(f"[RedisManager] 인덱스 생성 완료: {self.index_name}")
        except Exception as e:  # 인덱스 생성 실패 시 예외 발생
            print(f"[RedisManager] 인덱스 생성 실패: {e}")
            raise e

    """
    Hybrid Memory-Aware Retrieval System
    - 메모리 기반 검색 시스템
    - 대화 턴을 벡터와 함께 저장하고, 검색 시 벡터 유사도를 계산하여 상위 결과를 반환
    - 키워드 매칭을 통해 정확도를 높임
    - 세션별 필터링을 통해 특정 세션의 데이터만 검색 가능
    - 벡터 검색 시 RediSearch의 KNN 기능을 사용하여 빠른 검색 속도를 제공
    - 벡터 검색 결과를 파싱하여 봇이 쉽게 이해할 수 있는 형태로 반환
    """

    # -----------------------
    # A. Hybrid Memory 저장 로직
    # ------------------------

    async def save_memory_turn(self, session_id: str, turn_data: Dict[str, Any]):
        f"""
        대화 한 턴(user 프롬프트 + bot 응답)을 벡터와 함께 Redis Hash로 저장함
        키 패턴: mem:{session_id}:{turn_data['timestamp']}
        """
        try:
            # 고유 키 생성 (세션ID + timestamp 기반)
            key = f"mem:{session_id}:{turn_data['timestamp']}"

            # 임베딩 벡터(float 리스트나 배열)를 numpy array => 바이너리(bytes) 변환 (Redis 저장을 위해)
            embedding = turn_data.get('embedding')
            if embedding is not None:
                # float 32 리스트나 배열을 바이너리(bytes)로 변환
                if isinstance(embedding, list):
                    embedding = np.array(embedding, dtype=np.float32)
                embedding_bytes = embedding.astype(np.float32).tobytes()
            else:
                embedding_bytes = b""

            # Hash 데이터 구성, 저장할 데이터 매핑(Key-Value)
            mapping = {
                "session_id": session_id,
                "role": turn_data['role'],
                "text": turn_data['text'],
                "keywords": ",".join(turn_data.get('keywords', [])), # Tag 필드용
                "topic_id": str(turn_data.get('topic_id', 0)),
                "timestamp": turn_data['timestamp'],
                "embedding": embedding_bytes
            }

            # Redis Hash에 저장
            await self.client.hset(key, mapping=mapping)

            # TTL 설정 (7일) - 장기 기억이므로 상대적으로 길게 설정
            await self.client.expire(key, 604800) # 7일 = 604800초

        except Exception as e:
            print(f"[RedisManager] 메모리 턴 저장 실패: {e}")

    # -----------------------
    # 2. Hybrid Retrieval (Vector Search)
    # -----------------------
    async def search_vectors(self, session_id: str, query_vector: List[float], top_k: int = 10) -> List[Dict]:
        f"""
        Vector Similarity Search (KNN) 수행
        RediSearch 쿼리 문법: "(@session_id:{...})=>[KNN k @embedding $vec AS score]"
        1. 먼저 해당 session_id를 가진 데이터만 필터링
        2. 그 중에서, 임베딩 벡터 간의 거리가 가장 가까운 k개를 찾음
        """
        try:
            # 1. 쿼리 준비
            # 특정 세션 내에서만 검색하도록 필터링
            base_query = f"(@session_id:{{{session_id}}})"

            # KNN(K-Nearest Neighbors) 검색 쿼리 결합
            # vector_score는 코사인 유사도 값(작을수록 유사도 높음)으로 거리 값을 반환
            q = Query(f"{base_query}=>[KNN {top_k} @embedding $vec AS score]")\
                .sort_by("vector_score")\
                .return_fields("text", "role", "keywords", "timestamp", "vector_score")\
                .dialect(2) # RediSearch 2.0 호환 쿼리 문법

            # 2. 쿼리 벡터를 바이너리(bytes)로 변환
            vec_bytes = np.array(query_vector, dtype=np.float32).tobytes()
            params = {"vec": vec_bytes}

            # 3. 검색 실행
            # [수정] redis-py 최신 버전에서는 params 대신 query_params를 사용하는 경우가 있음
            # 또는 버전에 따라 params가 맞을 수도 있으나, 오류가 발생하므로 query_params로 변경 시도
            try:
                results = await self.client.ft(self.index_name).search(q, query_params=params)
            except TypeError:
                # 만약 query_params도 아니라면 params로 재시도 (혹은 구버전 호환)
                results = await self.client.ft(self.index_name).search(q, params=params)

            # 4. 결과 파싱 및 반환
            parsed_results = []
            for doc in results.docs:
                parsed_results.append({
                    "text": doc.text,
                    "role": doc.role,
                    "keywords": doc.keywords.split(",") if doc.keywords else [],
                    "timestamp": float(doc.timestamp),
                    "vector_score": float(doc.vector_score) # Cosine Distanse (0에 가까울수록 유사도 높음)
                })
            return parsed_results
        except Exception as e:
            print(f"[RedisManager] 벡터 검색 실패: {e}")
            return []


    # -----------------------
    # B. 기존 세션 상태 관리 로직 (대화 지속성, 미디어/문서 캐시, 대화 턴 캐시, PDF RAG 캐시)
    # -----------------------

    # -----------------------
    # 1) '답변 계속' 상태 저장소
    # -----------------------
    def _key_cont(self, session_id: str) -> str:
        return f"session:{session_id}:continuation"

    async def save_continuation_state(self, session_id: str, state: Dict[str, Any], ttl: Optional[int] = None) -> None:
        try:
            key = self._key_cont(session_id)
            await self.client.set(key, json.dumps(state), ex=ttl or self.default_ttl)
        except Exception:
            pass

    async def load_continuation_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            raw = await self.client.get(self._key_cont(session_id))
            return json.loads(raw) if raw else None
        except Exception:
            return None

    async def clear_continuation_state(self, session_id: str) -> None:
        try:
            await self.client.delete(self._key_cont(session_id))
        except Exception:
            pass

    # -----------------------
    # 2) 파일(미디어/문서) 캐시
    # -----------------------
    def _key_media(self, session_id: str) -> str:
        return f"session:{session_id}:media_list"

    def _key_docs(self, session_id: str) -> str:
        return f"session:{session_id}:doc_list"

    async def append_media(self, session_id: str, meta: Dict[str, Any], max_items: int = 50, ttl: Optional[int] = None) -> None:
        try:
            key = self._key_media(session_id)
            await self.client.lpush(key, json.dumps(meta))
            await self.client.ltrim(key, 0, max_items - 1)
            await self.client.expire(key, ttl or self.default_ttl)
        except Exception:
            pass

    async def list_media(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            key = self._key_media(session_id)
            items = await self.client.lrange(key, 0, max(0, limit - 1))
            return [json.loads(x) for x in items]
        except Exception:
            return []

    async def append_document(self, session_id: str, meta: Dict[str, Any], max_items: int = 50, ttl: Optional[int] = None) -> None:
        try:
            key = self._key_docs(session_id)
            await self.client.lpush(key, json.dumps(meta))
            await self.client.ltrim(key, 0, max_items - 1)
            await self.client.expire(key, ttl or self.default_ttl)
        except Exception:
            pass

    async def list_documents(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        try:
            key = self._key_docs(session_id)
            items = await self.client.lrange(key, 0, max(0, limit - 1))
            return [json.loads(x) for x in items]
        except Exception:
            return []

    # -----------------------
    # 3) 대화 턴 캐시 (Vector-DB 유사 활용을 위한 원본 보관)
    # -----------------------
    def _key_conv(self, session_id: str) -> str:
        """
        세션별 대화 턴 리스트 키를 생성합니다.

        - 각 아이템은 JSON 직렬화된 dict이며 다음 예시 형태를 권장합니다.
          {
            "turn_index": int,                # 대화 내 턴 번호 (작을수록 과거)
            "user_text": str,                # 사용자 프롬프트
            "assistant_text": str,           # 모델 응답
            "combined_text": str,            # 검색 편의용 합쳐진 텍스트 (User/Assistant 포함)
            "embedding": List[float] | None, # 임베딩(옵션). 없으면 검색 시 계산/무시 가능
            "ts": float                      # timestamp (epoch seconds)
          }
        """
        return f"session:{session_id}:conv_turns"

    async def append_conversation_turn(
        self,
        session_id: str,
        turn: Dict[str, Any],
        max_items: int = 2000,
        ttl: Optional[int] = None,
    ) -> None:
        """
        대화 턴을 세션 리스트에 추가합니다. 가장 최근 항목이 인덱스 0이 되도록 LPUSH 사용.

        - max_items: 보존할 최대 턴 수. 초과 시 오래된 항목을 잘라냅니다.
        - ttl: 키 TTL(초). None이면 default_ttl 사용.
        """
        try:
            key = self._key_conv(session_id)
            await self.client.lpush(key, json.dumps(turn))
            await self.client.ltrim(key, 0, max(0, max_items - 1))
            await self.client.expire(key, ttl or self.default_ttl)
        except Exception:
            # 캐시 실패는 치명적이지 않으므로 조용히 무시
            pass

    async def list_conversation_turns(
        self, session_id: str, start: int = 0, end: int = -1
    ) -> List[Dict[str, Any]]:
        """
        세션의 대화 턴 캐시를 구간 조회합니다. 기본은 전체.

        반환은 최신이 먼저 오는 순서(인덱스 0 = 최신)입니다.
        """
        try:
            key = self._key_conv(session_id)
            items = await self.client.lrange(key, start, end)
            return [json.loads(x) for x in items]
        except Exception:
            return []

    async def clear_conversation_turns(self, session_id: str) -> None:
        """세션의 대화 턴 캐시를 모두 삭제합니다."""
        try:
            await self.client.delete(self._key_conv(session_id))
        except Exception:
            pass

    # -----------------------
    # 4) PDF RAG 캐시 (청크 + 임베딩)
    # -----------------------
    def _key_pdf_rag_cache(self, session_id: str, file_hash: str) -> str:
        """PDF의 RAG 전처리(청크 + 임베딩) 결과를 저장하는 키"""
        return f"session:{session_id}:pdf_rag_cache:{file_hash}"

    async def save_pdf_rag_cache(
        self,
        session_id: str,
        file_hash: str,
        chunks: List[str],
        embeddings: np.ndarray,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        PDF의 RAG 전처리(청크 + 임베딩 리스트) 결과를 JSON 형식으로 저장
        """
        if not session_id:
            print(f"Redis Error: PDF RAG 캐시 저장 실패 - session_id가 비어있음")
            return False
        if not file_hash:
            print(f"Redis Error: PDF RAG 캐시 저장 실패 - file_hash가 비어있음 (session={session_id})")
            return False
        if not chunks:
            print(f"Redis Error: PDF RAG 캐시 저장 실패 - chunks가 비어있음 (session={session_id}, hash={file_hash})")
            return False
        if embeddings is None:
            print(f"Redis Error: PDF RAG 캐시 저장 실패 - embeddings가 None (session={session_id}, hash={file_hash})")
            return False
        
        try:
            key = self._key_pdf_rag_cache(session_id, file_hash)
            
            # np.ndarray를 JSON 직렬화 가능한 list[list[float]] 형식으로 변환
            try:
                embedding_list = embeddings.astype(float).tolist()
                print(f"Redis: 임베딩 변환 완료 - shape: {embeddings.shape}, list 길이: {len(embedding_list)}")
            except Exception as convert_err:
                print(f"Redis Error: 임베딩 변환 실패 (session={session_id}, hash={file_hash}): {convert_err}")
                return False

            payload = {
                "chunks": chunks,
                "embeddings": embedding_list,
                "created_at": asyncio.get_event_loop().time()
            }
            
            # JSON 직렬화 시도
            try:
                json_payload = json.dumps(payload)
                payload_size_mb = len(json_payload) / (1024 * 1024)
                print(f"Redis: JSON 페이로드 생성 완료 - 크기: {payload_size_mb:.2f}MB, 청크 수: {len(chunks)}")
            except Exception as json_err:
                print(f"Redis Error: JSON 직렬화 실패 (session={session_id}, hash={file_hash}): {json_err}")
                return False

            # Redis에 저장
            await self.client.set(
                key,
                json_payload,
                ex=ttl or self.default_ttl
            )
            print(f"Redis: PDF RAG 캐시 저장 성공 - key: {key}, TTL: {ttl or self.default_ttl}초")
            return True
        except Exception as e:
            print(f"Redis Error: PDF RAG 캐시 저장 실패 (session={session_id}, hash={file_hash}): {e}")
            import traceback
            print(traceback.format_exc())
            return False

    async def load_pdf_rag_cache(
        self, session_id: str, file_hash: str
    ) -> Optional[Tuple[List[str], np.ndarray]]:
        """
        Redis에서 RAG 캐시를 로드하여 (청크, 임베딩 np.ndarray) 튜플로 반환합니다.
        """
        if not session_id:
            print(f"Redis: PDF RAG 캐시 로드 실패 - session_id가 비어있음")
            return None
        if not file_hash:
            print(f"Redis: PDF RAG 캐시 로드 실패 - file_hash가 비어있음 (session={session_id})")
            return None
        
        try:
            key = self._key_pdf_rag_cache(session_id, file_hash)
            print(f"Redis: PDF RAG 캐시 조회 시작 - key: {key}")
            raw = await self.client.get(key)
            
            if not raw:
                print(f"Redis: PDF RAG 캐시 미스 - key가 존재하지 않음: {key}")
                return None
            
            print(f"Redis: 캐시 데이터 발견 - 크기: {len(raw) / 1024:.2f}KB")
            
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as json_err:
                print(f"Redis Error: JSON 파싱 실패 (session={session_id}, hash={file_hash}): {json_err}")
                return None
            
            chunks = payload.get("chunks", [])
            embeddings_list = payload.get("embeddings", [])
            
            if not chunks:
                print(f"Redis Error: 페이로드에 chunks가 없음 (session={session_id}, hash={file_hash})")
                return None
            if not embeddings_list:
                print(f"Redis Error: 페이로드에 embeddings가 없음 (session={session_id}, hash={file_hash})")
                return None
            
            # list[list[float]]를 np.ndarray로 복원
            try:
                embeddings_array = np.array(embeddings_list, dtype=np.float32)
                print(
                    f"Redis: PDF RAG 캐시 로드 성공 - "
                    f"청크: {len(chunks)}, 임베딩 shape: {embeddings_array.shape}"
                )
            except Exception as array_err:
                print(f"Redis Error: 임베딩 배열 복원 실패 (session={session_id}, hash={file_hash}): {array_err}")
                return None
            
            return chunks, embeddings_array
        except Exception as e:
            print(f"Redis Error: PDF RAG 캐시 로드 중 예외 발생 (session={session_id}, hash={file_hash}): {e}")
            import traceback
            print(traceback.format_exc())
            return None
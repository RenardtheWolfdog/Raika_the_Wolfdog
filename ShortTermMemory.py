# ShortTermMemory.py

"""
ShortTermMemory.py - Hybrid Memory-Aware Dialogue Retrieval System (Korean + English Support)

[개요]
이 모듈은 챗봇 Raika의 '장기 기억(Long-term Memory)'을 담당하는 핵심 로직입니다.
단순히 키워드만 매칭하는 방식의 한계를 넘어, Redis Vector Store를 활용해 
'의미적 유사도(Semantic Similarity)'와 '명사구 매칭(Keyword Matching)'을 결합한
하이브리드 검색 시스템을 구현했습니다.

[주요 변경점: Hybrid & Low-Latency & No-Decay]
1. 다국어 지원 (Bilingual): 입력 언어(한국어/영어)를 자동 감지하여, 
   한국어는 KoELECTRA, 영어는 SpaCy 모델을 사용해 핵심 키워드를 추출합니다.
2. 비동기 병렬 처리: 키워드 추출과 임베딩 생성을 동시에 수행하여 지연 시간(Latency)을 최소화했습니다.
"""

from collections import defaultdict
import re
import numpy as np
import logging
from typing import List, Tuple, Dict, Set
import asyncio
import time

# coreferee 조건부 import
try:
    import coreferee  # type: ignore # noqa
    coreferee_available = True
except ImportError:
    print("coreferee 모듈을 찾을 수 없습니다. 코어퍼런스 해결 기능이 제한됩니다.")
    coreferee_available = False

# 자연어 처리(NLP) 모듈
import spacy # 영어 모델
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
# NLTK는 선택적 의존성으로 처리 (미설치 환경 폴백)
try:
    import nltk  # type: ignore
    from nltk.tokenize import sent_tokenize as nltk_sent_tokenize  # type: ignore
    _nltk_available = True
except Exception:
    _nltk_available = False
    def nltk_sent_tokenize(text: str):
        # 간단한 문장 분리 폴백: 마침표/물음표/느낌표 기준
        try:
            import re as _re
            return [s.strip() for s in _re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        except Exception:
            return [text]

# Redis 기반 캐시/검색 유틸
try:
    from redis_utils import RedisManager
    _redis_available = True
except Exception:
    _redis_available = False

import torch
from transformers import AutoTokenizer, AutoModel

# --- 설정값 (Tuning Hyperparameters) ---
# (현재 대화 주제와 유사한) 검색 점수 산정 시 각 요소의 반영 비율
# 두 값의 합이 1.0이 되도록 설정
ALPHA_VECTOR = 0.7   # 벡터 유사도 가중치 (의미적 맥락 중시)
BETA_KEYWORD = 0.3   # 키워드 매칭 가중치 (구체적 명사 일치 중시)


""" English 처리 - SpaCy 모델 기반 """
# NLTK의 문장 토크나이저 다운로드 (있을 때만)
try:
    if _nltk_available:
        nltk.download('punkt', quiet=True)  # type: ignore
except Exception:
    pass

# spaCy 언어 모델 로드
try:
    nlp_en = spacy.load("en_core_web_sm")
except Exception as e:
    logging.warning(f"spaCy 모델 로드 실패, blank('en')로 폴백: {e}")
    nlp_en = spacy.blank("en")
    # 영어 NER / noun_chunks 기능 제한 플래그
    NER_AVAILABLE = False
    PARSER_AVAILABLE = False
else:
    NER_AVAILABLE = nlp_en.has_pipe("ner")
    PARSER_AVAILABLE = nlp_en.has_pipe("parser")

# Coreferee는 이미 조건부 import 되어 있음
if coreferee_available:
    try:
        if not nlp_en.has_pipe("coreferee"):
            nlp_en.add_pipe("coreferee")
        print("Coreferee 코어퍼런스 해석기 로드 성공")
    except Exception as e:
        print(f"Coreferee 파이프라인 추가 실패: {e}")
        coreferee_available = False
else:
    print("Coreferee 없이 spaCy 모델만 로드됨")

""" 한국어 처리 - KoELECTRA 및 KSS 문장 분리기 기반 개선"""
try:
    # torch 2.6 미만에서는 취약점 회피 정책으로 torch.load가 차단될 수 있어 스킵
    _torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
    if _torch_version >= (2, 6):
        from transformers import AutoTokenizer, AutoModel
        korean_model_name = "monologg/koelectra-small-v3-discriminator"
        korean_tokenizer = AutoTokenizer.from_pretrained(korean_model_name)
        korean_model = AutoModel.from_pretrained(korean_model_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        korean_model = korean_model.to(device)
        korean_model.eval()
        print("한국어 처리: KoELECTRA 키워드 추출기 로드 성공")
        korean_model_loaded = True
    else:
        raise RuntimeError("torch<2.6 환경: KoELECTRA 로드를 건너뜁니다")
except Exception as e:
    print(f"KoELECTRA 로드 오류: {e}")
    korean_model_loaded = False
    korean_tokenizer = None
    korean_model = None

# tfidf_vectorizer = TfidfVectorizer()    # TF-IDF 키워드 추출기 초기화
# (영어 키워드 추출에만 사용되는 예비 코드)

# 3. Sentence Transformer (임베딩 생성용) - 지연 로딩 적용
# Sentence Transformer 지연 로딩 래퍼 (서버 시작 시 불필요한 무거운 임포트 방지, 실제 임베딩이 필요할 때만 로드)
class _LazySentenceModel:
    _model = None

    def _ensure(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # 지연 임포트
                print("[Memory] Sentence-BERT 모델 로딩 중...")
                self._model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                print("[Memory] Sentence-BERT 모델 로딩 완료")
            except Exception as _e:
                print(f"[Memory] Sentence-BERT 모델 로딩 실패: {_e}")
                logging.warning(f"SentenceTransformer 로드 실패: {_e}")
                raise

    def encode(self, texts, *args, **kwargs):
        self._ensure()
        return self._model.encode(texts, *args, **kwargs)

sentence_model = _LazySentenceModel()

# --- 유틸리티 함수 ---

# 언어 감지 함수
def detect_language(text: str) -> str:
    """
    텍스트에 한글이 포함되어 있는지 확인하여 언어를 감지
    :return: "ko" (한국어 포함), "en" (그 외)
    """
    # 유니코드 범위로 한글 존재 여부 확인
    if any(0xAC00 <= ord(char) <= 0xD7A3 for char in text):
        return "ko"
    return "en"


class HybridMemorySystem:
    """
    Redis 기반의 하이브리드 메모리 검색 시스템 클래스
    """
    def __init__(self, redis_manager: RedisManager):
        self.redis = redis_manager
        self.stop_words = self._load_stopwords()

    def _load_stopwords(self) -> Set[str]:
        """
        검색 품질을 높이기 위해 의미 없는 단어(불용어) 목록을 정의
        :return: 불용어 집합(Set)
        """
        return {
            # 한국어 불용어
            '나', '너', '저', '그것', '이것', '저것', '있다', '없다', 
            '그', '이', '저', '안녕', '반가워', 'raika', 'renard', 
            '은', '는', '이', '가', '을', '를', '에', '의', '로', '으로',
            '오늘', '지금', '내일', '어제',
            # 영어 불용어
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'is', 'are', 'am', 'was', 'were',
            'the', 'a', 'an', 'this', 'that', 'there', 'here', 'to', 'of', 'in', 'on'
        }

    # --- 1. 데이터 전처리 및 저장 (Input Processing Layer) ---

    async def save_turn(self, session_id: str, role: str, text: str):
        """
        대화 한 턴을 분석하여 Redis에 저장 (비동기 실행)
        """
        if not text or len(text.strip()) < 2:
            return

        try:
            # 1. 임베딩 생성 (Dense Vector) - CPU/GPU 연산
            # asyncio.to_thread를 사용하여 무거운 연산을 별도 스레드에서 실행, 이벤트 루프 블로킹 방지
            embedding = await asyncio.to_thread(sentence_model.encode, text)
            embedding_list = embedding.tolist()

            # 2. 키워드 추출 (Sparse Tags) - 언어 감지 후 적절한 모델 사용
            lang = detect_language(text)
            keywords = await asyncio.to_thread(self._extract_keywords, text, lang)

            # 3. 데이터 패키징
            turn_data = {
                "role": role,
                "text": text,
                "keywords": keywords,
                "embedding": embedding_list,
                "timestamp": time.time(),
                "topic_id": 0 # 추후 토픽 클러스터링 로직 추가 시 활용 가능
            }

            # 4. Redis에 저장
            await self.redis.save_memory_turn(session_id, turn_data)
            # logging.info(f"[Memory] 저장 완료: {text[:20]}... (키워드: {keywords})")

        except Exception as e:
            logging.error(f"[Memory] 저장 중 오류 발생: {e}")

    async def save_interaction(self, session_id: str, user_text: str, bot_text: str):
        """
        [개선] 질문(User)과 답변(Bot)을 하나의 문맥(Interaction)으로 묶어서 저장
        검색 시 질문의 의도와 답변의 정보를 동시에 찾을 수 있어 RAG 성능이 향상됨.
        """
        if not user_text or not bot_text:
            return

        try:
            # 1. 텍스트 결합 (명시적 구분자 사용)
            combined_text = f"Q: {user_text}\nA: {bot_text}"
            
            # 2. 임베딩 생성 (결합된 텍스트 전체에 대해)
            embedding = await asyncio.to_thread(sentence_model.encode, combined_text)
            embedding_list = embedding.tolist()

            # 3. 키워드 추출 (질문과 답변 각각에서 추출하여 합침)
            lang_user = detect_language(user_text)
            lang_bot = detect_language(bot_text)
            kw_user = await asyncio.to_thread(self._extract_keywords, user_text, lang_user)
            kw_bot = await asyncio.to_thread(self._extract_keywords, bot_text, lang_bot)
            # 중복 제거 및 병합
            combined_keywords = list(set(kw_user + kw_bot))

            # 4. 데이터 패키징
            turn_data = {
                "role": "interaction", # 역할은 상호작용으로 통일
                "text": combined_text, # 실제 저장되는 텍스트는 Q&A 합본
                "keywords": combined_keywords,
                "embedding": embedding_list,
                "timestamp": time.time(),
                "topic_id": 0
            }

            # 5. Redis에 저장
            await self.redis.save_memory_turn(session_id, turn_data)
            # logging.info(f"[Memory] 인터랙션 저장 완료: {combined_text[:30]}...")

        except Exception as e:
            logging.error(f"[Memory] 인터랙션 저장 중 오류 발생: {e}")


    def _extract_keywords(self, text: str, lang: str) -> List[str]:
        """
        언어별 특성에 맞는 키워드 추출 로직
        """
        keywords = []

        # A. 한국어 처리 (KoELECTRA 사용)
        if lang == "ko":
            if not korean_model_loaded:
                # 모델 로드 실패 시 정규식으로 명사(2글자 이상)만 간단히 추출
                keywords = re.findall(r'[가-힣]{2,}', text)
            else:
                try:
                    # KoELECTRA 토크나이징
                    inputs = korean_tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
                    with torch.no_grad():
                        outputs = korean_model(**inputs, output_attentions=True)
                    
                    # Attention Score가 높은 토큰을 중요 단어로 선정
                    # 마지막 레이어의 어텐션 평균을 사용
                    attentions = outputs.attentions[-1].mean(dim=1).squeeze().sum(dim=0).cpu().numpy()
                    tokens = korean_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                    
                    scored_tokens = []
                    for token, score in zip(tokens, attentions):
                        # 특수 토큰 및 1글자 제외
                        if token not in ['[CLS]', '[SEP]', '[PAD]'] and len(token) > 1:
                            clean_token = token.replace('##', '') # 서브워드 접두사 제거
                            if clean_token not in self.stop_words:
                                scored_tokens.append((clean_token, score))
                    
                    # 점수순 정렬 후 상위 5개 반환
                    scored_tokens.sort(key=lambda x: x[1], reverse=True)
                    keywords = [t[0] for t in scored_tokens[:5]]
                except Exception:
                    pass
            
            # [추가] 한국어 문장에 섞인 영어 단어(고유명사 등)도 놓치지 않도록 추출
            english_words = re.findall(r'[a-zA-Z0-9]+', text)
            for w in english_words:
                if len(w) > 2 and w.lower() not in self.stop_words:
                    keywords.append(w)

        # B. 영어 처리 (SpaCy 사용)
        else:
            if nlp_en is None:
                # SpaCy 로드 실패 시 띄어쓰기 기준으로 3글자 이상 단어 추출
                keywords = [w for w in text.split() if len(w) > 3 and w.lower() not in self.stop_words]
            
            else:
                try:
                    doc = nlp_en(text)
                    
                    # 1. 고유명사(Entities) 우선 추출 (사람, 조직, 날짜 등)
                    for ent in doc.ents:
                        if ent.text.lower() not in self.stop_words:
                            keywords.append(ent.text)
                    
                    # 2. 명사구(Noun Chunks) 추출
                    for chunk in doc.noun_chunks:
                        # 관사 등을 제외한 핵심 명사만 추출
                        root_text = chunk.root.text.lower()
                        if root_text not in self.stop_words and len(root_text) > 2:
                            keywords.append(root_text)
                    
                    # 중복 제거
                    keywords = list(set(keywords))[:5]
                except Exception:
                    pass

            # [추가] 영어 문장에 섞인 한국어 단어도 추출
            korean_words = re.findall(r'[가-힣]{2,}', text)
            keywords.extend(korean_words)

        return list(set(keywords))

    # --- 2. 하이브리드 검색 및 순위 결정 (Retrieval & Ranking Layer) ---

    async def retrieve_relevant_memories(self, session_id: str, query: str, top_k: int = 5) -> List[str]:
        """
        사용자 질문(query)과 관련된 과거 기억을 Hybrid Scoring 방식으로 인출
        오래된 기억도 정확도가 높다면 상위에 노출됨
        """
        if not query: return []

        # A. 쿼리 분석 (임베딩 생성 및 키워드 추출 병렬 실행)
        lang = detect_language(query)
        embedding_future = asyncio.to_thread(sentence_model.encode, query)
        keywords_future = asyncio.to_thread(self._extract_keywords, query, lang)
        
        # 두 작업을 동시에 기다림 (Latency 최적화)
        query_embedding, query_keywords = await asyncio.gather(embedding_future, keywords_future)
        query_keywords_set = set(query_keywords)

        # B. 1차 후보군 검색 (Vector Search - High Recall 전략)
        # Redis에서 벡터 유사도 기준으로 충분한 수(top_k * 20)의 후보를 가져옵니다.
        # 다국어 환경 등에서 벡터 유사도가 낮게 측정될 수 있으므로, 최대한 많은 후보를 가져와
        # 키워드 매칭 점수로 보정할 기회를 확보해야 합니다.
        candidates = await self.redis.search_vectors(
            session_id, 
            query_embedding.tolist(), 
            top_k=top_k * 20
        )

        if not candidates:
            return []

        # C. 2차 Re-ranking (Hybrid Scoring)
        scored_candidates = []

        for doc in candidates:
            # 1. Vector Score (의미적 유사도)
            # Redis는 Cosine Distance(거리)를 반환하므로, 유사도(Score)로 변환
            # 거리: 0(완전일치) ~ 2(완전반대)
            dist = float(doc['vector_score'])
            
            # [개선] 지수 함수를 사용하여 거리에 따른 점수 감소를 완만하게 만듦
            # dist=0 -> score=1.0, dist=1 -> score=0.36
            # 다국어 매칭처럼 거리가 멀게(1.0 이상) 나오는 경우에도 점수를 잃지 않도록 함
            vector_score = np.exp(-dist) 

            # 2. Keyword Score (키워드 매칭 정확도)
            # 쿼리 키워드가 검색된 문서 키워드에 얼마나 포함되어 있는지 계산
            doc_keywords = set(doc['keywords'])
            if query_keywords_set and doc_keywords:
                # 교집합 개수 / 쿼리 키워드 개수
                overlap = len(query_keywords_set.intersection(doc_keywords))
                keyword_score = overlap / len(query_keywords_set) 
            else:
                keyword_score = 0.0

            # 3. 최종 점수 계산 (Weighted Sum)
            final_score = (ALPHA_VECTOR * vector_score) + (BETA_KEYWORD * keyword_score)

            # 추후 유저 발화(User)인지 AI (Raika) 발화인지에 따라 미세 조정도 가능 (현재는 동일)
            # if doc['role'] == 'assistant': final_score *= 0.9 

            scored_candidates.append((final_score, doc['text'], dist))

        # D. 정렬 및 필터링
        # 점수가 높은 순서대로 정렬
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # 상위 top_k개 텍스트 반환 (중복 제거 및 현재 쿼리 제외)
        final_memories = []
        seen_texts = set()
        
        # 쿼리와 완전히 똑같은 문장(방금 말한 내용)은 제외하여 '메아리' 현상 방지
        query_clean = query.replace(" ", "")
        
        for score, text, dist in scored_candidates:
            # 1. 텍스트 완전 일치 필터링
            if text.replace(" ", "") == query_clean:
                continue
            
            # 2. 의미적 완전 일치 필터링 (벡터 거리가 너무 가까우면 스킵)
            # Cosine Distance: 0에 가까울수록 매우 유사. 
            # 0.05 미만이면 사실상 같은 문장(혹은 매우 유사한 변형)으로 간주
            if dist < 0.05:
                continue

            if text not in seen_texts:
                final_memories.append(text)
                seen_texts.add(text)
                
            if len(final_memories) >= top_k:
                break

        return final_memories

# --- 테스트용 메인 함수 ---
if __name__ == "__main__":
    async def test():
        print("Redis 연결 및 하이브리드 메모리 시스템 초기화...")
        try:
            rm = await RedisManager.create_from_config()
            memory = HybridMemorySystem(rm)
            
            sid = "test_session_dialogue_full_001"
            
            # [초기화] 테스트 시작 전, 기존 세션 메모리 삭제 (메아리 방지 및 정밀 테스트)
            print(f"세션 ID: {sid} 로 테스트 시작 (기존 메모리 초기화 중...)")
            deleted_count = 0
            async for key in rm.client.scan_iter(match=f"mem:{sid}:*"):
                await rm.client.delete(key)
                deleted_count += 1
            print(f"  -> 초기화 완료: {deleted_count}개의 과거 기억 삭제됨.")
            
            # 대화 시나리오 정의
            dialogue = [
                ("user", "안녕하세요, 오늘 기분이 어떠세요?"),
                ("bot", "좋아요! *멍멍!*"),
                ("user", "아, 강아지를 산책시키고 계시네요? 강아지 이름이 뭐예요?"),
                ("bot", "밤비라고 해요. *멍멍!*"),
                ("user", "무슨 종이에요?"),
                ("bot", "푸들이에요."),
                ("user", "몇 살이에요?"),
                ("bot", "5살이에요."),
                ("user", "너무 귀엽네요! 그나저나, Apple에서 새 아이폰을 출시했다는 소식 들으셨나요?"),
                ("bot", "네, 당연하죠. 지난 주에 구매했어요."),
                ("user", "정말 성능이 좋다고 하더라고요."),
                ("bot", "정말 좋아요! 디자인도 멋지고, 카메라도 선명해요."),
                ("user", "카메라로 강아지 사진 찍어볼 수 있나요?"),
                ("bot", "물론이죠, 여기 있어요."),
                ("user", "와, 카메라 렌즈가 세 개나 있네요?"),
                ("bot", "네, 카메라 렌즈가 세 개 있어요. 요즘 아이폰의 특징이죠."),
                ("user", "각각 어떤 용도인가요?"),
                ("bot", "사실 각각이 무엇을 위한 것인지 아직 잘 모르겠어요."),
                ("user", "아, 아쉽군요. 그런데 강아지 이름이 뭐였고, 무슨 종이었죠? 잊어버렸어요.")
            ]

            print("\n[대화 시뮬레이션 및 실시간 저장 - Q&A Interaction 방식]")
            
            # 이전 사용자 발화를 버퍼에 저장했다가, 봇 발화 시점에 합쳐서 저장
            last_user_text = ""
            
            for i, (role, text) in enumerate(dialogue):
                print(f"Turn {i+1:02d} [{role}]: {text}")
                
                if role == "user":
                    last_user_text = text
                    # 유저 단독 발화는 일단 저장하지 않고(또는 개별 저장하고), 
                    # 봇 응답과 묶어서 'Interaction'으로 저장하는 것이 핵심.
                    # 여기서는 테스트 목적상 '봇 응답'을 기다림.
                
                elif role == "bot":
                    if last_user_text:
                        # 질문(User) + 답변(Bot) = 하나의 Interaction으로 저장
                        await memory.save_interaction(sid, last_user_text, text)
                        last_user_text = "" # 버퍼 초기화
                    else:
                        # 봇 혼자 말하는 경우 (인사 등) -> 그냥 단독 저장
                        await memory.save_turn(sid, role, text)
                
                # 2. 중간 검색 테스트 (예: 강아지 종을 물어볼 때 이름 정보가 나오는지)
                # "무슨 종이에요?" 질문 시점에는 직전 턴(이름 정보) 저장이 완료되지 않았을 수 있으므로
                # 봇이 "너무 귀엽네요!"라고 대답한 직후(Turn 07)에 검색을 시도해 봅니다.
                if "너무 귀엽네요!" in text:
                    print("   >> [중간 점검 1] '강아지 이름' 관련 기억이 검색되는지 확인...")
                    time.sleep(0.5) # 인덱싱 안정화 대기
                    dog_query = "강아지 이름이 뭐예요?"
                    context = await memory.retrieve_relevant_memories(sid, dog_query, top_k=2)
                    print(f"      질문: {dog_query}")
                    print(f"      검색 결과: {context}")

                # 3. 추가 검색 테스트 (아이폰 카메라 관련)
                if "각각 어떤 용도인가요?" in text:
                    print("   >> [중간 점검 2] '아이폰 카메라 렌즈' 관련 기억이 검색되는지 확인...")
                    time.sleep(0.2) 
                    iphone_query = "새로 산 아이폰 카메라 렌즈는 몇 개야?"
                    context = await memory.retrieve_relevant_memories(sid, iphone_query, top_k=2)
                    print(f"      질문: {iphone_query}")
                    print(f"      검색 결과: {context}")

            print("\n" + "="*50)
            print("[최종 검색 테스트]")
            print("="*50)
            
            # 마지막 질문: "강아지 이름이 뭐였고, 무슨 종이었죠? 잊어버렸어요."
            # 실제 서비스에서는 LLM이 긴 문맥에서 핵심 질문만 추출하지만, 
            # 여기서는 벡터 검색 성능 확인을 위해 핵심 질문만 쿼리로 사용합니다.
            last_query = "강아지 이름이 뭐였고, 무슨 종이었죠?"
            print(f"질문(Query): \"{last_query}\"")
            
            # 인덱싱 안정화를 위해 잠시 대기
            time.sleep(1)
            
            # 검색 수행
            results = await memory.retrieve_relevant_memories(sid, last_query, top_k=3)
            
            print("\n검색된 장기 기억 (Top-3):")
            if results:
                for idx, mem in enumerate(results, 1):
                    print(f"{idx}. {mem}")
            else:
                print("검색된 기억이 없습니다.")

        except Exception as e:
            print(f"테스트 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()

    asyncio.run(test())
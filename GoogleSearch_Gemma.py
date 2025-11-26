# GoogleSearch_Gemma.py

import requests
import configparser
import base64
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Optional, Tuple
import logging
import torch
from torch.cuda.amp import autocast
# NOTE: transformers의 비전 의존성(torchvision)로 인한 임포트 실패를 피하기 위해
# AutoProcessor / Gemma3ForConditionalGeneration 임포트를 모듈 로드 시점에 수행하지 않습니다.
# 필요한 경우(독립 테스트 함수 내부) 지연 임포트합니다.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Optional

# LangChain 관련 라이브러리
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

# LangGraph 관련 라이브러리
from langgraph.graph import StateGraph, END

# 로깅 설정
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================
# Helper for OSS-20B pipeline (2025-08-14)
# =============================================================

# === 기존 파이프라인 훅 (있으면 사용) ===
def _call_langgraph_pipeline(query: str, problem_type: str, language: str) -> Optional[dict]:
    """
    기존 코드베이스에 정의된 `search_and_reason_for_complex_problem_langgraph`가 있으면 사용.
    없거나 실패하면 None.
    """
    try:
        fn = globals().get('search_and_reason_for_complex_problem_langgraph')
        if callable(fn):
            return fn(
                query=query,
                problem_type=problem_type,
                additional_context=None,
                max_iterations=1,
                language=language
            )
    except Exception as e:
        import traceback
        logging.warning("LangGraph pipeline failed: %s\n%s", e, traceback.format_exc())
    return None

def compose_context_block(snippet: str, plan: str) -> str:
    """LLM에게 전달하기 좋은 압축 컨텍스트 블록을 구성합니다."""
    snippet = (snippet or "").strip()
    plan = (plan or "").strip()
    parts = []
    if snippet:
        parts.append("Web Findings (condensed):\n" + snippet)
    if plan:
        parts.append("\nPlan/Method:\n" + plan)
    return "\n\n".join(parts) if parts else "No high-confidence findings."

def get_web_context_for_llm(query: str,
                            problem_type: str = "complex_reasoning_problem",
                            language: str = "ko") -> str:
    """
    [ko] gpt-oss-20b의 [[SEARCH: ...]] 요청에 대응하여 간결한 컨텍스트를 리턴.
    [en] Compose a compact, drop-in context for the LLM when it asks for web search.

    성공 시:
        "Web Findings (condensed): ...\n\nPlan/Method: ..."
    실패 시:
        "Web search unavailable."
    """
    try:
        res = _call_langgraph_pipeline(query, problem_type, language)
        if isinstance(res, dict) and res.get("status") == "success":
            snippet = res.get("best_snippet", "") or res.get("snippet", "") or ""
            plan = res.get("best_plan", "") or res.get("plan", "") or ""
            ctx = compose_context_block(snippet, plan)
            logging.info("[get_web_context_for_llm] Composed context (len=%d)", len(ctx))
            return ctx
        elif isinstance(res, dict):
            # 실패이지만 요약 사유가 있을 수 있음
            summary = res.get("reasoning_summary", "") or "No high-confidence findings."
            logging.info("[get_web_context_for_llm] Fallback summary used.")
            return summary
    except Exception as e:
        import traceback
        logging.error("[get_web_context_for_llm] Failed: %s\n%s", e, traceback.format_exc())
    return "Web search unavailable."

# === [[SEARCH: ...]] 패턴 인식 ===
_SEARCH_RE = re.compile(r"\[\[\s*SEARCH\s*:\s*(.*?)\s*\]\]", re.IGNORECASE | re.DOTALL)

def extract_search_request(text: str) -> Optional[str]:
    """
    모델 출력에서 [[SEARCH: ...]] 패턴이 있으면 질의 문자열을 반환합니다.
    없으면 None을 반환합니다.
    """
    if not text:
        return None
    m = _SEARCH_RE.search(text)
    if not m:
        return None
    query = (m.group(1) or "").strip()
    # 과도한 개행/공백 정리
    query = re.sub(r"\s+", " ", query)
    logging.info("[extract_search_request] Detected search query: %s", query)
    return query
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.DEBUG, # 또는 INFO
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()]) # 명시적으로 핸들러 추가

# 1. 테스트
# 모델과 토크나이저 설정
# def load_model_and_tokenizer():
#     global model, tokenizer
#     model = AutoModel.from_pretrained(
#         'openbmb/MiniCPM-V-2_6-int4', 
#         trust_remote_code=True,
#         # **config
#     )

#     tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True)

#     model.eval()    
#     return model, tokenizer

# # 모델과 토크나이저 설정
# def set_model_and_tokenizer(loaded_model=None, loaded_tokenizer=None):
#     global model, tokenizer
#     if loaded_model is None or loaded_tokenizer is None:
#         model, tokenizer = load_model_and_tokenizer()
#     else:
#         model, tokenizer = loaded_model, loaded_tokenizer

# 2. 챗봇 연계
# 전역 변수로 model과 processor 선언
global model, processor
model = None
processor = None

def set_model_and_processor(loaded_model, loaded_processor):
    global model, processor
    model = loaded_model
    processor = loaded_processor

""" LangChain """

# LangChain의 Runnable 인터페이스를 사용하여 기존 Gemma 모델 호출을 래핑하는 함수
def gemma_llm_runner(inputs: dict) -> str:
    """
    LangChain Runnable로 사용하기 위해 기존 Gemma 모델 호출 로직을 감싸는 함수
    입력으로 dict를 받고 (체인 실행 시 RunnablePassthrough.assign을 통해 구성됨),
    이 dict는 'formatted_messages' (List[BaseMessage])와 'llm_params' (dict) 키를 포함해야 함.
    LLM의 응답 문자열을 반환함.
    """
    global model, processor
    if not model or not processor:
        logging.error("GoogleSearch_Gemma (gemma_llm_runner): Model or processor not set.")
        return "MODEL_OR_PROCESSOR_NOT_SET_ERROR"

    # RunnablePassthrough.assign을 통해 전달된 'formatted_messages'와 'llm_params'를 가져옴.
    formatted_messages_lc = inputs.get("formatted_messages") # List[BaseMessage] 형태
    llm_params = inputs.get("llm_params", {}) # max_new_tokens 등

    if not formatted_messages_lc:
        logging.error("GoogleSearch_Gemma (gemma_llm_runner): 'formatted_messages' not found in input dict.")
        return "FORMATTED_MESSAGES_MISSING_ERROR"
    
    # --- 중요: List[BaseMessage]를 List[Dict[str, str]]로 변환 ---
    # processor.apply_chat_template이 기대하는 형식으로 변환.
    # BaseMessage의 'type' 속성을 'role'로 매핑하고, 'content' 속성을 사용.
    conversation_for_processor = []
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
            logging.warning(f"GoogleSearch_Gemma (gemma_llm_runner): Unknown message type {type(msg)}, defaulting role to 'user'.")

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
                logging.warning(f"GoogleSearch_Gemma (gemma_llm_runner): Visual content detected in LangChain message. It will be ignored as this LLM call is text-only. Message content: {msg.content}")
        else:
            logging.error(f"GoogleSearch_Gemma (gemma_llm_runner): Message content is neither string nor list (type: {type(msg.content)}). Skipping message.")
            continue # 처리할 수 없는 메시지는 건너뛰기

        if processed_content: # 빈 콘텐츠는 추가하지 않음
            content_payload = [
                {"type": "text", "text": processed_content}
            ]
            conversation_for_processor.append({"role": role, "content": content_payload})

    try:
        # conversation이 비어있으면 apply_chat_template 오류 발생 가능
        if not conversation_for_processor:
            logging.warning("GoogleSearch_Gemma (gemma_llm_runner): No valid messages to apply chat template.")
            return "NO_VALID_MESSAGES_ERROR"

        tokenized_inputs = processor.apply_chat_template(
            conversation=conversation_for_processor,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        input_len = tokenized_inputs["input_ids"].shape[-1]

        max_new_tokens = llm_params.get("max_new_tokens", 150)
        do_sample_param = llm_params.get("do_sample", False)
        temperature = llm_params.get("temperature", 0.7)
        top_p = llm_params.get("top_p")
        top_k = llm_params.get("top_k")

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
            logging.warning("GoogleSearch_Gemma (gemma_llm_runner): LLM produced an empty response.")
        return decoded_text
    except Exception as e:
        # exc_info=True 대신 logging.exception 사용 또는 예외 객체 직접 전달
        logging.error(f"GoogleSearch_Gemma (gemma_llm_runner): Error during LLM call - {type(e).__name__}: {e}")
        import traceback
        logging.debug(traceback.format_exc()) # 디버그 레벨로 트레이스백 전체 출력
        return f"LLM_CALL_ERROR: {type(e).__name__} - {str(e)}"
    
# LangChain Runnable 객체 생성 (gemma_llm_runner 함수 기반)
# 이 객체는 여러 체인에서 재사용 가능
gemma_runnable = RunnableLambda(gemma_llm_runner)

""" LangChain """

# Error during message handling: local variable 'search_results' referenced before assignment 해결책으로 search_results 전역화
global search_results
search_results = []

def classify_search_type_langchain(search_query: str, language: str = "en") -> str:
    """
    [LangChain 적용 버전]
    주어진 검색 쿼리의 유형을 LLM을 사용해서 분류함.
    - LangChain의 ChatPromptTemplate, custom gemma_runnable, StrOutputParser를 사용

    Args:
        search_query (str): 분류할 검색 쿼리 문자열
        language (str, optional): 검색 쿼리의 언어, 기본값은 'en'

    Returns:
        str: 분류된 검색 유형 (문자열)
            (예: "simple_information_retrieval", "complex_math_problem")
    """
    global model, processor
    if not model or not processor:
        logging.error(f"GoogleSearch_Gemma (classify_search_type_langchain): Model or processor not set for query '{search_query}'.")
        return "simple_information_retrieval" # 모델 미설정 시 기본값 반환

    if language == "ko":
        prompt_text = f"""
        다음 사용자 검색 요청 내용을 분석하여 검색 유형을 분류해주세요:
        "{{search_query}}"

        분류할 수 있는 검색 유형은 다음과 같습니다:
        - "simple_information_retrieval": 간단한 사실, 정의, 최신 정보, 특정 개체에 대한 정보 등 직접적인 정보 검색 요청입니다. (예: "오늘 서울 날씨", "아인슈타인은 누구인가", "대한민국의 수도는?")
        - "complex_math_problem": 수학 공식 적용, 복잡한 계산, 수학적 증명, 특정 수학 이론 검색 등 복잡한 수학 문제 해결과 관련된 검색 요청입니다. (예: "페르마의 마지막 정리 증명 과정", "나비에-스토크스 방정식 풀이")
        - "complex_coding_problem": 특정 프로그래밍 언어의 알고리즘 구현 방법, 코드 디버깅, 라이브러리 사용법, 복잡한 소프트웨어 아키텍처 설계 등 코딩과 관련된 복잡한 문제 해결 검색 요청입니다. (예: "파이썬으로 이미지 인식 AI 만들기", "리액트에서 상태 관리 최적화 방법")
        - "complex_reasoning_problem": 여러 정보를 종합하거나, 특정 현상의 원인을 분석하거나, 미래 결과를 예측하거나, 비교 분석하는 등 깊이 있는 추론이 필요한 검색 요청입니다. (예: "양자컴퓨터가 미래 사회에 미칠 영향 분석", "기후 변화의 주요 원인과 해결 방안 비교")

        위 설명과 예시를 참고하여, 주어진 검색 요청 내용에 가장 적합한 유형 이름 하나만 정확하게 반환해주세요.
        만약 분류가 매우 애매하거나 위 유형에 명확히 속하지 않는다고 판단되면, "simple_information_retrieval"을 반환해주세요.
        다른 추가 설명 없이, 유형 이름 문자열만 반환해야 합니다.
        """
    else: # 영어 프롬프트
        prompt_text = f"""
        Analyze the following user search query content and classify its type:
        "{{search_query}}"

        Possible search type categories are:
        - "simple_information_retrieval": Requests for straightforward factual information, definitions, current events, or information about specific entities. (e.g., "weather in London today", "who is Albert Einstein", "capital of France")
        - "complex_math_problem": Searches related to solving complex mathematical problems, applying formulas, mathematical proofs, or finding specific mathematical theories. (e.g., "proof of Fermat's Last Theorem", "solving Navier-Stokes equations")
        - "complex_coding_problem": Searches for solutions to complex programming tasks, algorithm implementations, code debugging, library usage, or software architecture design. (e.g., "how to build an image recognition AI in Python", "optimizing state management in React")
        - "complex_reasoning_problem": Searches requiring in-depth reasoning, such as analyzing the cause of a phenomenon, predicting future outcomes, synthesizing multiple pieces of information, or comparative analysis. (e.g., "impact of quantum computing on future society", "comparing main causes and solutions for climate change")

        Based on the descriptions and examples above, return only the single most appropriate category name string for the given search query.
        If the classification is very ambiguous or does not clearly fall into any of the above categories, return "simple_information_retrieval".
        You must return only the category name string, with no additional explanation.
        """

    # LangChain ChatPromptTemplate 생성
    prompt_template = ChatPromptTemplate.from_messages([
        ("human", prompt_text)
    ])

    # LangChain 체인 구성 (LCEL - LangChain Expression Language)
    # 1. 입력을 받아 ({"search_query": ...})
    # 2. RunnablePassthrough.assign을 사용하여 gemma_runnable에 필요한 입력을 구성.
    #    - formatted_messages: prompt_template을 사용하여 생성
    #    - llm_params: 직접 정의한 dict
    # 3. 구성된 dict를 gemma_runnable에 전달
    # 4. 결과를 StrOutputParser로 파싱
    chain = (
        RunnablePassthrough.assign(
            formatted_messages=lambda x: prompt_template.invoke({"search_query": x["search_query"]}).to_messages(),
            llm_params=lambda x: { # LLM 생성 파라미터를 여기서 명시적으로 전달
                "max_new_tokens": 40, 
                "do_sample": False, # 분류는 샘플링 없이 일관성 있게
                "temperature": 0.1, # 온도를 낮춰 일관성 강화 (필요시 조정)
                "top_p": None, # do_sample=False이면 무시되므로 None으로 두거나, 1.0으로 명시
                "top_k": None  # do_sample=False이면 무시되므로 None으로 두거나, 50으로 명시
            }
        )
        | gemma_runnable 
        | StrOutputParser()
    )

    search_type = "simple_information_retrieval" # 기본값
    llm_raw_output = "" # LLM의 실제 출력을 저장할 변수
    try:
        # 체인 실행
        response = chain.invoke({
            "search_query": search_query,
            # max_new_tokens, do_sample 등은 chain 내부 llm_params에서 설정되므로 여기서 명시적으로 전달하지 않아도 됨.
            # 하지만 assign 람다 함수에서 x.get()을 사용하므로, 만약 여기서 전달하면 오버라이드 됨.
            # 여기서는 체인 내부에서 설정된 값을 따르도록 함.
        })

        llm_raw_output = response.strip()

        # gemma_llm_runner에서 에러 발생 시 반환되는 접두사 확인
        if llm_raw_output.startswith("MODEL_OR_PROCESSOR_NOT_SET_ERROR") or \
           llm_raw_output.startswith("FORMATTED_MESSAGES_MISSING_ERROR") or \
           llm_raw_output.startswith("LLM_CALL_ERROR") or \
           llm_raw_output.startswith("NO_VALID_MESSAGES_ERROR"):
            logging.error(f"GoogleSearch_Gemma (classify_search_type_langchain): LLM runner returned an error - '{llm_raw_output}'. Query: '{search_query}'")
            # 에러 발생 시 기본값(simple_information_retrieval) 유지
        else:
            valid_types = [
                "simple_information_retrieval",
                "complex_math_problem",
                "complex_coding_problem",
                "complex_reasoning_problem"
            ]
            found_type = next((v_type for v_type in valid_types if v_type in llm_raw_output), None)
            if found_type:
                search_type = found_type
            else:
                logging.warning(f"GoogleSearch_Gemma (classify_search_type_langchain): LLM output '{llm_raw_output}' did not exactly match a valid type. Defaulting. Query: '{search_query}'")
        
        logging.info(f"GoogleSearch_Gemma (classify_search_type_langchain): Classified query '{search_query}' as type: {search_type} (LLM raw: '{llm_raw_output}')")

    except Exception as e:
        logging.error(f"GoogleSearch_Gemma (classify_search_type_langchain): Error during chain execution for query '{search_query}': %s", e)
    
    return search_type

# RAG 시스템 클래스 정의
class RAGSystem:
    def __init__(self, max_context_length=1000, language=None):
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
        self.documents = []
        self.max_context_length = max_context_length
        self.language = language

    def preprocess_text(self, text: str) -> str:
        # 텍스트 전처리 (메뉴, 목차 등을 제거하고 핵심 본문만을 추출)

        # 줄바꿈을 기준으로 텍스트를 분할
        lines = text.split('\n')

        # 의미가 있는 텍스트 라인만 보존
        meaningful_lines = []
        for line in lines:
            # 공백을 제거
            line = line.strip()
            # 짧은 라인, 메뉴 항목(>, :으로 끝나는 텍스트)으로 보이는 라인 제외
            if len(line) > 30 and not line.endswith('>') and ':' not in line:
                meaningful_lines.append(line)

        # 의미 있는 라인들을 다시 하나의 텍스트로 결합
        preprocessed_text = ' '.join(meaningful_lines)

        # 연속된 공백을 하나의 공백으로 대체
        preprocessed_text = re.sub(r'\s+', ' ', preprocessed_text)

        return preprocessed_text

    def add_documents(self, new_documents: List[str]):
        # 빈 문서 체크
        if not new_documents or all(not doc.strip() for doc in new_documents):
            logging.warning("Empty documents provided to RAG system")
            # 기본 문서 추가
            self.documents = ["검색 결과가 없거나 처리할 수 없습니다." if self.language == "ko" else "No search results or unable to process."]
            self.vectors = self.vectorizer.fit_transform(self.documents)
            return

        # 새로운 문서를 전처리 후
        preprocessed_documents = [self.preprocess_text(doc) for doc in new_documents]

        # 전처리된 문서가 비어있는지 확인
        filtered_documents = [doc for doc in preprocessed_documents if doc.strip()]

        if not filtered_documents:
            logging.warning("All documents were empty after preprocessing")
            # 기본 문서 추가
            self.documents = ["전처리 후 모든 문서가 비어있습니다." if self.language == "ko" else "All documents were empty after preprocessing."]
        else:
            self.documents.extend(filtered_documents)

        # stop_words 옵션을 None으로 설정해 모든 단어 포함
        self.vectorizer = TfidfVectorizer(stop_words=None, min_df=1)
        try:
            if self.documents: # 문서가 있어야만 fit_transform 가능
                self.vectors = self.vectorizer.fit_transform(self.documents)
                logging.info(f"Vectorized {len(self.documents)} documents with vocabulary size {len(self.vectorizer.vocabulary_)}")
            else: # 문서가 비어있는 경우 (예: 초기화 직후 또는 add_documents에 빈 리스트 전달)
                logging.warning("RAGSystem: No documents to vectorize.")
                self.vectors = None
        except Exception as e:
            logging.error(f"RAGSystem: Error in vectorization: {e}")
            # 오류 발생 시 기본 문서와 벡터 설정
            self.documents = ["벡터화 오류 발생." if self.language == "ko" else "Vectorization error occurred."]
            self.vectorizer = TfidfVectorizer(stop_words=None, min_df=1) # vectorizer는 초기화
            if self.documents: # 다시 시도
                self.vectors = self.vectorizer.fit_transform(self.documents)
            else:
                self.vectors = None

    def get_relevant_chunks(self, query, n=3) -> List[str]:
        if not self.documents or self.vectors is None or self.vectors.shape[0] == 0: # 벡터가 비어있거나 문서가 없는 경우
            logging.warning("RAGSystem: No documents or vectors available to get relevant chunks.")
            return []
        try:
            # 쿼리와 가장 관련성 높은 청크를 선택
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.vectors)[0]
            top_indices = similarities.argsort()[-n:][::-1]
            return [self.documents[i] for i in top_indices if i < len(self.documents)]
        except Exception as e:
            logging.error(f"RAGSystem: Error getting relevant chunks for query '{query}': {e}")
            return []

    def create_prompt(self, query, relevant_chunks, language="en"):
        # 관련 청크를 사용해서 프롬프트 생성 (환각 최소화 지침 포함)
        context = " ".join(relevant_chunks)
        if language == "ko":
            prompt = (
                f"Query: {query}\n\n"
                f"Context: {context}\n\n"
                "Instructions (very important):\n"
                "- 답변은 반드시 위 Context에 근거한 사실만 단정적으로 서술합니다.\n"
                "- Context에 없는 내용은 추정하지 말고, '문맥에서 근거 확인 불가'로 표시합니다.\n"
                "- 최종 출력은 다음 두 블록으로 구성합니다:\n"
                "  1) 확실: Context에서 직접 확인되는 결론/절차만 간결히 정리\n"
                "  2) 불확실/추가확인 필요: 문맥에 없거나 상충하는 부분(있다면)과 다음 탐색 키워리스트 2-3개\n"
                "- 불필요한 배경지식이나 일반 상식은 넣지 않습니다.\n\n"
                "Answer:"
            )
        else:
            prompt = (
                f"Query: {query}\n\n"
                f"Context: {context}\n\n"
                "Instructions (very important):\n"
                "- Answer only based on the information in the Context.\n"
                "- If information is not in the Context, indicate 'No information found in context'.\n"
                "- The final output should be structured as follows:\n"
                "  1) Certain: Concisely summarize conclusions/procedures directly from the Context\n"
                "  2) Uncertain/Additional Confirmation Needed: Mention any conflicting or missing information (if any) and 2-3 follow-up search keywords\n"
                "- Do not include unnecessary background knowledge or general facts.\n\n"
                "Answer:"
            )

        # 간단한 토큰 카운팅 함수
        def count_tokens(text):
            # 매우 대략적인 추정치. 실제 토큰화는 더 복잡함.
            return len(re.findall(r'\w+', text))

        # 프롬프트가 일정 컨텍스트 길이를 초과하면 컨텍스트를 줄임.
        while count_tokens(prompt) > self.max_context_length and len(context) > 0: # context가 비어있지 않을 때만 줄임
            content_len_to_remove = max(1, int(len(context) * 0.05)) # 최소 1글자는 제거하도록 보장
            context = context[:-content_len_to_remove]  # (컨텍스트의 5%를 제거)
            prompt = f"Query: {query}\n\nContext: {context}\n\nAnswer:"
        return prompt
    

""" === MCP 유사 추론 로직 함수들 (2025.04.03) === """

def evaluate_relevance(problem: str, search_snippet: str, language="en") -> Tuple[bool, str, int]:
    """
    LLM을 사용하여 검색 스니펫이 문제 해결에 관련성이 있는지 평가합니다.
    Returns: (관련성 여부, 설명, 관련성 점수 0-10)
    """
    global model, processor
    if not model or not processor:
        logging.error("GoogleSearch_Gemma (evaluate_relevance): Model or processor not set.")
        return False, "Model not available", 0

    if language == "ko":
        prompt = f"""
        문제: "{problem}"

        검색된 정보 조각: "{search_snippet}"

        이 정보 조각이 위 '문제'를 해결하는 데 직접적으로 관련이 있고 유용한지 평가해주세요.
        1. 관련성 여부 (예/아니오 만 대답)
        2. 이유 (간략히 설명)
        3. 관련성 점수 (0부터 10까지의 정수)

        반드시 다음 형식으로 3줄로 응답해주세요:
        관련성: [예/아니오]
        이유: [설명]
        점수: [숫자]
        """
    else:
        prompt = f"""
        Problem: "{problem}"

        Search Snippet Found: "{search_snippet}"

        Evaluate if this snippet is directly relevant and useful for solving the 'Problem' above.
        1. Is it relevant? (Yes/No only)
        2. Why or why not? (Brief explanation)
        3. Relevance Score (Integer from 0 to 10)

        Respond ONLY in the following 3-line format:
        Relevant: [Yes/No]
        Reason: [Explanation]
        Score: [Number]
        """
    # messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    # inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
    # input_len = inputs["input_ids"].shape[-1]
    lc_chain = (
        RunnablePassthrough.assign(
            formatted_messages=lambda x: ChatPromptTemplate.from_messages([("human", x["prompt_text"])]).invoke({}).to_messages(),
            llm_params=lambda x: {"max_new_tokens": 100, "do_sample": False, "temperature": 0.1} # 온도 낮춰 일관성 확보
        )
        | gemma_runnable
        | StrOutputParser()
    )
    analysis = ""
    try:
        analysis = lc_chain.invoke({"prompt_text": prompt})

        # 결과 파싱
        lines = analysis.split('\n')
        if len(lines) < 3: raise ValueError(f"Incorrect LLM response format for relevance: {analysis}")
        relevant_text = lines[0].split(":", 1)[-1].strip().lower() # [-1] to handle missing colon robustly
        is_relevant = "yes" in relevant_text or "예" in relevant_text
        reason = lines[1].split(":", 1)[-1].strip()
        score_match = re.search(r'\d+', lines[2].split(":", 1)[-1]) # 숫자만 정확히 파싱
        if not score_match: raise ValueError(f"Score not found in LLM response: {lines[2]}")
        score = int(score_match.group())
        score = max(0, min(10, score)) # 점수 범위 보정

        logging.debug(f"Relevance Eval Result: Relevant={is_relevant}, Score={score}, Reason='{reason}' for snippet: '{search_snippet[:50]}...'")
        return is_relevant, reason, score
    except Exception as e:
        logging.error(f"Error parsing relevance evaluation: {e}\nLLM Response:\n{analysis}")
        return False, "Parsing error", 0


def plan_application(problem: str, relevant_snippet: str, language="en") -> str:
    """
    LLM을 사용하여 관련 스니펫을 문제에 적용할 계획을 세웁니다.
    Returns: 적용 계획 문자열.
    """
    global model, processor
    if not model or not processor:
        logging.error("GoogleSearch_Gemma (plan_application): Model or processor not set.")
        return "Error: Model not available for planning."

    if language == "ko":
        prompt = f"""
        문제: "{problem}"

        관련 정보: "{relevant_snippet}"

        주어진 '관련 정보'를 사용하여 '문제'를 해결하기 위한 단계별 계획을 간략하게 설명해주세요. 핵심 단계에 집중하고, 각 단계는 명확하게 작성해주세요.

        해결 계획:
        """
    else:
        prompt = f"""
        Problem: "{problem}"

        Relevant Information: "{relevant_snippet}"

        Briefly outline a step-by-step plan to use the 'Relevant Information' to potentially solve the 'Problem'. Focus on the key steps, making each step clear.

        Solution Plan:
        """
    lc_chain = (
        RunnablePassthrough.assign(
            formatted_messages=lambda x: ChatPromptTemplate.from_messages([("human", x["prompt_text"])]).invoke({}).to_messages(),
            llm_params=lambda x: {"max_new_tokens": 300, "do_sample": True, "temperature": 0.5}
        )
        | gemma_runnable
        | StrOutputParser()
    )
    plan_str = ""
    try:
        plan_str = lc_chain.invoke({"prompt_text": prompt})
        # "해결 계획:" 또는 "Solution Plan:" 같은 머리글 제거
        plan_str = re.sub(r"^(해결 계획:|Solution Plan:)\s*", "", plan_str, flags=re.IGNORECASE).strip()
        logging.debug(f"Generated Plan: {plan_str}")
        return plan_str
    except Exception as e:
        logging.error(f"Error during plan generation: {e}\nLLM Response:\n{plan_str}")
        return f"Error generating plan: {e}"

def evaluate_plan(problem: str, plan: str, language="en") -> Tuple[bool, str, int]:
    """
    LLM을 사용하여 생성된 계획의 성공 가능성을 평가함.
    Returns: (타당성 여부, 비평/개선점, 신뢰도 점수 0-10)
    """
    global model, processor
    if not model or not processor:
        logging.error("GoogleSearch_Gemma (evaluate_plan): Model or processor not set.")
        return False, "Model not available for plan evaluation", 0

    if language == "ko":
        prompt = f"""
        문제: "{problem}"

        제안된 해결 계획:
        "{plan}"

        이 계획이 주어진 '문제'를 해결하는 논리적이고 합리적인 접근 방식인지 평가해주세요.
        1. 계획의 합리성 (예/아니오 만 대답)
        2. 잠재적 문제점 또는 개선점 (간략히 언급, 없다면 "없음")
        3. 계획 성공 신뢰도 점수 (0부터 10까지의 정수)

        반드시 다음 형식으로 3줄로 응답해주세요:
        합리성: [예/아니오]
        개선점: [설명 또는 없음]
        신뢰도: [숫자]
        """
    else:
        prompt = f"""
        Problem: "{problem}"

        Proposed Plan:
        "{plan}"

        Evaluate if this plan is a logical and sound approach to solving the 'Problem'.
        1. Is the plan sound? (Yes/No only)
        2. Potential issues or improvements? (Briefly mention, or "None")
        3. Confidence Score in Plan Success (Integer from 0 to 10)

        Respond ONLY in the following 3-line format:
        Sound: [Yes/No]
        Critique: [Explanation or None]
        Confidence: [Number]
        """
    lc_chain = (
        RunnablePassthrough.assign(
            formatted_messages=lambda x: ChatPromptTemplate.from_messages([("human", x["prompt_text"])]).invoke({}).to_messages(),
            llm_params=lambda x: {"max_new_tokens": 150, "do_sample": False, "temperature": 0.1}
        )
        | gemma_runnable
        | StrOutputParser()
    )

    analysis = ""
    try:
        analysis = lc_chain.invoke({"prompt_text": prompt})
        # 결과 파싱
        lines = analysis.split('\n')
        if len(lines) < 3:
             raise ValueError(f"Incorrect response format from LLM for plan eval: {analysis}")
        sound_text = lines[0].split(":", 1)[-1].strip().lower()
        is_sound = "yes" in sound_text or "예" in sound_text
        critique = lines[1].split(":", 1)[-1].strip()
        confidence_match = re.search(r'\d+', lines[2].split(":", 1)[-1])
        if not confidence_match:
             raise ValueError(f"Confidence score not found in LLM response: {lines[2]}")
        confidence = int(confidence_match.group())
        confidence = max(0, min(10, confidence)) # 점수 범위 보정

        logging.debug(f"Plan Eval: Sound={is_sound}, Confidence={confidence}, Critique='{critique}' for plan: '{plan[:50]}...'")
        return is_sound, critique, confidence
    except Exception as e:
        logging.error(f"Error parsing plan evaluation: {e}\nLLM Response:\n{analysis}")
        return False, f"Parsing error: {e}", 0


# 핵심 추론 실행 함수
# def search_and_reason_for_complex_problem(query: str, problem_type: str, additional_context: Optional[str] = None, max_iterations: int = 2, language="en") -> Optional[str]:
#     """
#     복잡한 문제에 대해 검색, 관련성 평가, 적용 계획, 계획 평가를 수행합니다.
#     Returns: 최종 응답 생성을 위한 프롬프트 문자열 또는 None (오류 시).
#     """
#     global model, processor, search_results
#     if not model or not processor:
#         logging.error("Model or processor not set for search_and_reason_for_complex_problem.")
#         return None

#     logging.info(f"Starting complex search & reasoning for: '{query}' (Type: {problem_type}, Lang: {language})")
#     reasoning_log = []
#     best_snippet = ""
#     best_plan = ""
#     highest_confidence = -1
#     current_search_query = query # 초기 검색어

#     for i in range(max_iterations):
#         reasoning_log.append(f"\n추론 단계 {i+1}:")
#         logging.info(f"Reasoning Iteration {i+1}/{max_iterations}")

#         # 1. 검색 수행 (recursive_search 활용 또는 직접 google_search)
#         reasoning_log.append(f"- 검색 실행: '{current_search_query}'")
#         try:
#             # search_content는 검색 결과 요약 또는 전체 텍스트일 수 있음
#             search_content, _, _ = recursive_search(current_search_query, additional_context, max_iterations=1, language=language)
#             # recursive_search가 실패하거나 내용을 반환하지 못하는 경우 대비
#             if not search_content or not isinstance(search_content, str) or len(search_content) < 10:
#                  logging.warning("Recursive search yielded limited content, trying direct search.")
#                  search_results_list = google_search(current_search_query, num_results=5) # 직접 검색
#                  search_content = "\n\n".join([res.get('snippet', '') for res in search_results_list if res.get('snippet')])
#                  if not search_content:
#                       reasoning_log.append("- 검색 결과 없음.")
#                       continue # 다음 반복 시도 (쿼리 개선 로직이 있다면)
#             logging.debug(f"  - 검색 결과 내용 (일부): {search_content[:200]}...")
#         except Exception as e:
#             logging.error(f"Error during search in iteration {i+1}: {e}")
#             reasoning_log.append(f"- 검색 중 오류 발생: {e}")
#             continue # 다음 반복 시도

#         # 2. RAG로 후보 스니펫 추출
#         rag = RAGSystem(language=language)
#         # 검색 결과가 문자열이면 리스트로 감싸서 전달
#         rag.add_documents([search_content] if isinstance(search_content, str) else search_content)
#         candidate_snippets = rag.get_relevant_chunks(query, n=5) # 상위 5개 후보 추출
#         reasoning_log.append(f"- {len(candidate_snippets)}개의 후보 정보 조각 추출.")
#         if not candidate_snippets:
#             continue

#         # 3. 관련성 평가 (상위 3개 평가)
#         evaluated_snippets = []
#         for snippet in candidate_snippets[:3]:
#             if not snippet or len(snippet.strip()) < 10: continue # 너무 짧으면 스킵
#             is_relevant, reason, score = evaluate_relevance(query, snippet, language)
#             reasoning_log.append(f"  - 정보 조각 평가: 관련성={is_relevant}, 점수={score}, 이유='{reason}', 내용='{snippet[:80]}...'")
#             if is_relevant and score >= 5: # 관련성 임계값 (조정 가능)
#                 evaluated_snippets.append({'text': snippet, 'score': score})

#         if not evaluated_snippets:
#             reasoning_log.append("- 관련성 높은 정보 조각 없음.")
#             # TODO: 여기서 다음 검색 쿼리를 개선하는 로직 추가 가능
#             # 예: LLM에게 "관련 정보 부족 이유({reason}) 기반으로 검색어 '{current_search_query}' 개선 제안" 요청
#             continue

#         # 4. 최고 스니펫 선택 및 계획 수립/평가
#         current_best_snippet_info = max(evaluated_snippets, key=lambda x: x['score'])
#         current_snippet = current_best_snippet_info['text']
#         reasoning_log.append(f"- 최적 정보 조각 선택 (점수: {current_best_snippet_info['score']}): '{current_snippet[:80]}...'")

#         plan = plan_application(query, current_snippet, language)
#         if "Error generating plan" in plan:
#              reasoning_log.append("- 계획 생성 중 오류 발생.")
#              continue
#         reasoning_log.append(f"- 적용 계획 생성:\n{plan}")

#         is_sound, critique, confidence = evaluate_plan(query, plan, language)
#         reasoning_log.append(f"  - 계획 평가: 합리성={is_sound}, 신뢰도={confidence}, 개선점='{critique}'")

#         # 5. 최고 계획 업데이트
#         if is_sound and confidence > highest_confidence:
#             highest_confidence = confidence
#             best_snippet = current_snippet
#             best_plan = plan
#             reasoning_log.append(f"- 최고 계획 갱신 (신뢰도: {highest_confidence}).")
#             # 신뢰도가 매우 높으면 일찍 종료 가능
#             if confidence >= 8:
#                 reasoning_log.append("- 높은 신뢰도의 계획 발견, 추론 종료.")
#                 break

#         # TODO: 계획 평가가 나쁘면(`critique` 활용) 다음 검색 쿼리(`current_search_query`) 개선 로직

#     # 6. 최종 프롬프트 생성 및 반환
#     final_reasoning_summary = "\n".join(reasoning_log)
#     logging.info("Reasoning process finished. Generating final prompt for Raika.")

#     if best_plan and highest_confidence >= 5:
#         # 성공적인 경우, 검색된 핵심 정보, 계획, 추론 과정을 반환
#         logging.info(f"ComplexSearch: Success for query='{query}'. Confidence: {highest_confidence}")
#         return {
#             "status": "success",
#             "query": query, # 원본 문제/쿼리
#             "best_snippet": best_snippet,
#             "best_plan": best_plan,
#             "reasoning_summary": final_reasoning_summary,
#             "confidence": highest_confidence,
#             "language": language
#         }
#     else:
#         # 만족스러운 계획 수립 실패 시
#         logging.warning(f"ComplexSearch: Failure or low confidence for query='{query}'. Highest confidence: {highest_confidence}")
#         return {
#             "status": "failure",
#             "query": query,
#             "reasoning_summary": final_reasoning_summary, # 실패했지만, 시도한 로그는 전달
#             "message": "Could not formulate a confident plan based on search results." if language == "en" \
#                        else "검색 결과를 바탕으로 확신 있는 계획을 세우지 못했습니다.",
#             "language": language
#         }

    # ↑ 신버전 ("1.검색 결과", "2.유저 프롬프트", "3.라이카 초기 답변"을 조합하기 위해서는 순수한 "검색 및 추론의 결과물"이 필요)
    # ↓ 구버전 (LLM에게 넘겨주기 위한 단순 최종 프롬프트)

    # if best_plan and highest_confidence >= 5: # 최종 계획 채택 임계값 (조정 가능)
    #     if language == "ko":
    #         final_prompt = f"""
    #         원래 질문: "{query}"
    #         {f"추가 컨텍스트: '{additional_context}'" if additional_context else ""}

    #         이 질문을 해결하기 위해 내가 생각한 과정은 다음과 같아:
    #         {final_reasoning_summary}

    #         가장 도움이 될 것 같은 정보는 이거야:
    #         "{best_snippet}"

    #         그리고 이 정보를 사용하는 내 계획은 이래:
    #         "{best_plan}"
    #         (이 계획에 대한 내 신뢰도는 {highest_confidence}/10 정도야!)

    #         자, 이제 위의 내 생각 과정, 찾은 정보, 그리고 계획을 바탕으로 원래 질문에 대한 답변을 '라이카'로서 작성해줘! 내 생각 과정을 답변에 자연스럽게 녹여주면 좋겠어. 너무 길게 설명하지 않아도 괜찮아. 필요하다면 단계별로 설명해줘도 좋아!
    #         """
    #     else:
    #         final_prompt = f"""
    #         Original Question: "{query}"
    #         {f"Additional Context: '{additional_context}'" if additional_context else ""}

    #         Here's how I thought about solving this:
    #         {final_reasoning_summary}

    #         The most helpful piece of information I found seems to be:
    #         "{best_snippet}"

    #         And here's my plan to use that information:
    #         "{best_plan}"
    #         (My confidence in this plan is about {highest_confidence}/10!)

    #         Okay, now, using my thought process, the information I found, and the plan above, please formulate the final answer to the original question *as Raika*! Try to weave my reasoning into your response naturally, without being too lengthy. Feel free to explain step-by-step if it makes sense!
    #         """
    # else:
    #     # 만족스러운 계획 수립 실패
    #     if language == "ko":
    #         final_prompt = f"""
    #         원래 질문: "{query}"
    #         {f"추가 컨텍스트: '{additional_context}'" if additional_context else ""}

    #         내가 이 질문을 해결하려고 이렇게 생각해봤어:
    #         {final_reasoning_summary}

    #         *낑낑...* 그런데 찾아낸 정보들로는 확실하게 문제를 해결할 좋은 계획을 세우기가 어려웠어. 😥 정보가 부족하거나 내 계획이 좀 부족했을 수도 있어.
    #         그래서 일단 내가 아는 선에서 최선을 다해 답해볼게! 하지만 완벽하지 않을 수도 있다는 점은 알아줘!
    #         """
    #     else:
    #         final_prompt = f"""
    #         Original Question: "{query}"
    #         {f"Additional Context: '{additional_context}'" if additional_context else ""}

    #         Here was my thinking process for tackling this:
    #         {final_reasoning_summary}

    #         *Whimpers softly...* Unfortunately, I couldn't come up with a really solid plan to solve this using the information I found. 😥 Maybe the info wasn't quite right, or my plan wasn't good enough.
    #         So, I'll give you my best answer based on what I already know! Just be aware it might not be perfect!
    #         """

    # logging.debug(f"Final prompt generated for Raika:\n{final_prompt[:500]}...")
    # return final_prompt

""" === MCP 유사 추론 로직 함수들 (2025.04.03) === """


def generate_search_keywords_langchain(original_query: str, current_query: str, additional_context: Optional[str] = None, language: str = "en", search_history_summary: Optional[str] = None, strict_user_query_only: bool = False) -> str:
    """
    [LangChain 적용 버전]
    검색 결과를 기반으로 (원하는 결과가 온전히 나오도록) 검색 키워드를 생성
    
    Args:
        strict_user_query_only: True일 경우, search_history_summary와 additional_context를 무시하고
                               오로지 original_query만 기반으로 키워드 생성
    """
    
    global model, processor
    if not model or not processor:
        logging.error("GoogleSearch_Gemma (generate_search_keywords_langchain): Model or processor not set.")
        return original_query # 모델 없으면 원본 쿼리 반환 또는 빈 문자열
    
    # strict_user_query_only 모드일 경우 search_history와 additional_context 무시
    if strict_user_query_only:
        logging.info("generate_search_keywords_langchain: strict_user_query_only mode - ignoring search history and additional context")
        search_history_summary = None
        additional_context = None
    
    # 언어별 프롬프트 생성
    if language == "ko":
        prompt_text = f"""
        사용자의 원본 질문: {{original_query}}
        이전 검색 시도에서 사용한 검색어: {{current_query}}
        추가적인 맥락 정보: {{additional_context_str}}
        {f"이전 검색 결과 요약: {{search_history_summary}}" if search_history_summary else ""}

        위 정보를 바탕으로, 사용자의 원본 질문에 더 정확하고 유용한 답변을 찾기 위한 **새로운** 구글 검색 질의 3-4개를 제안하세요.
        
        **핵심 지침 (Query Rewriting Strategy):**
        1. **다국어 확장**: 원본 질문이 한국어라도, (한국 고유의 컨텐츠가 아니라면) 정보량이 많은 **영어(English)** 검색어를 반드시 1~2개 포함하세요. (예: '남미 영화 편지' -> 'South American movie letter writing')
        2. **구체적 묘사**: "가격/정보" 같은 단순 키워드보다는, 질문의 묘사적 특징(description)을 살린 구체적인 구문(phrase)을 사용하세요.
        3. **엔티티 보존**: 고유명사, 연도, 특정 행위 등 핵심 엔티티는 유지하되, 동의어나 유의어로 변형하여 시도하세요.
        4. **형식**: 질의는 쉼표(,)로 구분하고, 각 질의는 간결하게 작성하세요.
        5. 더 이상 개선된 질의가 없다면 "더 이상 좋은 키워드 없음"만 출력하세요.

        새로운 검색 질의:
        """
    else:
        prompt_text = f"""
        Original user request: {{original_query}}
        Search query used in the previous attempt: {{current_query}}
        Additional context: {{additional_context_str}}
        {f"Summary of previous search results: {{search_history_summary}}" if search_history_summary else ""}

        Based on the above, suggest 2-3 **new** Google search queries.
        Guidelines:
        - Preserve salient named entities (brands, model names, places, dates) from the original request.
        - Do not output generic words alone (e.g., price/specs/compare/info); if used, pair them with the salient entities.
        - Make queries domain-agnostic but specific (news, academic, guides, games, books, etc.).
        - Separate by commas, each query 3-8 tokens.
        - If no significantly better queries exist, respond only with "NO_BETTER_KEYWORDS".

        New search queries:
        """

    prompt_template = ChatPromptTemplate.from_messages([
        ("human", prompt_text)
    ])

    chain = (
        RunnablePassthrough.assign(
            formatted_messages=lambda x: prompt_template.invoke({
                "original_query": x["original_query"],
                "current_query": x["current_query"],
                "additional_context_str": x["additional_context_str"],
                "search_history_summary": x.get("search_history_summary") # None일 수 있음
            }).to_messages(),
            llm_params=lambda x: { # LLM 생성 파라미터를 여기서 명시적으로 전달
                "max_new_tokens": x.get("max_new_tokens", 75),
                "do_sample": x.get("do_sample", True),
                "temperature": x.get("temperature", 0.6),
                "top_p": x.get("top_p"),
                "top_k": x.get("top_k")
            }
        )
        | gemma_runnable
        | StrOutputParser()
    )

    keywords_str = original_query # 기본값: LLM 호출 실패 시 원본 쿼리를 키워드로 사용
    llm_raw_output = ""
    try:
        # additional_context가 None일 경우 빈 문자열 또는 "추가 컨텍스트 없음" 등으로 처리
        context_str = additional_context if additional_context else ("추가 컨텍스트 없음" if language == "ko" else "No additional context provided")
        history_summary_str = search_history_summary if search_history_summary else ("이전 검색 기록 없음" if language == "ko" else "No previous search history")

        response = chain.invoke({
            "original_query": original_query,
            "current_query": current_query,
            "additional_context_str": context_str,
            "search_history_summary": history_summary_str,
            "max_new_tokens": 75, # 키워드 생성이므로 적절한 길이
            "do_sample": True,     # 다양한 키워드 생성을 위해 True
            "temperature": 0.6     # 너무 벗어나지 않는 선에서 창의성 부여
        })
        llm_raw_output = response.strip()

        if llm_raw_output.startswith("MODEL_OR_PROCESSOR_NOT_SET_ERROR") or \
           llm_raw_output.startswith("FORMATTED_MESSAGES_MISSING_ERROR") or \
           llm_raw_output.startswith("LLM_CALL_ERROR"):
            logging.error(f"GoogleSearch_Gemma (generate_search_keywords_langchain): LLM runner returned an error - '{llm_raw_output}'. Query: '{original_query}'")
            # 에러 발생 시 기본값(original_query) 유지
        elif not llm_raw_output: # LLM 응답이 비어있는 경우
            logging.warning(f"GoogleSearch_Gemma (generate_search_keywords_langchain): LLM returned empty keywords for query '{original_query}'. Using original query as fallback.")
        else:
            # [25.11.26 파싱 강화] LLM 출력이 번호 목록이나 개행으로 구분될 경우 처리
            # 예: "1. kw1\n2. kw2" -> "kw1, kw2"
            cleaned_lines = []
            for line in llm_raw_output.split('\n'):
                line = line.strip()
                if not line: continue
                # 번호(1., 1)) 및 불렛(-, *) 제거
                line = re.sub(r'^[\d]+[\.\)]\s*|^[\-\*]\s*', '', line)
                # 앞뒤 따옴표 제거
                line = line.strip('"\'')
                if line:
                    cleaned_lines.append(line)
            
            # 줄바꿈으로 분리된 항목들을 콤마로 연결
            joined_text = ",".join(cleaned_lines)
            
            # 콤마로 재분리하여 깔끔한 리스트 생성
            final_kws = [k.strip() for k in joined_text.split(',') if k.strip()]
            
            if final_kws:
                keywords_str = ", ".join(final_kws)
            else:
                keywords_str = llm_raw_output # 파싱 실패 시 원본 사용 (fallback)
        
        logging.info(f"GoogleSearch_Gemma (generate_search_keywords_langchain): Generated keywords '{keywords_str}' for query '{original_query}' (LLM raw: '{llm_raw_output}')")
    except Exception as e:
        logging.error(f"GoogleSearch_Gemma (generate_search_keywords_langchain): Error generating keywords for query '{original_query}': %s", e)

    return keywords_str

def recursive_search(initial_query: str, additional_context: Optional[str] = None, max_iterations: int = 3, language="en", *, user_query: Optional[str] = None, user_info_uncertain: bool = False) -> tuple:
    """
    재귀적 검색 수행 함수
    
    Args:
        user_info_uncertain: True일 경우, 오로지 사용자의 원본 질문만 기반으로 키워드 생성.
                            이전 검색 결과나 컨텍스트를 키워드 생성에 포함하지 않음.
    """
    global model, processor

    user_query_for_prompt = (user_query or additional_context or initial_query or "").strip()
    current_query = initial_query
    best_result_content = "" # 가장 만족스러운 검색 결과 콘텐츠를 저장
    all_results_history = [] # 모든 반복에서의 검색 결과를 기록함
    search_context_accumulated = [] # 이전 검색 결과 컨텍스트 누적
    has_any_search_snippet = False

    # 모델/프로세서가 준비되지 않았을 때는 외부 LLM 평가 단계를 건너뛰고
    # 순수 구글 스니펫 기반 요약만 반환하도록 안전 폴백
    model_ready = model is not None and processor is not None

    if user_info_uncertain:
        logging.info(f"[Recursive Search] user_info_uncertain=True - 키워드는 오로지 사용자 질문만 기반으로 생성됩니다.")

    for i in range(max_iterations):
        logging.info(f"[Recursive Search] Iteration {i+1}/{max_iterations} - Current Query: '{current_query}'")
        
        # 1. 검색 키워드 생성
        # user_info_uncertain이 True이면 search_history_summary를 사용하지 않음
        search_history_summary = None
        if not user_info_uncertain:
            search_history_summary = "\n".join(all_results_history[-2:]) # 최근 2개 검색 결과 요약
            if not search_history_summary:
                search_history_summary = None
        
        search_keywords_str = generate_search_keywords_langchain(
            user_query_for_prompt or initial_query, 
            current_query, 
            additional_context, 
            language,
            search_history_summary,
            strict_user_query_only=user_info_uncertain  # user_info_uncertain일 때 strict 모드 활성화
        )

        if search_keywords_str == "NO_BETTER_KEYWORDS" or not search_keywords_str.strip():
            logging.info(f"[Recursive Search] No better keywords generated. Ending search loop.")
            break # 더 이상 개선된 키워드가 없으면 종료

        current_search_terms = [kw.strip() for kw in search_keywords_str.split(',') if kw.strip()]
        if not current_search_terms: # 키워드가 파싱되지 않았을 때
            logging.warning(f"[Recursive Search] Keyword generation yielded no usable terms. Using last query for search.")
            current_search_terms = [current_query] # 최소한 현재 쿼리로라도 시도

        # 2. 생성된 키워드로 구글 검색 수행
        snippets_this_iteration = []
        for term in current_search_terms:
            logging.debug(f"[Recursive Search] Searching Google for term: '{term}'")
            raw_search_results = google_search(term, num_results=3) # 각 키워드당 3개 결과
            # 스니펫에 출처(도메인)와 제목을 함께 포함하여 근거 가시성을 높임
            try:
                from urllib.parse import urlparse
            except Exception:
                urlparse = None
            for res in raw_search_results:
                snippet = res.get('snippet', '')
                if not snippet:
                    continue
                title = res.get('title', '')
                link = res.get('link', '')
                domain = ''
                if link and urlparse:
                    try:
                        domain = urlparse(link).netloc
                    except Exception:
                        domain = ''
                prefix = f"[{domain}] " if domain else ''
                title_part = f"{title} — " if title else ''
                enriched = f"{prefix}{title_part}{snippet}"
                snippets_this_iteration.append(enriched)

        combined_snippets_this_iteration = "\n\n".join([s for s in snippets_this_iteration if s.strip()])
        if not combined_snippets_this_iteration:
            logging.warning(f"[Recursive Search] No useful snippets found for terms: {current_search_terms}")
            all_results_history.append(f"No results for '{', '.join(current_search_terms)}'.")
            if i == max_iterations - 1: # 마지막 시도인데 결과가 없으면
                logging.info(f"[Recursive Search] Max iterations reached without satisfactory results.")
                if language == "ko":
                    best_result_content = "검색 결과를 찾을 수 없거나 관련 정보가 부족합니다."
                else:
                    best_result_content = "No search results found or insufficient relevant information."
            continue # 다음 반복으로

        # 검색 결과를 하나의 문자열로 결합
        all_results_history.append(f"Results for '{', '.join(current_search_terms)}':\n{combined_snippets_this_iteration}")
        search_context_accumulated.append(combined_snippets_this_iteration)
        has_any_search_snippet = True

        # 3. LLM을 사용하여 검색 결과 평가 및 다음 쿼리 제안 (재귀적 개선)
        # 이전 컨텍스트와 현재 결과를 모두 고려
        full_context_for_evaluation = (
            f"Original user query: {user_query_for_prompt or initial_query}\n"
            f"Latest search query: {current_query}\n"
            f"Additional context: {additional_context or 'None'}\n\n"
            "All search results so far:\n" + "\n\n".join(search_context_accumulated)
        )

        if language == "ko":
            eval_prompt = f"""
            원본 사용자 요청: "{user_query_for_prompt or initial_query}"
            추가 컨텍스트: {additional_context or "추가 컨텍스트 없음"}
            현재까지의 검색 결과 종합:
            ---
            {full_context_for_evaluation}
            ---
            
            이 결과들이 원본 요청에 답하기에 충분한지 평가하세요. 충분하면 "만족 (SATISFACTORY)"만 출력합니다.
            충분하지 않다면, 원본 질의의 의미를 보존하면서도 더 잘 답을 찾을 수 있는 **새로운 검색 질의** 2-3개를 제안합니다.
            - 도메인 무관(뉴스/학문/가이드/게임/서적 등)하게 적용 가능한 구체성 유지
            - 일반어만 단독으로 쓰지 말고, 의미상 핵심어와 결합
            - 쉼표로 구분
            응답 형식: "부족한 점: [내용], 새 쿼리: [질의1, 질의2]" 또는 "만족 (SATISFACTORY)"
            """
        else:
            eval_prompt = f"""
            Original user request: "{user_query_for_prompt or initial_query}"
            Additional context: {additional_context or "No additional context provided"}
            Accumulated search results so far:
            ---
            {full_context_for_evaluation}
            ---
            
            Decide sufficiency. If sufficient, return 'SATISFACTORY' only. If not, propose 2-3 new search queries that preserve the meaning and increase specificity, domain-agnostic. Separate with commas.
            Format: "Lacking: [description], New Query: [query1, query2]" or "SATISFACTORY".
            """

        # 모델이 준비되지 않은 경우: 스니펫 길이/질문어 포함 여부로 간단 평가하여 계속/종료 결정
        if not model_ready:
            long_enough = len(combined_snippets_this_iteration) > 400
            lowered_query = (user_query_for_prompt or initial_query or "").lower()
            has_question_words = any(w in lowered_query for w in ["what", "who", "why", "how", "when", "where"]) or any(w in (user_query_for_prompt or initial_query or "") for w in ["왜", "무엇", "어디", "언제", "어떻게"])
            if long_enough and has_question_words:
                eval_analysis = "SATISFACTORY"
            else:
                # 다음 반복을 위한 간단 새 키워드 제안 형식 유지
                # LLM 미준비 시에도, 원본 질의의 핵심 어휘를 유지한 간단한 재질의를 구성
                # 너무 일반적인 토큰만 남지 않도록 트리밍
                base = (user_query_for_prompt or initial_query or "")
                base = base if len(base) <= 80 else base[:80]
                fallback_terms = ", ".join([base] + current_search_terms[:1]) if current_search_terms else base
                eval_analysis = f"Lacking: heuristic, New Query: {fallback_terms}"
        else:
            eval_messages = [{"role": "user", "content": [{"type": "text", "text": eval_prompt}]}]
            eval_inputs = processor.apply_chat_template(eval_messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
            eval_input_len = eval_inputs["input_ids"].shape[-1]

            # 모델 추론 수행
            with torch.inference_mode():
                eval_generation = model.generate(**eval_inputs, max_new_tokens=200, do_sample=True, temperature=0.5)
                eval_generation = eval_generation[0][eval_input_len:]
            eval_analysis = processor.decode(eval_generation, skip_special_tokens=True).strip()
        logging.info(f"[Recursive Search] Evaluation for iteration {i+1}: '{eval_analysis}'")

        if "SATISFACTORY" in eval_analysis.upper() or "만족" in eval_analysis:
            best_result_content = "\n\n".join(search_context_accumulated).strip()
            logging.info(f"[Recursive Search] Search deemed SATISFACTORY after {i+1} iterations.")
            return best_result_content, True, i + 1
        else:
            # 새로운 쿼리 추출
            new_query_match = re.search(r'(새 쿼리|New Query):\s*([^,]+(?:,\s*[^,]+)*)', eval_analysis, re.IGNORECASE)
            if new_query_match:
                new_keywords_str = new_query_match.group(2).strip()
                current_query = new_keywords_str # 다음 반복을 위한 새 쿼리
                if not current_query.strip(): # 새 쿼리가 비어있으면 종료
                    logging.warning(f"[Recursive Search] Extracted new query is empty. Ending search.")
                    break
            else:
                logging.warning(f"[Recursive Search] Could not extract new query from evaluation. Ending search.")
                break # 새 쿼리를 추출할 수 없으면 종료

    # 최대 반복 횟수를 채웠거나, 더 이상 개선될 쿼리가 없는 경우
    final_content_for_response = "\n\n".join(search_context_accumulated).strip()
    
    if not has_any_search_snippet:
        logging.info(f"[Recursive Search] No web snippets gathered after {max_iterations} iterations.")
    else:
        logging.info(f"[Recursive Search] Max iterations reached or no further refinement. Returning best available content.")
    return final_content_for_response, False, max_iterations

# Google cse api 사용 (Google Custom Search Engine) https://programmablesearchengine.google.com
def google_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:

    # 설정 파일 로드 (절대 경로 사용으로 FastAPI 모듈 import 시에도 정상 작동)
    try:
        import os
        config = configparser.ConfigParser()
        # 이 스크립트의 디렉토리 기준으로 config.ini 경로 설정
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
        try:
            config.read(config_path, encoding='utf-8')
        except Exception:
            config.read(config_path)
        
        google_api_key = config['DEFAULT']['google_api_key'].strip()
        cx = config['DEFAULT']['cx'].strip()
        
        # API 키 유효성 검증
        if not google_api_key:
            raise ValueError("google_api_key is empty or not set in config.ini")
        if not cx:
            raise ValueError("cx (Custom Search Engine ID) is empty or not set in config.ini")
            
        logging.debug(f"Config loaded from: {config_path}, API key length: {len(google_api_key)}, CX: {cx[:10]}...")
    except Exception as e:
        logging.error(f"Failed to load API configuration: {e}")
        # 에러 발생 시 빈 결과 반환
        return []

    # 한글 쿼리는 URL 인코딩을 명시적으로 처리
    import urllib.parse
    encoded_query = urllib.parse.quote(query)

    # 한글 검색을 위한 추가 매개변수 설정
    has_korean = any('\uAC00' <= char <= '\uD7A3' for char in query)

   # 한글 검색 최적화 매개변수 (한글이 감지된 경우)
    if has_korean:
        url = f"https://www.googleapis.com/customsearch/v1?q={encoded_query}&key={google_api_key}&cx={cx}&hl=ko&lr=lang_ko&num={num_results}"
        logging.info(f"Korean query detected: {query}")
    else:
        url = f"https://www.googleapis.com/customsearch/v1?q={encoded_query}&key={google_api_key}&cx={cx}&num={num_results}"
    
    # url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={google_api_key}&cx={cx}"
    
    # 로깅에서 API 키 마스킹
    log_url = url.replace(google_api_key, "API_KEY_MASKED")
    logging.info(f"Requesting Google API: {log_url}")

    try:
        response = requests.get(url)
        response.raise_for_status() # HTTP 오류 발생 시 예외를 발생시킴
        search_results = response.json()
    except requests.RequestException as e:
        logging.error(f"Error occurred while requesting Google API: {e}")
        return []
    
    logging.debug(f"Google API response: {search_results}")

    # 각 검색 결과에서 관련 정보 추출
    results = []
    if 'items' in search_results:
        for item in search_results['items']:
            result = {
                'title': item.get('title', ''),
                'link': item.get('link', ''),
                'snippet': item.get('snippet', '')
            }

            try:
                # 텍스트에 한글이 포함되어 있는지 확인
                has_korean = any('\uAC00' <= char <= '\uD7A3' for char in result['snippet'])
                if has_korean:
                    logging.debug(f"Korean text detected in snippet: {result['snippet'][:30]}...")
                else:
                    logging.debug(f"Non-Korean text in snippet: {result['snippet'][:30]}...")
            except Exception as e:
                logging.warning(f"Error checking text language: {e}")

            # Google API의 경우 디코딩 불필요함.

            # # Base64로 인코딩된 필드가 있다면
            # if 'snippet' in item:
            #     encoded_text = item['snippet']
            #     try:
            #         # Base64 디코딩
            #         decoded_text = base64.b64decode(encoded_text).decode('utf-8')
            #         result['snippet'] = decoded_text
            #     except Exception as e:
            #         # Base64 디코딩 실패 시 원본 텍스트 반환
            #         print(f"Base64 decoding fail: {e}")

            results.append(result)
    else:
        logging.warning("No 'items' key in search results. Available keys: " + 
                        str(list(search_results.keys())))
        # 에러 대신 기본 응답을 생성
        results = [{
            'title': '검색 결과 없음',
            'link': '',
            'snippet': '이 검색어에 대한 결과를 찾을 수 없습니다. 일일 할당량이 초과되었거나 검색어가 너무 모호할 수 있습니다.'
        }]

    logging.debug(f"Search results: {results}")
    return results


import logging as log

# === 기존 파이프라인 훅 (있으면 사용) ===
def _call_langgraph_pipeline(query: str, problem_type: str, language: str) -> Optional[dict]:
    """
    기존 코드베이스에 정의된 `search_and_reason_for_complex_problem_langgraph`가 있으면 사용.
    없거나 실패하면 None.
    """
    try:
        fn = globals().get('search_and_reason_for_complex_problem_langgraph')
        if callable(fn):
            return fn(
                query=query,
                problem_type=problem_type,
                additional_context=None,
                max_iterations=1,
                language=language
            )
    except Exception as e:
        import traceback
        log.warning("LangGraph pipeline failed: %s\n%s", e, traceback.format_exc())
    return None




# 주어진 URL에서 주요 텍스트 내용을 추출
def extract_content_from_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=5)

        # HTML 내용 파싱
        soup = BeautifulSoup(response.content, 'html.parser')

        # script와 style 요소를 제거
        for script in soup(["script", "style"]):
            script.decompose()

        # 파싱된 HTML에서 텍스트 추출
        text = soup.get_text()

        # 추출된 텍스트 정리
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        logging.debug(f"Length of text extracted from URL {url}: {len(text)}")
        return text
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return ""

def get_relevant_information(query: str) -> List[Dict[str, str]]:

    global search_results

    # Google 검색 수행
    search_results = google_search(query)
    relevant_info = [] # 관련 정보를 포함하는 딕셔너리 리스트

    # 각 검색 결과에서 내용을 추출
    for result in search_results:
        content = extract_content_from_url(result['link'])
        relevant_info.append({
            'title': result['title'],
            'content': content[:1000],
            'url': result['link'],
            'snippet': result.get('snippet', '') # 디코딩된 스니펫 포함
        })

    logging.debug(f"Number of relevant information extracted for query '{query}': {len(relevant_info)}")
    return relevant_info

def process_with_rag(query: str, additional_context: Optional[str] = None, max_context_length: int = 1000, language=None) -> str:
    """RAG 시스템을 사용, 관련 정보를 처리하고 프롬프트를 생성"""

    # 언어 자동 감지 (언어가 지정되지 않은 경우)
    if language is None:
        # Raika_Gemma_FastAPI의 detect_language 함수 사용
        from Raika_Gemma_FastAPI import detect_language
        language = detect_language(query)

    try:
        # RAG 시스템 초기화
        rag = RAGSystem(max_context_length=max_context_length, language=language)

        # 재귀적 검색 수행
        relevant_info, is_satisfactory, iterations = recursive_search(
            query,
            additional_context or "",
            language=language,
            user_query=query
        )

        # 빈 결과 체크
        if not relevant_info.strip():
            if language == "ko":
                return "검색 결과를 찾을 수 없습니다. 내부 지식을 기반으로 응답하겠습니다."
            else:
                return "No search results found. I'll respond based on my internal knowledge."

        # RAG 시스템에 문서를 추가
        rag.add_documents([relevant_info])

        # 가장 관련성 높은 청크 가져오기
        relevant_chunks = rag.get_relevant_chunks(query)

        # 관련 청크가 비어있는지 확인
        if not relevant_chunks or all(not chunk.strip() for chunk in relevant_chunks):
            if language == "ko":
                return "검색 결과에서 관련 정보를 추출할 수 없습니다. 내부 지식을 기반으로 응답하겠습니다."
            else:
                return "Could not extract relevant information from search results. I'll respond based on my internal knowledge."

        # 언어별 프롬프트 생성
        if is_satisfactory:
            prompt = rag.create_prompt(query, relevant_chunks, language=language)
            
            # 라이카 캐릭터 유지를 위한 안내 추가 (한영 모두)
            if language == "ko":
                prompt += "\n(우선순위: 늑대개 라이카 캐릭터를 유지하는 것이 가장 중요합니다. 사실 정보를 제공하거나 RAG 시스템을 사용할 때에도 항상 캐릭터를 유지하세요.)"
            else:
                prompt += "\n(Priority: Maintaining Raika's wolfdog character is the highest priority. Even when providing factual information or using RAG systems, always stay in character.)"
        else:
            if language == "ko":
                # 한국어 프롬프트
                additional_context_prompt = f" 추가 컨텍스트: '{additional_context}'" if additional_context else ""
                prompt = f"""
                {iterations}번의 검색 결과 개선 시도 후에도, 쿼리에 대한 만족스러운 정보를 찾지 못했습니다: "{query}"{additional_context_prompt}. 
                하지만 찾은 가장 좋은 결과는 다음과 같습니다:

                {rag.create_prompt(query, relevant_chunks, language=language)}

                이 결과{' 및 추가 컨텍스트' if additional_context else ''}를 바탕으로 가능한 가장 관련성 높은 정보를 제공해주세요. 그리고 정보가 완전하거나 정확하지 않을 수 있음을 언급해주세요.
                """
                prompt += "\n(우선순위: 늑대개 라이카 캐릭터를 유지하는 것이 가장 중요합니다. 사실 정보를 제공하거나 RAG 시스템을 사용할 때에도 항상 캐릭터를 유지하세요.)"
            else:
                # 영어 프롬프트 (기존)
                additional_context_prompt = f" with additional context: '{additional_context}'" if additional_context else ""
                prompt = f"""
                After {iterations} attempts to improve the search results, we couldn't find fully satisfactory information for the query: "{query}"{additional_context_prompt}. 
                However, here are the best results we found:

                {rag.create_prompt(query, relevant_chunks, language=language)}

                Please provide the most relevant information possible based on these results{' and the additional context' if additional_context else ''}, and acknowledge that the information might not be complete or fully accurate.
                """ 
                prompt += "\n(Priority: Maintaining Raika's wolfdog character is the highest priority. Even when providing factual information or using RAG systems, always stay in character.)"

        return prompt
    
    except Exception as e:
        logging.error(f"Error in RAG processing: {str(e)}")
        if language == "ko":
            return f"검색 처리 중 오류가 발생했습니다: {str(e)}. 내부 지식을 기반으로 응답하겠습니다."
        else:
            return f"An error occurred during search processing: {str(e)}. I'll respond based on my internal knowledge."

# 프로세서와 모델 로드를 위한 함수 (독립 실행 시 사용)
def load_model_for_testing():
    global model, processor
    
    # model_id = "google/gemma-3-4b-it"
    model_id = "unsloth/gemma-3-12b-it-bnb-4bit"
    # 지연 임포트: torchvision 비의존 경로로만 사용
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_id)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    ).eval()
    
    print("Model and processor loaded successfully for testing.")
    return model, processor


# TODO: (25.05.18) LangGraph의 온전한 구현

from typing import TypedDict

# """ --- LangGraph를 위한 상태 정의 (25.05.18) --- """
class ComplexSearchGraphState(TypedDict):
    # 필수 입력 값
    original_query: str
    problem_type: str # e.g., "complex_math_problem"
    language: str
    max_iterations: int # LangGraph 루프의 최대 반복 횟수

    # 선택적 입력 값
    additional_context: Optional[str]
    user_info_uncertain: Optional[bool] # True이면 오로지 사용자 질문만 기반으로 키워드 생성

    # 그래프 실행 중 업데이트되는 값
    current_search_query: str # 현재 검색에 사용할 쿼리
    iteration_count: int # 현재 반복 횟수
    reasoning_log: List[str] # 추론 과정을 기록하는 로그
    search_results_snippets: List[str] # 현재 반복에서의 검색 결과 스니펫들
    relevant_snippets_evaluation: List[dict] # [{'text': snippet, 'score': score, 'reason': reason}]
    current_best_snippet_from_iteration: Optional[dict] # 현재 반복에서 가장 좋은 스니펫 정보
    current_plan: Optional[str] # 현재 스니펫 기반으로 생성된 계획
    current_plan_evaluation: Optional[dict] # {'is_sound': bool, 'critique': str, 'confidence': int}

    # 전체 실행 중 가장 좋았던 결과
    best_overall_snippet_text: Optional[str]
    best_overall_plan_text: Optional[str]
    highest_overall_confidence: int

    # 최종 출력
    final_output_for_raika: Optional[Dict[str, any]] # Raika 에이전트에게 전달될 최종 결과


# """ --- LangGraph 노드 함수 정의 (25.05.18) --- """

def node_initialize_graph(state: ComplexSearchGraphState) -> ComplexSearchGraphState:
    """그래프 실행 시작 시 상태를 초기화하는 노드"""
    logging.info(f"[Graph] Initializing state for query: '{state['original_query']}'")
    return {
        **state, # 입력으로 들어온 값 유지
        "current_search_query": state["original_query"], # 첫 검색은 원본 쿼리로 시작
        "iteration_count": 0,
        "reasoning_log": [f"LangGraph 복합 검색 시작: '{state['original_query']}' (유형: {state['problem_type']}, 언어: {state['language']})"],
        "search_results_snippets": [],
        "relevant_snippets_evaluation": [],
        "current_best_snippet_from_iteration": None,
        "current_plan": None,
        "current_plan_evaluation": None,
        "best_overall_snippet_text": None,
        "best_overall_plan_text": None,
        "highest_overall_confidence": -1, # -1로 초기화하여 어떤 유효한 점수든 더 높게 처리
        "final_output_for_raika": None
    }

def node_perform_search(state: ComplexSearchGraphState) -> ComplexSearchGraphState:
    """현재 쿼리로 구글 검색을 수행하고 스니펫을 추출하는 노드"""
    log = state["reasoning_log"]
    current_query = state["current_search_query"]
    lang = state["language"]
    iter_count = state["iteration_count"]

    log.append(f"\n[반복 {iter_count+1}] 검색 수행: '{current_query}'")
    logging.info(f"[Graph][Iter {iter_count+1}] Performing search for: '{current_query}'")

    # recursive_search 또는 Google Search 직접 사용 가능.
    # 여기서는 Google Search를 직접 사용하여 스니펫 리스트를 얻고, 다음 노드에서 RAG/평가 수행
    search_results_items = google_search(current_query, num_results=5) # 5개 결과 요청
    snippets = [item.get('snippet', '') for item in search_results_items if item.get('snippet')]

    if not snippets:
        log.append("- 검색 결과에서 스니펫을 찾을 수 없습니다.")
        logging.warning(f"[Graph][Iter {iter_count+1}] No snippets found for query '{current_query}'.")
    else:
        log.append(f"- {len(snippets)}개의 스니펫 찾음.")
        logging.debug(f"[Graph][Iter {iter_count+1}] Found snippets: {[s[:50] + '...' for s in snippets]}")

    return {**state, "search_results_snippets": snippets}

def node_evaluate_snippets(state: ComplexSearchGraphState) -> ComplexSearchGraphState:
    """검색된 스니펫들의 관련성을 평가하는 노드"""
    log = state["reasoning_log"]
    snippets = state["search_results_snippets"]
    original_query = state["original_query"]
    lang = state["language"]
    iter_count = state["iteration_count"]

    log.append(f"[반복 {iter_count+1}] 스니펫 관련성 평가 중...")
    logging.info(f"[Graph][Iter {iter_count+1}] Evaluation {len(snippets)} snippets...")

    evaluated_this_iteration = []
    if not snippets:
        log.append("- 평가할 스니펫이 없습니다.")
        return {**state, "relevant_snippets_evaluation": [], "current_best_snippet_from_iteration": None}
    
    for snippet in snippets[:3]: # 시간 효율상 최대 3개 스니펫 평가
        if not snippet.strip():
            continue
        is_relevant, reason, score = evaluate_relevance(original_query, snippet, lang)
        log.append(f"   - 스니펫 평가: 관련성={is_relevant}, 점수={score}, 이유='{reason}', 내용='{snippet[:60].replace(chr(10), ' ')}...'")
        if is_relevant and score >= 5: # 관련성 임계값
            evaluated_this_iteration.append({'text': snippet, 'score': score, 'reason': reason})

    # 현재 반복에서 가장 점수가 높은 스니펫 선택
    current_best_snippet = None
    if evaluated_this_iteration:
        evaluated_this_iteration.sort(key=lambda x: x['score'], reverse=True)
        current_best_snippet = evaluated_this_iteration[0] # 가장 점수가 높은 것
        log.append(f"- 이번 반복의 최적 스니펫 (점수: {current_best_snippet['score']}): '{current_best_snippet['text'][:60].replace(chr(10), ' ')}...'")
        logging.info(f"[Graph][Iter {iter_count+1}] Best snippet from this iteration (Score: {current_best_snippet['score']}).")
    else:
        log.append("- 이번 반복에서 유의미한 관련 스니펫을 찾지 못했습니다.")
        logging.warning(f"[Graph][Iter {iter_count+1}] No significantly relevant snippets found in this iteration.")

    return {**state, "relevant_snippets_evaluation": evaluated_this_iteration, "current_best_snippet_from_iteration": current_best_snippet}

def node_generate_and_evaluate_plan(state: ComplexSearchGraphState) -> ComplexSearchGraphState:
    """최적 스니펫을 기반으로 계획을 생성하고 평가하는 노드"""
    log = state["reasoning_log"]
    best_snippet_info = state["current_best_snippet_from_iteration"]
    original_query = state["original_query"]
    lang = state["language"]
    iter_count = state["iteration_count"]

    current_plan_text = None
    current_plan_eval = None

    if not best_snippet_info or not best_snippet_info.get('text'):
        log.append(f"[반복 {iter_count+1}] 계획 생성을 위한 스니펫이 없습니다.")
        logging.warning(f"[Graph][Iter {iter_count+1}] No snippet available for plan generation.")
        current_plan_eval = {'is_sound': False, 'critique': "정보 부족으로 계획 생성 불가", 'confidence': 0}
    else:
        snippet_text = best_snippet_info['text']
        log.append(f"[반복 {iter_count+1}] 스니펫 기반 계획 생성 중: '{snippet_text[:60].replace(chr(10), ' ')}...'")
        logging.info(f"[Graph][Iter {iter_count+1}] Generating plan based on snippet (Score: {best_snippet_info['score']}).")

        plan_text = plan_application(original_query, snippet_text, lang)
        if "Error generating plan" in plan_text:
            log.append(f"- 계획 생성 중 오류: {plan_text}")
            current_plan_text = None
            current_plan_eval = {'is_sound': False, 'critique': f"계획 생성 오류: {plan_text}", 'confidence': 0}
        else:
            current_plan_text = plan_text
            log.append(f"- 생성된 계획:\n{plan_text}")
            logging.debug(f"[Graph][Iter {iter_count+1}] Generated plan:\n{plan_text}")

            # 계획 평가
            log.append(" - 계획 평가 중...")
            is_sound, critique, confidence = evaluate_plan(original_query, plan_text, lang)
            current_plan_eval = {'is_sound': is_sound, 'critique': critique, 'confidence': confidence}
            log.append(f" - 계획 평가 결과: 합리성={is_sound}, 신뢰도={confidence}, 개선점='{critique}'")
            logging.info(f"[Graph][Iter {iter_count+1}] Plan evaluation: Sound={is_sound}, Confidence={confidence}, Critique='{critique}'.")
    
    return {**state, "current_plan": current_plan_text, "current_plan_evaluation": current_plan_eval}

def node_update_overall_best(state: ComplexSearchGraphState) -> ComplexSearchGraphState:
    """현재 반복의 계획 평가 결과를 바탕으로 전체 최적해를 업데이트하는 노드"""
    log = state["reasoning_log"]
    current_eval = state["current_plan_evaluation"]
    current_plan = state["current_plan"]
    current_snippet_info = state["current_best_snippet_from_iteration"]
    iter_count = state["iteration_count"]

    highest_overall_confidence = state.get("highest_overall_confidence", -1)
    best_overall_plan = state.get("best_overall_plan_text")
    best_overall_snippet = state.get("best_overall_snippet_text")

    if current_eval and current_eval.get('is_sound') and current_eval.get('confidence', 0) > highest_overall_confidence:
        new_highest_confidence = current_eval['confidence']
        log.append(f"[반복 {iter_count+1}] 새로운 최적의 계획을 발견함! 신뢰도: {new_highest_confidence}, 이전 최고 신뢰도: {highest_overall_confidence})")
        logging.info(f"[Graph][Iter {iter_count+1}] New best solution found with confidence {new_highest_confidence}.")
        return {
            **state,
            "highest_overall_confidence": new_highest_confidence,
            "best_overall_plan_text": current_plan,
            "best_overall_snippet_text": current_snippet_info.get('text') if current_snippet_info else None
        }
    else:
        if current_eval:
            log.append(f"[반복 {iter_count+1}] 이번 반복의 계획은 기존 최적 계획을 넘어서지 못했습니다 (현재 신뢰도: {current_eval.get('confidence', 0)}, 전체 최고 신뢰도: {highest_overall_confidence}).")
        else:
            log.append(f"[반복 {iter_count+1}] 계획 평가 정보가 없어 최적해를 업데이트하지 않습니다.")
        return state # 변경 없음
    
def node_refine_search_query(state: ComplexSearchGraphState) -> ComplexSearchGraphState:
    """검색 결과를 개선하기 위해 다음 검색 쿼리를 생성하는 노드"""
    log = state["reasoning_log"]
    original_query = state["original_query"]
    # 이전 검색 쿼리는 current_search_query에 이미 있음.
    additional_context = state.get("additional_context")
    lang = state["language"]
    iter_count = state["iteration_count"]
    user_info_uncertain = state.get("user_info_uncertain", False)

    # user_info_uncertain이 True이면 search_history를 사용하지 않음
    search_history_summary = None
    if not user_info_uncertain:
        # 이전 검색 결과 요약: 모든 평가된 스니펫의 이유와 점수, 그리고 가장 최근 계획의 비평을 사용
        critique_info = ""
        if state.get("current_plan_evaluation"):
            critique_info = f"최근 계획의 문제점: {state['current_plan_evaluation'].get('critique', '없음')}."

        relevance_summary_parts = []
        for eval_snip in state.get("relevant_snippets_evaluation", []):
            relevance_summary_parts.append(f"정보 조각(점수 {eval_snip['score']}): '{eval_snip['text'][:50].replace(chr(10), ' ')}...' (이유: {eval_snip['reason']})")
        relevance_summary = "\n".join(relevance_summary_parts)
        if not relevance_summary: relevance_summary = "관련 정보 조각을 찾지 못했거나 평가 정보가 없음"

        search_history_summary = f"{critique_info}이전 검색에서 평가된 정보: {relevance_summary}"
        search_history_summary = search_history_summary[:1000]

    log.append(f"[반복 {iter_count+1}] 다음 검색을 위한 새 키워드 생성 시도...")
    logging.info(f"[Graph][Iter {iter_count+1}] Attempting to generate new keywords.")

    new_keywords = generate_search_keywords_langchain(
        original_query,
        state["current_search_query"], # 이전 검색어
        additional_context,
        lang,
        search_history_summary=search_history_summary,
        strict_user_query_only=user_info_uncertain
    )

    if new_keywords == "NO_BETTER_KEYWORDS" or not new_keywords.strip() or new_keywords == state["current_search_query"]:
        log.append("- 더 이상 개선된 검색어를 생성하지 못했습니다. 이전 검색어를 유지하거나 검색을 종료합니다.")
        logging.warning(f"[Graph][Iter {iter_count+1}] No new keywords generated or same as previous. Signal to potentially end.")
        # 이 경우, conditional_edge_decide_next_step에서 루프 종료를 결정할 수 있도록 current_search_query를 변경하지 않거나,
        # 특별한 플래그를 상태에 추가할 수 있음. 여기서는 current_search_query를 그대로 둠.
        # 또는, "NO_MORE_REFINEMENT" 같은 상태를 추가하여 명시적으로 알림
        return {**state, "current_search_query": "FINAL_ATTEMPT_NO_REFINEMENT_POSSIBLE"} # 특수 값으로 설정하여 루프 종료 유도
    else:
        log.append(f"- 생성된 새 검색어: '{new_keywords}'")
        logging.info(f"[Graph][Iter {iter_count+1}] Generated new search query: '{new_keywords}'")
        return {**state, "current_search_query": new_keywords, "iteration_count": state["iteration_count"] + 1}

def node_prepare_final_output(state: ComplexSearchGraphState) -> ComplexSearchGraphState:
    """최종 결과를 Raika 에이전트 형식에 맞게 준비하는 노드"""
    log = state["reasoning_log"]
    log.append("\n최종 결과 준비 중...")
    logging.info("[Graph] Preparing final output for Raika.")

    final_reasoning_summary_str = "\n".join(log)

    if state.get("best_overall_plan_text") and state.get("highest_overall_confidence", -1) >= 5: # 성공 임계값
        final_output = {
            "status": "success",
            "query": state["original_query"],
            "best_snippet": state.get("best_overall_snippet_text", "해당 없음"),
            "best_plan": state["best_overall_plan_text"],
            "reasoning_summary": final_reasoning_summary_str,
            "confidence": state["highest_overall_confidence"],
            "language": state["language"]
        }
        log.append(f"성공적인 결과 생성 (신뢰도: {state['highest_overall_confidence']}).")
    else:
        final_output = {
            "status": "failure",
            "query": state["original_query"],
            "reasoning_summary": final_reasoning_summary_str,
            "message": "충분히 신뢰할 수 있는 계획을 수립하지 못했습니다." if state["language"] == "ko" \
                        else "Could not formulate a confident plan based on search results.",
            "language": state["language"],
            "highest_confidence_achieved": state.get("highest_overall_confidence", -1)
        }
        log.append(f"만족스러운 계획 수립 실패 (최고 신뢰도: {state.get('highest_overall_confidence', -1)}).")

    return {**state, "final_output_for_raika": final_output}


# --- LangGraph 조건부 엣지 함수 ---
def conditional_edge_decide_next_step(state: ComplexSearchGraphState) -> str:
    """다음 단계를 결정하는 조건부 엣지 로직"""
    iter_count = state["iteration_count"] # initialize에서 0으로 시작하고 refine query에서 1 증가
    max_iters = state["max_iterations"]
    current_plan_eval = state.get("current_plan_evaluation")
    current_query = state.get("current_search_query", "")

    logging.debug(f"[Graph][Router] Iter: {iter_count}, Max: {max_iters}, Confidence: {current_plan_eval.get('confidence') if current_plan_eval else 'N/A'}, Query: '{current_query}'")


    # current_search_query가 특수 값이면 무조건 종료
    if current_query == "FINAL_ATTEMPT_NO_REFINEMENT_POSSIBLE":
        logging.info("[Graph][Router] No more query refinement possible. Ending.")
        return "prepare_output_node" # 최종 출력 노드로
    
    if iter_count >= max_iters:
        logging.info(f"[Graph][Router] Max iterations ({max_iters}) reached. Proceeding to output.")
        return "prepare_output_node" # 최종 출력 노드로
    
    if current_plan_eval:
        confidence = current_plan_eval.get('confidence', 0)
        is_sound = current_plan_eval.get('is_sound', False)
        # 높은 신뢰도의 계획이 이미 발견된 경우 (전체 최고 신뢰도 또는 현재 신뢰도 기준)
        current_confidence = current_plan_eval.get('confidence', 0) if current_plan_eval else 0
        if current_confidence >= 8 or state.get("highest_overall_confidence", -1) >= 8:
            logging.info(f"[Graph][Router] High confidence solution found (Current: {current_confidence}, Overall: {state.get('highest_overall_confidence', -1)}). Ending.")
            return "prepare_output_node"

        # 현재 반복의 계획이 좋지 않거나, 스니펫이 없었거나, 계획 생성이 안된 경우 -> 쿼리 개선 시도
        if not is_sound or confidence < 5 or not state.get("current_plan"):
            logging.info(f"[Graph][Router] Current plan is not good enough (Sound: {is_sound}, Conf: {confidence}) or no plan/snippet. Refining query.")
            return "refine_query_node" # 검색어 개선 노드로

    # 위 조건에 해당하지 않으면 다음 검색 반복 (실제로는 refine_query_node를 거쳐 perform_search_node로 감)
    # 명시적으로 검색어 개선을 먼저 시도하도록 refine_query_node로 보냄
    logging.info(f"[Graph][Router] Defaulting to query refinement or next iteration.")
    return "refine_query_node"

# LangGraph 빌더 함수
def build_complex_search_graph():
    """복합 검색 및 추론을 위한 LangGraph를 빌드하고 컴파일함."""
    if not model or not processor:
        raise ValueError("Model and Processor must be set before building the graph.")
    
    graph_builder = StateGraph(ComplexSearchGraphState)

    # 노드 추가
    graph_builder.add_node("initialize_node", node_initialize_graph)
    graph_builder.add_node("perform_search_node", node_perform_search)
    graph_builder.add_node("evaluate_snippets_node", node_evaluate_snippets)
    graph_builder.add_node("generate_evaluate_plan_node", node_generate_and_evaluate_plan)
    graph_builder.add_node("update_overall_best_node", node_update_overall_best)
    graph_builder.add_node("refine_query_node", node_refine_search_query)
    graph_builder.add_node("prepare_output_node", node_prepare_final_output)

    # 진입점 설정
    graph_builder.set_entry_point("initialize_node")

    # 엣지 연결
    graph_builder.add_edge("initialize_node", "perform_search_node") # 초기화 후 바로 검색
    graph_builder.add_edge("perform_search_node", "evaluate_snippets_node")
    graph_builder.add_edge("evaluate_snippets_node", "generate_evaluate_plan_node")
    graph_builder.add_edge("generate_evaluate_plan_node", "update_overall_best_node")

    # 조건부 엣지: update_overall_best_node 이후 다음 단계 결정
    graph_builder.add_conditional_edges(
        "update_overall_best_node", # 이 노드 실행 후
        conditional_edge_decide_next_step, # 이 함수로 다음 경로를 결정
        {
            "refine_query_node": "refine_query_node",   # 쿼리 개선 필요 시
            "prepare_output_node": "prepare_output_node"    # 종료 조건 충족 시 
        }
    )

    # refine_query_node 실행 후 다시 검색 수행
    graph_builder.add_edge("refine_query_node", "perform_search_node")

    # 최종 출력 노드 이후 그래프 종료
    graph_builder.add_edge("prepare_output_node", END)

    # 그래프 컴파일
    compiled_graph = graph_builder.compile()
    logging.info("LangGraph for complex search has been compiled.")
    return compiled_graph

# 전역 그래프 인스턴스 (애플리케이션 시작 시 한 번 빌드)
# 주의: 모델/프로세서가 로드된 후에 빌드해야 함.
# Raika_Gemma_FastAPI.py의 startup 이벤트에서 set_model_and_processor 호출 후 빌드하는 것을 권장.
compiled_complex_search_graph: Optional[StateGraph] = None

def initialize_and_get_compiled_graph():
    global compiled_complex_search_graph
    if compiled_complex_search_graph is None:
        if model and processor:
            compiled_complex_search_graph = build_complex_search_graph()
        else:
           logging.error("Cannot build LangGraph: Model or processor not yet initialized.")
            # 이 경우, LangGraph를 사용하는 함수는 호출되면 안됨.
    return compiled_complex_search_graph


def search_and_reason_for_complex_problem_langgraph(
    query: str,
    problem_type: str,
    additional_context: Optional[str] = None,
    max_iterations: int = 2, # 그래프 루프 반복 횟수
    language="en",
    user_info_uncertain: bool = False
) -> Optional[Dict[str, any]]:
    """
    [LangGraph 적용 버전]
    복잡한 문제에 대해 검색, 관련성 평가, 적용 계획, 계획 평가를 LangGraph를 사용하여 수행.
    
    Args:
        user_info_uncertain: True일 경우, 오로지 사용자 질문만 기반으로 키워드 생성
    
    Returns: 최종 응답 생성을 위한 딕셔너리 또는 None (오류 시).
    """
    global model, processor # 함수 내에서 사용 전에 확인
    if not model or not processor:
        logging.error("LangGraph search_and_reason: Model or processor not set. Cannot proceed.")
        return {
            "status": "error",
            "message": "LLM Model/Processor not initialized.",
            "query": query,
            "language": language
        }
    
    graph_app = initialize_and_get_compiled_graph() # 그래프 가져오기 (필요시 빌드)
    if not graph_app:
        logging.error("LangGraph search_and_reason: Compiled graph is not available.")
        return {
            "status": "error",
            "message": "LangGraph application not compiled or available.",
            "query": query,
            "language": language
        }

    logging.info(f"Starting LangGraph complex search for: '{query}' (Type: {problem_type}, Lang: {language}, MaxIters: {max_iterations}, UserInfoUncertain: {user_info_uncertain})")

    initial_state: ComplexSearchGraphState = {
        "original_query": query,
        "problem_type": problem_type,
        "language": language,
        "max_iterations": max_iterations,
        "additional_context": additional_context,
        "user_info_uncertain": user_info_uncertain,
        # 나머지 필드들은 initialize_node에서 채워짐
        "current_search_query": query, # 명시적 초기화
        "iteration_count": 0,
        "reasoning_log": [],
        "search_results_snippets": [],
        "relevant_snippets_evaluation": [],
        "current_best_snippet_from_iteration": None,
        "current_plan": None,
        "current_plan_evaluation": None,
        "best_overall_snippet_text": None,
        "best_overall_plan_text": None,
        "highest_overall_confidence": -1,
        "final_output_for_raika": None
    }

    try:
        # config={"recursion_limit": max_iterations + 5} 와 같이 재귀 깊이 설정 가능 (루프가 있는 경우)
        # LangGraph는 내부적으로 상태를 전달하며 노드를 실행하므로, 일반적인 Python 재귀 깊이와는 다름.
        # max_iterations는 그래프 로직 내에서 반복을 제어하기 위한 용도.
        final_graph_state = graph_app.invoke(initial_state, {"recursion_limit": max_iterations * 4 + 10}) # 충분한 재귀 깊이 제공

        if final_graph_state and "final_output_for_raika" in final_graph_state:
            output = final_graph_state["final_output_for_raika"]
            logging.info(f"LangGraph execution completed. Status: {output.get('status') if output else 'N/A'}")
            # 최종 로그 출력 (디버깅용)
            # full_log = "\n".join(final_graph_state.get("reasoning_log", []))
            # logging.debug(f"Full LangGraph Reasoning Log for query '{query}':\n{full_log}")
            return output
        else:
            logging.error(f"LangGraph execution finished but 'final_output_for_raika' not found in state for query '{query}'. State: {final_graph_state}")
            return {
                "status": "error",
                "message": "LangGraph execution error: Final output missing.",
                "query": query,
                "language": language,
                "reasoning_log_summary": "\n".join(final_graph_state.get("reasoning_log", ["No log available"])[:5]) # 로그 일부
            }

    except Exception as e:
        import traceback
        logging.error(f"Exception during LangGraph execution for query '{query}': {e}\n{traceback.format_exc()}")
        return {
            "status": "error",
            "message": f"LangGraph invocation exception: {str(e)}",
            "query": query,
            "language": language
        }


if __name__ == "__main__":
    # 독립 테스트를 위한 모델 로드
    model, processor = load_model_for_testing()

    # test_query_en = "What were the key concepts used to prove Fermat's Last Theorem?"
    # print(f"\n--- 복잡한 검색 테스트 (한국어) ---")
    # final_prompt_en = search_and_reason_for_complex_problem(test_query_en, "complex_math_problem", language="en")

    # # 영어 쿼리 테스트
    # test_query_en = "Get a quote for a desktop with RTX 5080"
    # prompt_en = process_with_rag(test_query_en, max_context_length=1000, language="en")
    # print(f"Generated prompt for English query '{test_query_en}':")
    # print(prompt_en)
    
    # # 한국어 쿼리 테스트
    # test_query_ko = "RTX 5080 그래픽카드가 장착된 데스크탑 견적 알려줘"
    # prompt_ko = process_with_rag(test_query_ko, max_context_length=1000, language="ko")
    # print(f"한국어 쿼리에 대한 생성된 프롬프트 '{test_query_ko}':")
    # print(prompt_ko)
 
    # # 영어 RAG 시스템 결과를 통한 응답 생성 테스트
    # messages_en = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": final_prompt_en}
    #         ]
    #     }
    # ]

    # # 메시지를 모델에 맞게 처리
    # inputs_en = processor.apply_chat_template(
    #     messages_en, 
    #     add_generation_prompt=True, 
    #     tokenize=True,
    #     return_dict=True, 
    #     return_tensors="pt"
    # ).to(model.device, dtype=torch.bfloat16)

    # input_len_en = inputs_en["input_ids"].shape[-1]

    # # 모델 추론 수행
    # with torch.inference_mode():
    #     generation_en = model.generate(
    #         **inputs_en, 
    #         max_new_tokens=1500, 
    #         do_sample=True,
    #         temperature=0.7
    #     )
    #     generation_en = generation_en[0][input_len_en:]

    # # 생성된 텍스트 디코딩
    # response_en = processor.decode(generation_en, skip_special_tokens=True)
    
    # print("\nGenerated English response:")
    # print(response_en)
    
    # # 한국어 RAG 시스템 결과를 통한 응답 생성 테스트
    # messages_ko = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": prompt_ko}
    #         ]
    #     }
    # ]

    # # 메시지를 모델에 맞게 처리
    # inputs_ko = processor.apply_chat_template(
    #     messages_ko, 
    #     add_generation_prompt=True, 
    #     tokenize=True,
    #     return_dict=True, 
    #     return_tensors="pt"
    # ).to(model.device, dtype=torch.bfloat16)

    # input_len_ko = inputs_ko["input_ids"].shape[-1]

    # # 모델 추론 수행
    # with torch.inference_mode():
    #     generation_ko = model.generate(
    #         **inputs_ko, 
    #         max_new_tokens=500, 
    #         do_sample=True,
    #         temperature=0.7
    #     )
    #     generation_ko = generation_ko[0][input_len_ko:]

    # # 생성된 텍스트 디코딩
    # response_ko = processor.decode(generation_ko, skip_special_tokens=True)
    
    # print("\n생성된 한국어 응답:")
    # print(response_ko)


# --- (25.05.15) 개선된 RAG를 위한 메인 함수 ---

    print("\n--- GoogleSearch_Gemma.py: Test Suite ---")

    # # 테스트 케이스 1: 단순 정보 검색 (영어)
    # test_query_simple_en = "What is the capital of France?"
    # print(f"\n[Test Case 1: Simple Info Retrieval - EN] Query: '{test_query_simple_en}'")
    # search_type_1 = classify_search_type_langchain(test_query_simple_en, language="en")
    # print(f"  - Classified Search Type: {search_type_1}")
    # if model and processor: # 모델이 로드되었을 경우에만 실행
    #     # recursive_search는 (str, bool, int) 튜플을 반환
    #     rag_content_1, satisfied_1, iterations_1 = recursive_search(test_query_simple_en, language="en", max_iterations=1)
    #     print(f"  - Recursive Search Satisfied: {satisfied_1}, Iterations: {iterations_1}")
    #     print(f"  - RAG Content (first 200 chars): {rag_content_1[:200] if rag_content_1 else 'N/A'}")
    # else:
    #     print("  - Skipping RAG test as model/processor not fully loaded.")

    # # 테스트 케이스 2: 단순 정보 검색 (한국어)
    # test_query_simple_ko = "RTX 5080 그래픽카드가 장착된 데스크탑 견적 알려줘."
    # print(f"\n[Test Case 2: Simple Info Retrieval (Movie Search) - KO] Query: '{test_query_simple_ko}'")
    # search_type_2 = classify_search_type_langchain(test_query_simple_ko, language="ko")
    # print(f"  - Classified Search Type: {search_type_2}")
    # if model and processor:
    #     rag_content_2, satisfied_2, iterations_2 = recursive_search(test_query_simple_ko, language="ko", max_iterations=2)
    #     print(f"  - Recursive Search Satisfied: {satisfied_2}, Iterations: {iterations_2}")
    #     print(f"  - RAG Content (first 200 chars): {rag_content_2[:200] if rag_content_2 else 'N/A'}")
    # else:
    #     print("  - Skipping RAG test as model/processor not fully loaded.")


    # 테스트 케이스 3: 복잡한 문제 해결 검색 (수학)
    # test_query_complex_math = "페르마의 마지막 정리를 증명하는 데 사용된 핵심 개념들은 무엇인가?"
    # # test_query_complex_math = "explain the RSA algorithm steps with a simple example"
    # print(f"\n[Test Case 3: Complex Math Problem - KO] Query: '{test_query_complex_math}'")
    # search_type_3 = classify_search_type_langchain(test_query_complex_math, language="ko")
    # print(f"  - Classified Search Type: {search_type_3}")
    # if model and processor:
    #     complex_search_result_3 = search_and_reason_for_complex_problem(test_query_complex_math, search_type_3, language="ko", max_iterations=1)
    #     print(f"  - Complex Search Result Status: {complex_search_result_3.get('status') if complex_search_result_3 else 'N/A'}")
    #     if complex_search_result_3 and complex_search_result_3.get('status') == 'success':
    #         print(f"    - Best Snippet (first 100): {complex_search_result_3.get('best_snippet', '')[:100]}")
    #         print(f"    - Best Plan (first 100): {complex_search_result_3.get('best_plan', '')[:100]}")
    #         print(f"    - Confidence: {complex_search_result_3.get('confidence')}")
    #     elif complex_search_result_3:
    #          print(f"    - Message: {complex_search_result_3.get('message')}")
    #     # print(f"  - Reasoning Summary (first 300 chars): {complex_search_result_3.get('reasoning_summary', '')[:300] if complex_search_result_3 else 'N/A'}")
    # else:
    #     print("  - Skipping Complex Search test as model/processor not fully loaded.")

    # 테스트 케이스 3.1: 복잡한 문제 해결 검색 (수학) (LangGraph)

    # LangGraph 빌드 (모델/프로세서 설정 후)
    graph_app_instance = initialize_and_get_compiled_graph()

    if graph_app_instance:
        print("\n--- LangGraph 복합 검색 테스트 (한국어) ---")
        # test_query_complex_ko = "서울에서 제주도까지 가장 빠르게 가는 방법과 비용은 얼마인가요? 렌트카 포함해서 알려주세요."
        test_query_complex_ko = "페르마의 마지막 정리를 증명하는 데 사용된 핵심 개념들은 무엇인가?"
        problem_type_ko = classify_search_type_langchain(test_query_complex_ko, language="ko") # "complex_reasoning_problem" 또는 "complex_math_problem"
        
        print(f"테스트 쿼리: '{test_query_complex_ko}', 분류된 유형: {problem_type_ko}")

        final_result_ko = search_and_reason_for_complex_problem_langgraph(
            query=test_query_complex_ko,
            problem_type=problem_type_ko,
            max_iterations=2, # 테스트를 위해 반복 줄임
            language="ko"
        )

        print("\n--- LangGraph 실행 결과 ---")
        if final_result_ko:
            print(f"상태: {final_result_ko.get('status')}")
            print(f"원본 쿼리: {final_result_ko.get('query')}")
            if final_result_ko.get('status') == 'success':
                print(f"최고 스니펫 (일부): {final_result_ko.get('best_snippet', '')[:200]}...")
                print(f"최고 계획 (일부): {final_result_ko.get('best_plan', '')[:200]}...")
                print(f"신뢰도: {final_result_ko.get('confidence')}")
            else:
                print(f"메시지: {final_result_ko.get('message')}")
                print(f"달성된 최고 신뢰도: {final_result_ko.get('highest_confidence_achieved')}")
            
            # 전체 추론 로그 (너무 길 수 있으므로 일부만 출력 또는 파일 저장 고려)
            # print("\n--- 전체 추론 로그 ---")
            # reasoning_log_summary = final_result_ko.get('reasoning_summary', "추론 로그 없음.")
            # print(reasoning_log_summary[:1000] + "..." if len(reasoning_log_summary) > 1000 else reasoning_log_summary)
        else:
            print("LangGraph 실행 중 심각한 오류 발생 또는 결과 없음.")
    else:
        print("LangGraph 애플리케이션을 빌드할 수 없습니다. 모델/프로세서 설정을 확인하세요.")

    print("\n--- GoogleSearch_Gemma.py: LangGraph 테스트 완료 ---")


    # # 테스트 케이스 4: 키워드 추출 실패 시나리오 (RAG_Result.txt의 내용 재현)
    # # 이 테스트는 Raika_Gemma_FastAPI.py의 handle_general_conversation 내에서 테스트하는 것이 더 적합합니다.
    # # GoogleSearch_Gemma.py는 키워드를 입력받는 것을 전제로 하므로, 여기서는 직접적인 재현이 어렵습니다.
    # # 다만, 빈 검색어로 recursive_search를 호출했을 때의 동작을 확인할 수 있습니다.
    # test_query_empty_keywords = "" # assess_search_requirement가 키워드 추출에 실패한 상황 가정
    # print(f"\n[Test Case 4: Empty Keywords Scenario] Query: '{test_query_empty_keywords}' (Simulating keyword extraction failure)")
    # search_type_4 = classify_search_type_langchain(test_query_empty_keywords, language="en") # "simple_information_retrieval" 반환 예상
    # print(f"  - Classified Search Type: {search_type_4}")
    # if model and processor:
    #     # recursive_search는 검색어가 비어있으면 Google Search에서 빈 결과를 받고, 이를 처리해야 함
    #     rag_content_4, satisfied_4, iterations_4 = recursive_search(test_query_empty_keywords, language="en", max_iterations=1)
    #     print(f"  - Recursive Search Satisfied: {satisfied_4}, Iterations: {iterations_4}")
    #     print(f"  - RAG Content: {rag_content_4 if rag_content_4 else 'N/A'}") # "No search results found..." 예상
    # else:
    #     print("  - Skipping RAG test as model/processor not fully loaded.")

    # # 테스트 케이스 5: Raika가 영화를 찾아달라는 요청 (RAG_Result.txt)
    # # Renard: *Pet your head smoothly* I had a dream that I was watching a movie I saw a long time ago. Raika, I'm looking for a this movie. 🎞️ Can you help me? 🤔
    # # Raika의 handle_general_conversation에서 assess_search_requirement를 호출하고, 그 결과로 검색 키워드를 생성해야 함.
    # # 여기서는 assess_search_requirement가 "old movie dream" 같은 키워드를 생성했다고 가정하고 테스트.
    # user_input_movie = "I had a dream that I was watching a movie I saw a long time ago. Raika, I'm looking for a this movie."
    # # 가정된 키워드 (원래는 assess_search_requirement가 생성)
    # assumed_keywords_for_movie = "old movie dream" # 또는 "movie long time ago dream" 등
    # print(f"\n[Test Case 5: Movie Search from RAG_Result.txt - User: '{user_input_movie}']")
    # print(f"  - Assumed Keywords (from assess_search_requirement): '{assumed_keywords_for_movie}'")
    # search_type_5 = classify_search_type_langchain(assumed_keywords_for_movie, language="en")
    # print(f"  - Classified Search Type: {search_type_5}")
    # if model and processor:
    #     rag_content_5, satisfied_5, iterations_5 = recursive_search(assumed_keywords_for_movie, user_input_movie, language="en", max_iterations=1)
    #     print(f"  - Recursive Search Satisfied: {satisfied_5}, Iterations: {iterations_5}")
    #     print(f"  - RAG Content (first 200 chars): {rag_content_5[:200] if rag_content_5 else 'N/A'}")

    #     # 만약 RAG_Result.txt 처럼 Iter 1: No content found... 가 반복된다면,
    #     # recursive_search 내부의 Google Search_keywords)가 빈 결과를 반환했거나,
    #     # 그 이후 combined_results가 비어있어서 LLM 평가 단계로 넘어가지 못하고 다음 iteration으로 간 것입니다.
    #     # generate_search_keywords가 빈 키워드를 반환하거나, Google Search 자체가 결과를 못 찾는 경우 발생.
    #     # RAG_Result.txt의 "Iter 1: No content found from any individual keyword searches in this iteration."는 FastAPI 쪽의 루프에서 발생.
    #     # GoogleSearch_Gemma.py에서는 recursive_search가 빈 결과를 반환하는 형태로 나타날 것입니다.
    #     if not rag_content_5 or "No search results found" in rag_content_5:
    #          print(f"  - !!! Simulating RAG_Result.txt issue: recursive_search returned no meaningful content for keywords '{assumed_keywords_for_movie}'. This would lead to 'No content found' in FastAPI.")
    # else:
    #     print("  - Skipping RAG test as model/processor not fully loaded.")


    print("\n--- Test Suite Finished ---")
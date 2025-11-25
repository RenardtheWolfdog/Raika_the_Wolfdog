# Raika_the_Wolfdog
AI Agent/Companion Raika the Wolfdog

## [KR] 🐺 Raika the Wolfdog: 실시간 멀티모달 AI 컴패니언
- Raika는 단순한 챗봇이 아닌, 사용자와 교감하며 성장하는 Full-Stack AI Agent입니다. 로컬 환경(RTX 4060Ti)에서의 효율적인 LLM 구동부터, RAG 기반의 장기 기억, 실시간 음성 대화까지 하나의 유기적인 시스템으로 구현되었습니다.

### 📂 주요 서버 파일 설명 (Server File Descriptions)
- 이 리포지토리는 Raika의 두뇌와 감각을 담당하는 백엔드 서버 로직을 포함합니다.

### 1. 핵심 로직 & 오케스트레이션 (Core Logic & Orchestration)

- Server/Raika_Gemma_FastAPI.py

- 역할: 전체 시스템의 메인 컨트롤러입니다.
- 상세: FastAPI 기반의 비동기 서버로, WebSocket을 통해 클라이언트(React)와 실시간으로 소통합니다. LLM 모델(Gemma-3)의 **지연 로딩(Lazy Loading)**을 구현하여 초기 리소스 점유를 최적화했고, 텍스트 생성과 동시에 립싱크(Lip-sync) 데이터를 스트리밍하여 생동감 있는 인터랙션을 제공합니다.

- Server/ShortTermMemory.py 

- 역할: Raika의 '장기 기억'을 담당하는 핵심 모듈입니다.
- 상세: 단순 키워드 매칭의 한계를 넘어, Redis Vector Store를 활용한 **하이브리드 검색 시스템(Vector Similarity + Keyword Matching)**을 직접 구현했습니다. 사용자의 발화 의도를 정확히 파악하고 과거의 맥락을 놓치지 않도록 설계되었습니다.

- Server/GoogleSearch_Gemma.py

- 역할: 웹 검색 및 복합 추론을 수행하는 에이전트 모듈입니다.
- 상세: LangGraph를 도입하여 단순 검색을 넘어선 '추론 루프'를 구축했습니다. 검색 결과가 불충분할 경우 스스로 쿼리를 재구성하여 재검색하는 재귀적 검색(Recursive Search) 로직이 포함되어 있습니다.

### 2. 문서 분석 & RAG (Document Analysis & RAG)

- Server/document_summarizer_Gemma_Lang.py

- 역할: 대용량 문서 처리 및 질의응답 시스템입니다.
- 상세: 문서를 의미 단위 청크(Chunk)로 분할하고 임베딩하여 벡터 DB에 저장합니다. LangChain과 LangGraph를 활용해 문서의 맥락을 유지하며 사용자의 복잡한 질문에도 정확하게 답변할 수 있는 RAG 파이프라인을 구축했습니다.

### 3. 멀티모달 서비스 (Multimodal Services)
   
- Server/Raika_TTS_Server.py & Server/Raika_TTS.py

- 역할: Raika의 목소리를 생성하는 TTS(Text-to-Speech) 서버입니다.
- 상세: Chatterbox 모델을 기반으로 하며, 단순 음성 생성뿐만 아니라 Live2D 아바타를 위한 **실시간 립싱크 에너지 값(Viseme)**을 계산하여 WebSocket으로 전송합니다. 한국어/영어를 자동 감지하여 자연스럽게 발화합니다.

- Server/deepseek_ocr_server.py 등 (_client.py, _pipeline.py)

- 역할: PDF 및 이미지 내 텍스트를 인식하는 OCR 마이크로서비스입니다.
- 상세: DeepSeek-OCR 모델을 서빙하며, 처리된 결과는 Redis에 캐싱되어 동일한 문서에 대한 중복 연산을 방지합니다.

### 4. 인프라 & 유틸리티 (Infrastructure & Utilities)

- Server/Raika_MongoDB_FastAPI.py: Motor 라이브러리를 사용한 비동기 MongoDB 클라이언트로, 대화 로그와 세션 데이터를 효율적으로 관리합니다.
- Server/redis_utils.py: Redis를 단순 캐시가 아닌 Vector Database로 활용하기 위한 유틸리티입니다. RediSearch 인덱스 생성 및 벡터 검색 쿼리를 처리합니다.
- Server/Raika_S3.py: AWS S3와 연동하여 멀티미디어 파일을 비동기로 업로드 및 다운로드합니다.
- Server/run_servers_FastAPI.py: Python의 Multiprocessing을 활용하여 메인 서버, DB 서버, TTS 서버 등을 병렬로 실행하고 관리하는 스크립트입니다.

## [EN] 🐺 Raika the Wolfdog: Real-time Multimodal AI Companion
- Raika is not just a chatbot, but a Full-Stack AI Agent that interacts and grows with the user. It is implemented as an organic system, ranging from efficient LLM operation in a local environment (RTX 4060Ti) to RAG-based long-term memory and real-time voice conversation.

### 📂 Server File Descriptions
- This repository contains the backend server logic responsible for Raika's brain and senses.

### 1. Core Logic & Orchestration

- Server/Raika_Gemma_FastAPI.py

- Role: Main controller of the entire system.
- Details: An asynchronous server based on FastAPI that communicates with the client (React) in real-time via WebSocket. It implements Lazy Loading for the LLM model (Gemma-3) to optimize initial resource usage and streams lip-sync data simultaneously with text generation for lively interactions.

- Server/ShortTermMemory.py 

- Role: The core module responsible for Raika's 'Long-term Memory'.
- Details: Implemented a Hybrid Retrieval System (Vector Similarity + Keyword Matching) using Redis Vector Store, going beyond simple keyword matching. Designed to accurately grasp the user's intent and retain past context.

- Server/GoogleSearch_Gemma.py

- Role: Agent module performing web search and complex reasoning.
- Details: Built a 'reasoning loop' beyond simple search using LangGraph. Includes Recursive Search logic that reconstructs queries and re-searches if the initial results are insufficient.

### 2. Document Analysis & RAG

- Server/document_summarizer_Gemma_Lang.py

- Role: Large-scale document processing and QA system.
- Details: Splits documents into semantic chunks, embeds them, and stores them in a Vector DB. Built a RAG pipeline using LangChain and LangGraph to answer complex user questions accurately while maintaining document context.

### 3. Multimodal Services

- Server/Raika_TTS_Server.py & Server/Raika_TTS.py

- Role: TTS (Text-to-Speech) server generating Raika's voice.
- Details: Based on the Chatterbox model, it calculates and sends real-time lip-sync energy values (Viseme) via WebSocket for the Live2D avatar, in addition to generating speech. Automatically detects Korean/English for natural pronunciation.

- Server/deepseek_ocr_server.py etc. (_client.py, _pipeline.py)

- Role: OCR microservice recognizing text in PDFs and images.
- Details: Serves the DeepSeek-OCR model and caches processed results in Redis to prevent redundant computations for the same document.

### 4. Infrastructure & Utilities

- Server/Raika_MongoDB_FastAPI.py: Asynchronous MongoDB client using Motor library, efficiently managing chat logs and session data.
- Server/redis_utils.py: Utility for using Redis as a Vector Database, not just a cache. Handles RediSearch index creation and vector search queries.
- Server/Raika_S3.py: Integrates with AWS S3 to upload and download multimedia files asynchronously.
- Server/run_servers_FastAPI.py: Script managing the parallel execution of the main server, DB server, TTS server, etc., using Python's Multiprocessing.

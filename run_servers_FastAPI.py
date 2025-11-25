# run_servers_FastAPI.py

import multiprocessing
import uvicorn
import logging
import time
import sys
import os

# --- 서버 실행 함수 ---

def run_gemma_fastapi_server():
    """
    Gemma FastAPI 서버를 초기화하고 실행합니다.
    """
    try:
        # torchvision으로 인한 transformers 초기 임포트 실패 방지
        import os as _os
        _os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
        _os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
        from Raika_Gemma_FastAPI import create_app
        gemma_app = create_app()

        host = os.environ.get("RAIKA_GEMMA_HOST", "0.0.0.0")
        port = int(os.environ.get("RAIKA_GEMMA_PORT", "5000"))
        workers = int(os.environ.get("RAIKA_GEMMA_WORKERS", "1"))

        logging.info(f"GEMMA-3 서버용 Uvicorn을 시작합니다... host={host} port={port} workers={workers}")
        uvicorn.run(gemma_app, host=host, port=port, workers=workers)
    except Exception as e:
        import traceback
        logging.error(f"GEMMA-3 12B 4bit FastAPI 서버 프로세스 오류: {e}\n{traceback.format_exc()}")

def run_mongodb_fastapi_server():
    """
    MongoDB FastAPI 서버를 초기화하고 실행합니다.
    """
    try:
        from Raika_MongoDB_FastAPI import app as mongo_app
        host = os.environ.get("RAIKA_MONGO_HOST", "0.0.0.0")
        port = int(os.environ.get("RAIKA_MONGO_PORT", "5002"))
        workers = int(os.environ.get("RAIKA_MONGO_WORKERS", "1"))
        logging.info(f"MongoDB 서버용 Uvicorn을 시작합니다... host={host} port={port} workers={workers}")
        uvicorn.run(mongo_app, host=host, port=port, workers=workers)
    except Exception as e:
        import traceback
        logging.error(f"MongoDB 서버 프로세스 오류: {e}\n{traceback.format_exc()}")

def run_tts_server():
    """
    TTS(WebSocket) 서버를 초기화하고 실행합니다. (립싱크용 /ws/lipsync 포함)
    """
    try:
        from Raika_TTS_Server import app as tts_app
        host = os.environ.get("RAIKA_TTS_HOST", "0.0.0.0")
        port = int(os.environ.get("RAIKA_TTS_PORT", "8000"))
        workers_env = int(os.environ.get("RAIKA_TTS_WORKERS", "1"))
        # WebSocket 세션 일관성을 위해 기본 1 워커 권장
        workers = max(1, workers_env)
        if workers != workers_env:
            logging.warning(f"TTS workers 조정: 요청 {workers_env} → 적용 {workers} (WS 일관성)")
        logging.info(f"TTS 서버용 Uvicorn을 시작합니다... host={host} port={port} workers={workers}")
        uvicorn.run(tts_app, host=host, port=port, workers=workers)
    except Exception as e:
        import traceback
        logging.error(f"TTS 서버 프로세스 오류: {e}\n{traceback.format_exc()}")

# --- 메인 실행 블록 ---

def main():
    """
    로깅을 설정하고 서버 프로세스를 시작하는 메인 함수입니다.
    """
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
    )

    logging.info("메인 프로세스가 시작되었습니다. 서버 프로세스를 순차적으로 초기화합니다...")

    # 프로세스 정의
    p_gemma = multiprocessing.Process(
        target=run_gemma_fastapi_server,
        name="GemmaFastAPIProcess"
    )
    p_mongo = multiprocessing.Process(
        target=run_mongodb_fastapi_server,
        name="MongoDBFastAPIProcess"
    )
    # p_tts = multiprocessing.Process(
    #     target=run_tts_server,
    #     name="TTSServerProcess"
    # )

    # =============================================================
    # 1. Gemma FastAPI 서버를 먼저 시작
    # =============================================================
    logging.info(f"프로세스 시작: {p_gemma.name}")
    p_gemma.start()

    # Gemma 서버가 초기화될 시간을 기다림
    # 이 시간 동안 Gemma 서버의 로그를 관찰하여 문제가 발생하는지 확인
    logging.info("Gemma 서버가 안정적으로 시작될 때까지 10초간 대기합니다...")
    time.sleep(10)

    # =============================================================
    # 2. MongoDB FastAPI 서버를 나중에 시작
    # =============================================================
    logging.info(f"프로세스 시작: {p_mongo.name}")
    p_mongo.start()

    # =============================================================
    # 3. TTS 서버 마지막에 시작 (립싱크용 WebSocket 포함)
    # =============================================================
    # logging.info(f"프로세스 시작: {p_tts.name}")
    # p_tts.start()

    # Ctrl+C로 종료 시 프로세스를 정상적으로 종료하기 위한 대기
    try:
        p_gemma.join()
        p_mongo.join()
        # p_tts.join()
    except KeyboardInterrupt:
        logging.info("키보드 인터럽트를 받았습니다. 모든 서버 프로세스를 종료합니다...")
        if p_gemma.is_alive():
            p_gemma.terminate()
        if p_mongo.is_alive():
            p_mongo.terminate()
        # if p_tts.is_alive():
        #     p_tts.terminate()
        
        p_gemma.join()
        p_mongo.join()
        # p_tts.join()
        logging.info("모든 서버 프로세스가 종료되었습니다.")

if __name__ == '__main__':
    # Windows에서 spawn 방식을 사용할 때 필수적인 코드
    multiprocessing.freeze_support()
    main()
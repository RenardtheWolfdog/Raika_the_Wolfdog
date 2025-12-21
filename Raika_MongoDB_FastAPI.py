# from flask import Flask, request, jsonify, session
# from flask_cors import CORS
# from waitress import serve
# """세션의 대화 내용을 실시간으로 MongoDB와 연동"""
# from pymongo.mongo_client import MongoClient

from fastapi import FastAPI, HTTPException, APIRouter, Body, Path, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import configparser
import uuid
import pytz
import secrets
import os
import logging

# --- Windows 콘솔(cp949) 환경 대응: UTF-8 스트림 재설정 ---
try:
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# 로깅 설정 부분 추가 (파일 상단에 추가)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(stream=sys.stdout),  # 콘솔 출력 (UTF-8 stdout)
        logging.FileHandler("mongodb_fastapi.log", encoding="utf-8")  # 파일 로깅 UTF-8
    ]
)

from motor.motor_asyncio import AsyncIOMotorClient # 동기 pymongo 대신 비동기 motor 사용

"""AWS"""
from Raika_S3 import S3Handler, AsyncS3Handler
# S3Handler, AsyncS3Handler 인스턴스 생성
s3_handler = S3Handler('imageandvediobucket')
async_s3_handler = AsyncS3Handler('imageandvediobucket')
import asyncio # run_in_executor

from botocore.exceptions import ClientError

# 사용자와 챗봇 이름 설정
user_name = "Renard"
bot_name = "Raika"

# MongoDB Atlas 연결 설정
config = configparser.ConfigParser()
try:
    config.read('config.ini', encoding='utf-8')
except Exception:
    config.read('config.ini')
username = config['DEFAULT']['username']
password = config['DEFAULT']['password']
dbname = config['DEFAULT']['dbname']

# --- 비동기 MongoDB 클라이언트 설정 ---
mongo_url = f'mongodb+srv://{username}:{password}@{dbname}.udmyrmv.mongodb.net/'
async_client = AsyncIOMotorClient(mongo_url)
async_db = async_client[dbname] # 비동기 DB 객체
async_conversations = async_db.conversations # 비동기 컬렉션 객체

# --- 시간 설정 ---
user_timezone = 'Asia/Seoul'
timezone = pytz.timezone(user_timezone)

"""세션의 대화 내용을 MongoDB에 저장 (비동기)"""
async def async_save_message(session_id: str, role: str, message: str, file_urls: list = None):
    now = datetime.now(timezone)
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    message_data = {
        'role': role,
        'message': message,
        'timestamp': current_time,
    }
    if file_urls:
        message_data['file_urls'] = file_urls

    # motor의 update_one 사용 (await 필요)
    result = await async_conversations.update_one(
        {'session_id': session_id},
        {'$push': {'conversation_history': message_data}},
        upsert=True
    )
    print(f"Message saved for session {session_id}. Matched: {result.matched_count}, Modified: {result.modified_count}, UpsertedId: {result.upserted_id}")
    return message_data

async def async_save_context(session_id: str, context: list):
    """대화 컨텍스트를 MongoDB에 비동기적으로 저장"""
    # motor의 update_one 사용 (await 필요)
    result = await async_conversations.update_one(
        {'session_id': session_id},
        {'$set': {'conversation_context': context}},
        upsert=True
    )
    print(f"Context saved for session {session_id}. Matched: {result.matched_count}, Modified: {result.modified_count}, UpsertedId: {result.upserted_id}")

async def async_load_session(session_id: str):
    """특정 세션의 대화 기록과 컨텍스트를 MongoDB에서 비동기적으로 로드"""
    # motor의 find_one 사용
    session_data = await async_conversations.find_one(
        {'session_id': session_id},
        {'_id': 0, 'conversation_history': 1, 'conversation_context': 1}
    )
    if session_data:
        conversation_history = session_data.get('conversation_history', [])
        conversation_context = session_data.get('conversation_context', [])

        # Flask 버전과 동일한 컨텍스트 재구성 및 파일 URL 처리 로직 유지
        # [핵심 수정]
        # conversation_context는 conversation_history로부터 파생되는 "캐시"라서,
        # 과거 버그/저장 누락으로 일부만 들어있는 경우가 자주 발생할 수 있음.
        # 이때 context가 비어있지 않더라도 history 대비 짧으면 재구성하여 최신 대사를 반영한다.
        should_rebuild_context = (not conversation_context) or (conversation_history and len(conversation_context) < len(conversation_history))
        if should_rebuild_context and conversation_history: # 컨텍스트 비었거나(또는 부분 저장) 기록으로 재구성
            conversation_context = []
            for msg in conversation_history:
                role = msg.get('role')
                message_content = msg.get('message', '')
                if role == user_name and message_content.startswith('Files:'):
                    parts = message_content.split("\n", 1)
                    text = parts[1] if len(parts) > 1 else ""
                    conversation_context.append(f"{role}: {text}\n")
                else:
                    conversation_context.append(f"{role}: {message_content}\n")

        # 기록에 파일 URL 필드 추가 (프론트엔드 표시용)
        for msg in conversation_history:
            message_content = msg.get('message', '')
            if msg.get('role') == user_name and message_content.startswith("Files:"):
                parts = message_content.split("\n", 1)
                file_urls_str = parts[0].replace("Files: ", "").strip()
                msg['file_urls'] = file_urls_str.split(", ") if file_urls_str else []
                msg['text'] = parts[1] if len(parts) > 1 else ""
            else:
                msg['text'] = message_content
            if 'file_urls' not in msg: # file_urls 필드가 없는 경우 빈 리스트 추가
                 msg['file_urls'] = []

        return conversation_history, conversation_context
    else:
        return [], []

# 마지막으로 사용한 세션 ID를 비동기적으로 저장
async def async_save_last_session(session_id):
    # 'last_session' 문서를 업데이트하거나 생성함
    # motor의 update_one 사용 (await 필요)
    await async_db.system_info.update_one(
        {'_id': 'last_session'},
        {'$set': {'session_id': session_id}},
        upsert=True
    )
    print(f"Last session ID saved: {session_id}")

# (서버 재연결 시) 마지막으로 사용한 세션 ID를 비동기적으로 가져옴
async def async_get_last_session():
    # 'last_session' 문서에서 세션 ID를 가져옴
    last_session = await async_db.system_info.find_one({'_id': 'last_session'})
    return last_session.get('session_id') if last_session else None


# --- FastAPI 라우터 설정 ---
router = APIRouter() # 라우터 객체 생성

# 세션 리스트 반환 엔드포인트
@router.get("/sessions")
async def get_sessions():
    """MongoDB에 저장된 모든 세션의 ID와 이름을 비동기적으로 조회하여 반환"""
    try:
        sessions = []
        # motor의 find 사용 (비동기 반복자)
        cursor = async_conversations.find({}, {'_id': 0, 'session_id': 1, 'name': 1})
        async for session in cursor:
            sessions.append({
                # MongoDB의 ObjectId는 문자열로 변환할 필요 없음 (FastAPI는 자동 처리)
                'session_id': session['session_id'],
                'name': session.get('name', 'Untitled Session') # 이름 없으면 기본값
            })
        return {"sessions": sessions} # FastAPI는 딕셔너리를 자동으로 JSON으로 변환
    except Exception as e:
        # 오류 발생 시 500 상태 코드와 함께 오류 메시지 반환
        raise HTTPException(status_code=500, detail=f"Failed to fetch sessions: {str(e)}")

# 새로운 세션 생성 엔드포인트
@router.post("/start_session")
async def start_session():
    """새로운 세션 ID를 생성하고 MongoDB에 저장한 후 반환"""
    try:
        session_id = str(uuid.uuid4()) # 고유 세션 ID 생성
        # 비동기 count_documents 사용 (await 필요)
        session_count = await async_conversations.count_documents({})
        name = f"새 세션 {session_count + 1}" # 새 세션 이름 생성

        # motor의 insert_one 사용 (await 필요)
        await async_conversations.insert_one({
            'session_id': session_id,
            'name': name,
            'conversation_history': [],
            'conversation_context': []
        })
        await async_save_last_session(session_id) # 새 세션을 마지막 세션으로 저장
        return {"session_id": session_id, "name": name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

# 특정 세션 로드 엔드포인트
@router.get("/load_session/{session_id}")
# 경로 매개변수 타입 힌트 사용 (FastAPI 기능)
async def load_session_endpoint(session_id: str = Path(..., title="The ID of the session to load")):
    """주어진 세션 ID에 해당하는 대화 기록과 컨텍스트를 로드하여 반환"""
    try:
        conversation_history, conversation_context = await async_load_session(session_id)
        if not conversation_history and not conversation_context:
             # 세션이 존재하지 않거나 내용이 없는 경우 404 오류 발생
             raise HTTPException(status_code=404, detail="Session not found or empty")
        return {
            "conversation_history": conversation_history,
            "conversation_context": conversation_context,
            "session_id": session_id
        }
    except HTTPException:
         raise # 이미 HTTPException이면 그대로 전달
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load session: {str(e)}")

# 마지막 세션 ID 가져오기 엔드포인트
# Flask의 '/current_session'과 기능 동일 -> '/get_last_session'으로 이름 변경
@router.get("/get_last_session")
async def get_last_session_endpoint():
    """MongoDB에 저장된 마지막 사용 세션 ID를 반환"""
    try:
        session_id = await async_get_last_session()
        return {"session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get last session ID: {str(e)}")

# 특정 세션 삭제 엔드포인트
@router.delete("/delete_session/{session_id}")
async def delete_session_endpoint(session_id: str = Path(..., title="The ID of the session to delete")):
    """주어진 세션 ID에 해당하는 데이터를 MongoDB와 S3에서 삭제"""
    try:
        # MongoDB에서 세션 정보 가져오기 (비동기)
        session_data = await async_conversations.find_one({'session_id': session_id})
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        s3_delete_success = await delete_session_folder_s3(session_id)

        # MongoDB에서 세션 삭제 (비동기)
        result = await async_conversations.delete_one({'session_id': session_id})
        if result.deleted_count == 0:
            # 이 경우는 거의 발생하지 않음 (위에서 find_one으로 확인했기 때문)
            raise HTTPException(status_code=404, detail="Session not found in DB (unexpected)")

        if s3_delete_success:
            return {"message": "Session and associated files deleted successfully"}
        else:
            # S3 삭제 실패 시 메시지 명시
            return JSONResponse(
                status_code=200, # DB 삭제는 성공했으므로 200 OK 또는 다른 적절한 코드 사용 가능
                content={"message": "Session deleted from MongoDB, but S3 deletion might be incomplete"}
            )

    except HTTPException:
        raise # 404 등 미리 발생한 HTTPException은 그대로 전달
    except Exception as e:
        print(f"Error deleting session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error during session deletion: {str(e)}")

# --- S3에서 세션 폴더 및 관련 파일 삭제 ---
async def delete_session_folder_s3(session_id: str):
    """S3에서 특정 세션 폴더 내 모든 객체를 비동기적으로 삭제"""
    try:     
        # 세션 ID를 접두사로 하는 객체 목록 가져오기
        prefix = f"{session_id}/"
        logging.info(f"Listing objects with prefix: {prefix}")

        # 비동기적으로 객체 목록 가져오기
        objects_list = await async_s3_handler.async_list_objects(prefix)

        if not objects_list:
            logging.info(f"No objects found with prefix {prefix}")
            return True # 삭제할 객체가 없으면 성공으로 간주
        
        logging.info(f"Found {len(objects_list)} objects to delete: {objects_list}")

        # 비동기적으로 객체 삭제
        delete_success = await async_s3_handler.async_delete_objects(objects_list)

        if delete_success:
            logging.info(f"Successfully deleted all objects with prefix {prefix}")
            return True
        else:
            logging.error(f"Failed to delete objects with prefix {prefix}")
            return False
            
    except Exception as e:
        import traceback
        logging.error(f"Error in delete_session_files_from_s3: {str(e)}\n{traceback.format_exc()}")
        return False

# 특정 세션 이름 업데이트 엔드포인트
@router.put("/update_session/{session_id}")
# 요청 본문을 받기 위해 Body 사용 및 타입 힌트 지정
async def update_session_endpoint(
    session_id: str = Path(..., title="The ID of the session to update"),
    session_data: dict = Body(...)
):
    """주어진 세션 ID의 이름을 업데이트"""
    try:
        logging.info(f"Received update request for session {session_id} with data: {session_data}")

        new_name = session_data.get('name')
        if not new_name or not isinstance(new_name, str) or not new_name.strip():
            logging.warning(f"Invalid session name in request: {new_name}")
            raise HTTPException(status_code=400, detail="Valid 'name' field is required in the request body")

        # motor의 update_one 사용 (await 필요)
        result = await async_conversations.update_one(
            {'session_id': session_id},
            {'$set': {'name': new_name.strip()}} # 공백 제거
        )

        if result.matched_count == 0:
            logging.warning(f"Session {session_id} not found for update")
            raise HTTPException(status_code=404, detail="Session not found")

        logging.info(f"Successfully updated session {session_id} name to '{new_name.strip()}'")
        return {"message": f"Session '{session_id}' name updated successfully to '{new_name.strip()}'"}
    except HTTPException as he:
        logging.error(f"HTTP Exception in update_session: {he.detail}")
        raise # 400, 404 오류는 그대로 전달
    except Exception as e:
        import traceback
        logging.error(f"Unexpected error updating session {session_id}:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to update session name: {str(e)}")

# 마지막 세션 ID 저장 엔드포인트
@router.post("/save_last_session")
async def save_last_session_endpoint(data: dict = Body(...)):
    """요청 본문에서 session_id를 받아 마지막 사용 세션으로 저장"""
    session_id = data.get('session_id')
    if not session_id:
        raise HTTPException(status_code=400, detail="No 'session_id' provided in request body")

    try:
        await async_save_last_session(session_id)
        return {"message": "Last session saved successfully"}
    except Exception as e:
         raise HTTPException(status_code=500, detail=f"Failed to save last session: {str(e)}")

# --- 서버 초기화 및 실행 로직 (FastAPI에서는 run_servers_FastAPI.py 에서 처리) ---
# Flask 앱 생성 및 실행 부분 제거

# FastAPI 앱 인스턴스 생성 (Uvicorn이 이 'app' 객체를 찾아서 실행)
app = FastAPI(title="Raika_MongoDB_FastAPI")
app.include_router(router) # 위에서 정의한 라우터들을 앱에 포함

# CORS 미들웨어 추가
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # 허용할 출처 목록
    allow_credentials=True, # 자격 증명 허용 여부
    allow_methods=["*"], # 허용할 HTTP 메소드 (GET, POST 등)
    allow_headers=["*"], # 허용할 HTTP 헤더
)

# 서버 초기화 관련 로직 (필요 시 유지)
async def reset_database_connection_async():
    global async_client, async_db, async_conversations
    if async_client:
        async_client.close() # 비동기 클라이언트 종료
    async_client = AsyncIOMotorClient(mongo_url) # mongo_url 변수 사용 확인 (전역에 정의됨)
    async_db = async_client[dbname]
    async_conversations = async_db.conversations
    print("Async Database connection reset successfully")

async def clear_all_sessions_async():
    # motor의 delete_many 사용 (await 필요)
    result = await async_conversations.delete_many({})
    print(f"All session data cleared. Deleted count: {result.deleted_count}")
    
    
""" --- 보안 에이전트 관련 Function --- """
async def async_get_all_threats() -> list:
    """
    Fetches all documents from the threat_intelligence collection.
    """
    try:
        threats = []
        cursor = async_db.threat_intelligence.find({}, {'_id': 0})  # _id 제외
        async for threat in cursor:
            threats.append(threat)
        logging.info(f"Successfully fetched {len(threats)} items from the threat intelligence DB.")
        return threats
    except Exception as e:
        logging.error(f"Failed to fetch threat intelligence from MongoDB: {e}")
        return []
    
""" --- 사용자별 무시 목록 관리 함수 --- """
async def async_add_to_ignore_list(user_name: str, item_name: str):
    """특정 아이템을 사용자의 무시 목록에 추가"""
    if not user_name or not item_name:
        return
    
    # 'user_preferences' 컬렉션에서 사용자 무시 목록 저장
    # $addToSet 연산자를 사용하여 아이템 중복 방지
    await async_db.user_preferences.update_one(
        {'user_name': user_name},
        {'$addToSet': {'ignore_list': item_name}},
        upsert=True  # 사용자가 없으면 새로 생성
    )
    logging.info(f"'{item_name}'이(가) {user_name}의 무시 목록에 추가되었습니다. '{user_name}'s ignore list updated with '{item_name}'.")

async def async_remove_from_ignore_list(user_name: str, item_name: str):
    """특정 아이템을 사용자의 무시 목록에서 제거"""
    if not user_name or not item_name:
        return
    
    await async_db.user_preferences.update_one(
        {'user_name': user_name},
        {'$pull': {'ignore_list': item_name}}
    )
    logging.info(f"'{item_name}'이(가) {user_name}의 무시 목록에서 제거되었습니다. '{user_name}'s ignore list updated without '{item_name}'.")
    
async def async_get_ignore_list_for_user(user_name: str) -> list[str]:
    """특정 사용자의 무시 목록을 반환"""
    if not user_name:
        return []
    
    preferences = await async_db.user_preferences.find_one({'user_name': user_name})
    if preferences and 'ignore_list' in preferences:
        return preferences['ignore_list']
    return []  # 사용자가 없거나 무시 목록이 없는 경우 빈 리스트 반환
    
# 이 파일 자체를 uvicorn으로 직접 실행할 경우 (개발용 테스트)
if __name__ == "__main__":
    import uvicorn
    print("Starting Raika_MongoDB_FastAPI server directly...")
    # asyncio.run(clear_all_sessions_async()) # 테스트용: 시작 시 모든 세션 삭제
    # uvicorn.run("Raika_MongoDB_FastAPI:app", ...) -> 아래처럼 변경하는 것이 일반적
    uvicorn.run(app, host='0.0.0.0', port=5002, reload=True) # 직접 'app' 객체를 전달
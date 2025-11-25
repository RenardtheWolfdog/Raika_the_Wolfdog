from fastapi import FastAPI, WebSocket
from Raika_TTS import text_to_speech, text_to_speech_mixed, detect_language, play_wav, initialize_tts_model
import os
import asyncio
import json
import numpy as np
import re
import time
import logging
import base64
import io

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """
    애플리케이션 시작 시 TTS 모델을 초기화.
    """
    print("[TTS Server] Startup event: Initializing TTS model...")
    # 별도 스레드에서 동기 함수 실행 (이벤트 루프 차단 방지)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, initialize_tts_model)
    
    try:
        print("[TTS Server] Warming up audio playback...")
        import simpleaudio as sa
        # 짧은 무음 데이터 생성하기
        sr = 16000
        silence = np.zeros(int(sr * 0.1), dtype=np.int16)
        play_obj = sa.play_buffer(silence, 1, 2, sr)
        play_obj.wait_done()
        print("[TTS Server] Audio playback warmed up successfully.")
    except Exception as audio_warmup_error:
        print(f"[TTS Server] Audio playback warming up failed: {audio_warmup_error}")

def clean_text_for_tts_light(text: str) -> str:
    """
    간단한 TTS 전처리:
    - 코드 블록/인라인 코드(```code```, `code`) 제거
    - 행동/서술 표기(별표 *...*), 대괄호 [ ... ] 제거
    - 여분 공백 정리
    - 이모지(😚💕😍🤗...) 제거
    """
    if not isinstance(text, str):
        return ''
    try:
        # triple backticks
        text = re.sub(r"```[\s\S]*?```", " ", text)
        # inline backticks
        text = re.sub(r"`[^`]+`", " ", text)
        # asterisk actions (limit length to avoid catastrophic)
        text = re.sub(r"\*[^\*]{1,200}\*", " ", text)
        # bracketed stage directions
        text = re.sub(r"\[[^\]]{1,200}\]", " ", text)
        # collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # emoji
        text = re.sub(r"[:;=]+[-~]*[><]+[-~]*[:;=]+", " ", text)
    except Exception:
        pass
    return text

def _head_sentences_safe(text: str, lang: str, max_sentences: int = 2) -> str:
    """
    한국어/영어 공통 안전 문장 추출:
    - 강한 정규식 사용을 피하고, 문장부호(.,!? 및 동등한 유니코드) 기준으로 앞쪽 n문장만 반환
    - 문장부호가 없으면 최대 글자수로 잘라 반환
    """
    try:
        s = (text or '').replace('\n', ' ').replace('\r', ' ')
        s = ' '.join(s.split())
        if not s:
            return ''

        # 문장 구분자를 기준으로 텍스트를 분리 (긍정형 후방탐색 사용)
        # 구분자도 결과에 포함
        parts = re.split(r'(?<=[.!?\u3002\uff01\uff1f])\s+', s)
        
        if not parts or len(parts) <= max_sentences:
            # 분리되지 않았거나 문장 수가 충분히 적으면 그대로 반환
            return s

        # 필요한 만큼의 문장만 합쳐서 반환
        head = ' '.join(parts[:max_sentences]).strip()
        return head

    except Exception:
        # 오류 발생 시, 원본 텍스트의 앞부분만 안전하게 잘라서 반환
        return (text or '')[:120]

# 영어권 꼬리 반복("How can I" 등) 제거용 경량 후처리
def _dedupe_tail_repeat_en(text: str) -> str:
    try:
        s = (text or '').strip()
        if not s:
            return s
        parts = re.split(r'(?<=[.!?\u3002\uff01\uff1f])\s+', s)
        if not parts:
            return s
        last = parts[-1]
        m = re.match(r'^([A-Za-z]+(?:\s+[A-Za-z]+){0,3})\b', last)
        if not m:
            return s
        prefix = m.group(1)
        if s.endswith(' ' + prefix) and last != prefix:
            return s[:-(1 + len(prefix))].rstrip()
        return s
    except Exception:
        return text

async def generate_audio_chunks(text, mode, language: str | None = None):
    # 언어에 따라 기본 보이스 선택
    if language is None:
        language = detect_language(text)
    speaker_wav = "./default_voice/Raika_ko.wav" if language == 'ko' else "./default_voice/Raika.wav" # Let Raika_TTS.py handle this
    if mode == 1: # 음소거
        return
    elif mode == 2: # 대사의 첫 두 문장
        # 경량 전처리 후 안전 문장 추출
        text = _head_sentences_safe(clean_text_for_tts_light(text), language or 'en', max_sentences=2)
    else:
        text = clean_text_for_tts_light(text)

    loop = asyncio.get_event_loop()
    # 혼합 텍스트(ko+en) 간단 판정 -> 구간 합성으로 품질/안정 향상
    def _is_mixed(s: str) -> bool:
        try:
            has_ko = any(0xAC00 <= ord(ch) <= 0xD7A3 for ch in s)
            has_en = any(('A' <= ch <= 'Z') or ('a' <= ch <= 'z') for ch in s)
            return has_ko and has_en
        except Exception:
            return False
    
    # 합성 (사용 시 상위에서 수행). 이 제너레이터는 현재 비활성 WS 경로용 헬퍼로 유지됨.
    # NOTE: 현재 /ws/tts 경로는 비활성화되어 wav_data를 생성하지 않으므로, 안전하게 반환합니다.
    return

@app.websocket("/ws/tts")
async def websocket_endpoint(websocket: WebSocket):
    # TTS 중복 이슈를 해결하기 위해 주석 처리: /ws/tts 오디오 스트리밍 비활성화, 립싱크 WS만 사용
    await websocket.accept()
    try:
        await websocket.send_json({"disabled": True, "reason": "tts_stream_disabled_for_dedup"})
    except Exception:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# =============================================================
# Live2D 립싱크용 경량 에너지(입 모양) 스트리밍 WebSocket
# - 오디오 파일/바이트 전송 없이, 실시간 에너지(0~1)를 전송
# - 서버단에서는 실제 음성 재생(play_wav)을 수행해 클라이언트와 동기 유지
# =============================================================

def _compute_envelope(values: np.ndarray, frame_size: int = 512, hop_size: int = 256) -> np.ndarray:
    """
    실수 파형(values, float32)로부터 RMS 기반 에너지(envelope)를 계산합니다.
    - 반환값 범위: 0.0 ~ 1.0 로 정규화
    - frame_size, hop_size는 22050Hz 기준 60~90fps 수준으로 맞춤
    """
    if values is None or len(values) == 0:
        return np.zeros(0, dtype=np.float32)

    # 안전 가드: NaN/Inf 제거
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    num_frames = 1 + max(0, (len(values) - frame_size) // hop_size)
    if num_frames <= 0:
        num_frames = 1
    envelope = np.zeros(num_frames, dtype=np.float32)

    # RMS 계산
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        frame = values[start:end]
        if len(frame) == 0:
            rms = 0.0
        else:
            rms = float(np.sqrt(np.mean(frame * frame)))
        envelope[i] = rms

    # 소프트 정규화(robust): 상위 분위수를 기준으로 스케일
    if envelope.size > 0:
        ref = float(np.quantile(envelope, 0.98)) or 1e-6
        envelope = np.clip(envelope / max(ref, 1e-6), 0.0, 1.0)

    return envelope


def _compute_features(values: np.ndarray, sample_rate: int = 16000, frame_size: int = 512, hop_size: int = 256):
    """
    음성 파형으로부터 두 가지 특징을 계산:
    - v: RMS 기반 에너지 (0~1 정규화) → 입 벌림(ParamMouthOpenY)
    - f: 스펙트럼 중심(centroid) 기반 고주파 비율 (0~1 정규화) → 입 모양(ParamMouthForm)
    """
    if values is None or len(values) == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    # 안전 가드 및 전처리
    x = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # 에너지(envelope)
    v = _compute_envelope(x, frame_size, hop_size)

    # 스펙트럼 특징(centroid 비율)
    num_frames = 1 + max(0, (len(x) - frame_size) // hop_size)
    if num_frames <= 0:
        num_frames = 1
    f = np.zeros(num_frames, dtype=np.float32)

    # 창 함수
    try:
        win = np.hanning(frame_size).astype(np.float32)
    except Exception:
        win = np.ones(frame_size, dtype=np.float32)

    nyquist = sample_rate / 2.0
    for i in range(num_frames):
        start = i * hop_size
        frame = x[start:start + frame_size]
        if len(frame) < frame_size:
            # 제로패딩
            pad = np.zeros(frame_size, dtype=np.float32)
            pad[:len(frame)] = frame
            frame = pad

        # 윈도잉 후 스펙트럼
        spec = np.fft.rfft(frame * win)
        mag = np.abs(spec).astype(np.float32)
        if mag.size <= 1:
            f[i] = 0.0
            continue

        # 주파수 벡터
        freqs = np.linspace(0, nyquist, mag.size, dtype=np.float32)
        # 스펙트럼 중심 (무게중심)
        denom = float(np.sum(mag)) or 1e-6
        centroid = float(np.sum(freqs * mag) / denom)
        # 0~1 정규화
        f[i] = np.clip(centroid / nyquist, 0.0, 1.0)

    # 부드러움 개선: 간단한 이동 평균(선택)
    if f.size > 3:
        f = np.convolve(f, np.ones(3, dtype=np.float32) / 3.0, mode='same')

    return v, f

@app.websocket("/ws/lipsync")
async def websocket_lipsync(websocket: WebSocket):
    """
    클라이언트로부터 {text, mode, language?}를 받고,
    - 서버단에서 음성 재생을 시작
    - 동시에 RMS 기반 에너지를 0~1 범위로 WebSocket JSON({"v": float}) 스트리밍
    - 전송 주기는 hop_size/16000 초, 종료 시 {"end": true}
    """
    await websocket.accept()
    logging.info("[LipSync] WebSocket connection accepted.")
    try:
        while True:
            # 입력 수신 -----------------------------------------------------
            logging.info("[LipSync] Waiting to receive text...")
            payload = await websocket.receive_text()
            logging.info(f"[LipSync] Received payload (len: {len(payload)})")

            data = json.loads(payload)
            logging.info("[LipSync] Payload parsed as JSON.")

            raw_text = data.get('text', '')
            mode = int(data.get('mode', 3))
            language = data.get('language')
            ex_override = data.get('exaggeration')
            logging.info(f"[LipSync] Mode: {mode}, Lang: {language}, Text: {raw_text[:50]}...")

            text = clean_text_for_tts_light(raw_text)
            logging.info("[LipSync] Text cleaned.") # 로깅 추가

            if not isinstance(text, str) or len(text.strip()) == 0:
                await websocket.send_json({"error": "empty_text"})
                continue

            if language is None:
                language = detect_language(text)
                logging.info(f"[LipSync] Language detected: {language}")


            # 합성 시작 전 즉시 ACK 전송 (문장 분리/합성 이전에 전송)
            try:
                await websocket.send_json({
                    "ack": True,
                    "lang": language,
                    "mode": mode,
                    "recv_len": len(text)
                })
            except Exception as _e:
                print(f"send ack error: {_e}")

            # 모드 전처리 --------------------------------------------------
            if mode == 1:  # 음소거
                logging.info("[LipSync] Mute mode. Skipping synthesis.")
                await websocket.send_json({"end": True})
                continue
            elif mode == 2:  # 첫 두 문장만
                logging.info("[LipSync] Brief mode. Extracting head sentences...")
                text = _head_sentences_safe(text, language or 'en', max_sentences=2)
                logging.info("[LipSync] Head sentences extracted.")
            # 초기 인삿말 반복 대사 제거는 애플리케이션 레이어에서 수행
       

            print(f"LipSync request - Language: {language}, Mode: {mode}")
            print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")

            # 위에서 ACK 전송 완료

            # 음성 합성 -----------------------------------------------------
            loop = asyncio.get_event_loop()
            speaker_wav = "./default_voice/Raika_ko.wav" if language == 'ko' else "./default_voice/Raika.wav" # Let Raika_TTS.py handle this
            
            # 혼합 텍스트는 구간 합성
            def _is_mixed(s: str) -> bool:
                try:
                    has_ko = any(0xAC00 <= ord(ch) <= 0xD7A3 for ch in s)
                    has_en = any(('A' <= ch <= 'Z') or ('a' <= ch <= 'z') for ch in s)
                    return has_ko and has_en
                except Exception:
                    return False

            # 합성 대기 중 주기적 keepalive 전송
            synth_done = asyncio.Event()

            async def _keepalive():
                try:
                    while not synth_done.is_set():
                        await websocket.send_json({"ka": True})
                        await asyncio.sleep(0.6)
                except Exception as _e:
                    # 전송 실패 시 조용히 종료 (상대가 끊었을 수 있음)
                    print(f"keepalive exit: {_e}")

            keepalive_task = asyncio.create_task(_keepalive())

            async def _synthesize():
                # 분할 합성 비활성화: 항상 단일 언어 합성만 사용
                # speaker_wav는 None으로 전달하여 Raika_TTS.py의 기본 로직을 따르도록 함
                # Exaggeration 일시 오버라이드 (환경변수 기반 파이프라인을 활용)
                prev_ex = os.environ.get('RAIKA_TTS_EXAGGERATION')
                try:
                    if ex_override is not None:
                        os.environ['RAIKA_TTS_EXAGGERATION'] = str(ex_override)
                    if language == 'ko':
                        return await loop.run_in_executor(None, text_to_speech, text, speaker_wav, 'ko')
                    if language == 'en':
                        return await loop.run_in_executor(None, text_to_speech, text, speaker_wav, 'en')
                    return await loop.run_in_executor(None, text_to_speech, text, speaker_wav, language)
                finally:
                    if ex_override is not None:
                        if prev_ex is None:
                            try:
                                del os.environ['RAIKA_TTS_EXAGGERATION']
                            except Exception:
                                pass
                        else:
                            os.environ['RAIKA_TTS_EXAGGERATION'] = prev_ex

            # 합성 타임아웃 가드
            try:
                timeout_sec = float(os.environ.get("RAIKA_TTS_TIMEOUT_SEC", "25"))
            except Exception:
                timeout_sec = 25.0

            try:
                wav_data = await asyncio.wait_for(_synthesize(), timeout=timeout_sec)
            except asyncio.TimeoutError:
                synth_done.set()
                try:
                    keepalive_task.cancel()
                except Exception:
                    pass
                await websocket.send_json({"error": "synthesis_timeout", "timeout_sec": timeout_sec})
                continue
            except Exception as _e:
                synth_done.set()
                try:
                    keepalive_task.cancel()
                except Exception:
                    pass
                await websocket.send_json({"error": "synthesis_failed", "detail": str(_e)[:200]})
                continue
            else:
                synth_done.set()
                try:
                    await keepalive_task
                except Exception:
                    pass
                # 합성 시간 측정 변수 기록
                try:
                    t_synth_end = time.perf_counter()
                except Exception:
                    t_synth_end = None
                try:
                    t_synth_start
                except NameError:
                    t_synth_start = t_synth_end

            # 합성된 오디오를 클라이언트 재생용으로 1회 전송 (옵션)
            try:
                send_audio_b64 = str(os.environ.get("RAIKA_TTS_SEND_AUDIO_B64", "1")).lower() not in ("0", "false", "no")
            except Exception:
                send_audio_b64 = True
            if send_audio_b64:
                try:
                    bio = io.BytesIO()
                    from scipy.io.wavfile import write as _wav_write
                    # float32 -> int16 스케일 후 WAV로 기록
                    x = np.asarray(wav_data, dtype=np.float32)
                    peak = float(np.max(np.abs(x))) if x.size > 0 else 1.0
                    if peak <= 0:
                        peak = 1.0
                    x16 = np.int16(np.clip(x / peak, -1.0, 1.0) * 32767)
                    _wav_write(bio, 16000, x16)
                    bio.seek(0)
                    b64 = base64.b64encode(bio.read()).decode('ascii')
                    await websocket.send_json({"audio_b64": b64, "sr": 16000})
                except Exception as _e:
                    print(f"audio_b64 send error: {_e}")

            # 오디오 재생을 병렬로 시작 (서버측 스피커)
            # 주: 영어 chipmunk 스타일은 합성 단계에서 이미 템포가 빨라짐.
            #    따라서 별도 재생 속도 보정을 하지 않음.
            speed_factor = 1.00
            # 립싱크 WS 경로를 단일 기준으로 사용하며, 서버측 재생은 여기서만 수행
            def _play():
                try:
                    play_wav(wav_data, speed_factor)
                except Exception as _e:
                    print(f"play_wav error: {_e}")

            # 주의: create_task 는 코루틴만 받음. executor는 Future를 반환하므로 그대로 실행만 하고 await하지 않음.
            try:
                loop.run_in_executor(None, _play)
            except Exception as _e:
                print(f"executor start error: {_e}")

            # 에너지 계산 및 스트리밍 --------------------------------------
            # - 16kHz 전제 (TTS 출력 및 play_wav 기본 샘플레이트)
            sample_rate = 16000
            frame_size = 512
            hop_size = 256
            t_feat_start = time.perf_counter()
            v, f = _compute_features(np.asarray(wav_data, dtype=np.float32), sample_rate, frame_size, hop_size)
            t_feat_end = time.perf_counter()

            # 시작 신호(예상 총 재생 시간 ms 포함: 재생 속도 고려)
            total_ms = int(len(wav_data) / float(sample_rate) * 1000 / max(speed_factor, 1e-6))
            await websocket.send_json({"start": True, "dur_ms": total_ms, "sr": sample_rate, "hs": hop_size})
            try:
                if t_synth_end is not None and t_synth_start is not None:
                    synth_ms = int((t_synth_end - t_synth_start) * 1000)
                else:
                    synth_ms = -1
                feat_ms = int((t_feat_end - t_feat_start) * 1000)
                print(f"Synth {synth_ms}ms | features {feat_ms}ms | frames={len(v)}")
            except Exception:
                pass

            # 실시간 간격으로 전송 (hop_size/sample_rate 초), 재생 속도 보정 포함
            interval = (hop_size / float(sample_rate)) / max(speed_factor, 1e-6)
            for idx in range(len(v)):
                vi = float(v[idx])
                fi = float(f[idx]) if idx < len(f) else 0.0
                try:
                    await websocket.send_json({"v": vi, "f": fi})
                except Exception as _e:
                    print(f"send_json error: {_e}")
                    break
                await asyncio.sleep(interval)

            # 종료 신호
            await websocket.send_json({"end": True})

    except Exception as e:
        print(f"Error in LipSync WebSocket: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


if __name__=="__main__":
    import uvicorn
    # Raika_TTS_Server.py를 직접 실행할 때 uvicorn 서버를 시작
    print("Starting Raika TTS Server directly...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
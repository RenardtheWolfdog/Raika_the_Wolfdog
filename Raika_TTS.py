import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
import sys
import types

# torchvision 안전 스텁: Transformers가 vision 경로를 만족할 때 순환 임포트 방지
if os.environ.get("TRANSFORMERS_NO_TORCHVISION", "1") == "1":
    if "torchvision" not in sys.modules:
        _tv = types.ModuleType("torchvision")
        try:
            _tv.__version__ = "0.0"
        except Exception:
            pass
        # 패키지로 인식되도록 __spec__ / __path__ 설정
        try:
            import importlib.machinery as _ilm
            _tv.__spec__ = _ilm.ModuleSpec("torchvision", loader=None)
        except Exception:
            _tv.__spec__ = None
        try:
            _tv.__path__ = []  # mark as namespace-like pkg
        except Exception:
            pass
        # 일부 버전에서 torchvision.extension 속성을 조회함
        class _TVExtension:
            _HAS_OPS = False
        _tv.extension = _TVExtension()
        sys.modules["torchvision"] = _tv
        # 하위 모듈 최소 스텁 (+ __spec__)
        try:
            _submods = {}
            for _name in ("transforms", "io", "ops"):
                _m = types.ModuleType(f"torchvision.{_name}")
                try:
                    _m.__spec__ = _ilm.ModuleSpec(f"torchvision.{_name}", loader=None)
                except Exception:
                    _m.__spec__ = None
                # transforms는 하위(v2 등) 서브모듈을 가질 수 있으므로 패키지로 표시
                if _name == "transforms":
                    try:
                        _m.__path__ = []
                    except Exception:
                        pass
                _submods[_name] = _m
                sys.modules[f"torchvision.{_name}"] = _m
        except Exception:
            pass
        # transforms.InterpolationMode 심볼 제공
        try:
            import enum as _enum
            _tr = sys.modules.get("torchvision.transforms")
            if _tr is not None and not hasattr(_tr, "InterpolationMode"):
                class _InterpolationMode(_enum.Enum):
                    NEAREST = 0
                    BILINEAR = 2
                    BICUBIC = 3
                    BOX = 4
                    HAMMING = 5
                    LANCZOS = 1
                _tr.InterpolationMode = _InterpolationMode
        except Exception:
            # 최악의 경우 단순 네임스페이스라도 제공
            try:
                _tr = sys.modules.get("torchvision.transforms")
                if _tr is not None and not hasattr(_tr, "InterpolationMode"):
                    class _InterpolationMode:  # type: ignore
                        NEAREST = 0
                        BILINEAR = 2
                        BICUBIC = 3
                        BOX = 4
                        HAMMING = 5
                        LANCZOS = 1
                    _tr.InterpolationMode = _InterpolationMode
            except Exception:
                pass
        # transforms.v2 및 v2.functional 스텁
        try:
            _tr = sys.modules.get("torchvision.transforms")
            if _tr is not None:
                _v2 = types.ModuleType("torchvision.transforms.v2")
                try:
                    _v2.__spec__ = _ilm.ModuleSpec("torchvision.transforms.v2", loader=None)
                except Exception:
                    _v2.__spec__ = None
                try:
                    _v2.__path__ = []
                except Exception:
                    pass
                sys.modules["torchvision.transforms.v2"] = _v2

                _v2f = types.ModuleType("torchvision.transforms.v2.functional")
                try:
                    _v2f.__spec__ = _ilm.ModuleSpec("torchvision.transforms.v2.functional", loader=None)
                except Exception:
                    _v2f.__spec__ = None
                sys.modules["torchvision.transforms.v2.functional"] = _v2f
        except Exception:
            pass

import torch
import random
import gc
import numpy as np
import io
# simpleaudio는 선택적 의존성으로 처리 (서버 무음/WS 립싱크 기준 시 불필요)
try:
    import simpleaudio as sa  # type: ignore
except ImportError:
    sa = None
from scipy.io.wavfile import write
from scipy import signal
import threading
import functools
import tempfile
from pathlib import Path
import soundfile as sf
try:
    import librosa  # type: ignore
except Exception:
    librosa = None
import io as _io

# Korean Phonemizer (IPA)
try:
    from phonemizer import phonemize
    _phonemizer_available = True
except ImportError:
    _phonemizer_available = False

# 디바이스 전역 결정 (프로세스 시작 시 1회)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 전역 TTS 인스턴스 (lazy init, 초기화 함수 호출 후 설정)
_tts = None
_tts_lock = threading.Lock()

def initialize_tts_model():
    """TTS 모델을 명시적으로 초기화함"""
    global _tts
    if _tts is None:
        with _tts_lock:
            if _tts is None:
                print("[TTS] Starting server, initializing TTS model...")
                torch.cuda.empty_cache(); gc.collect()
                # 후보들을 순차적으로 시도 (멀티링궐 → 단일 → 기타)
                last_err = None
                candidates = _import_chatterbox_candidates() # 함수 호출 위치 변경 가능
                print(f"[TTS] Found {len(candidates)} Chatterbox candidates: {', '.join(name for name, _, _ in candidates)}")
                for cls_name, Cls, repo_id in candidates:
                    try:
                        if repo_id:
                            print(f"[TTS] Trying {cls_name} with repo_id: {repo_id}")
                            _tts = cls.from_pretrained(repo_id, device=device)  # type: ignore
                        else:
                            print(f"[TTS] {cls_name} without repo_id")
                            _tts = cls.from_pretrained(device=device)  # type: ignore
                        print(f"[TTS] {cls_name} initialized successfully")
                        break
                    except Exception as e1:
                        last_err = e1
                        print(f"[TTS] {cls_name} initialization failed with repo_id: {repo_id} - {e1}")
                        # repo_id 없이도 재시도
                        try:
                            print(f"[TTS] {cls_name} retrying without repo_id")
                            _tts = Cls.from_pretrained(device=device)  # type: ignore
                            print(f"[TTS] {cls_name} initialized successfully")
                            break
                        except Exception as e2:
                            last_err = e2
                            print(f"[TTS] {cls_name} initialization failed without repo_id - {e2}")
                            continue
                if _tts is None:
                    # 최종 폴백: 무음 TTS
                    try:
                        msg = str(last_err)[:200] if last_err else "unknown"
                        print(f"[TTS] Chatterbox initialization failed, fallback to silent TTS: {msg}")
                    except Exception:
                        pass
                    class _SilentTTS:
                        def __init__(self):
                            self.sr = 16000
                        def generate(self, text: str, **kwargs):
                            dur_sec = max(0.2, min(3.0, len(text) / 40.0))
                            return np.zeros(int(self.sr * dur_sec), dtype=np.float32)
                    _tts = _SilentTTS()
                    print("[TTS] 무음 TTS 활성화됨.")
                # 모델 로딩 후 메모리 정리 (optional)
                torch.cuda.empty_cache(); gc.collect()

# Add language detection function
def detect_language(text):
    """
    Detect the language of the input text
    Returns language code: "en" for English, "ko" for Korean, etc.
    """
    # Check for Korean characters (Hangul Unicode range: AC00-D7A3)
    if any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in text):
        return "ko"
    # Default to English
    else:
        return "en"

def _import_chatterbox_candidates():
    """
    사용 가능한 Chatterbox 클래스 후보들을 우선순위대로 반환.
    각 항목은 (name, class_obj, repo_id_or_None) 형태.
    """
    candidates = []
    # 1) 표준 경로
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # type: ignore
        candidates.append(("ChatterboxMultilingualTTS", ChatterboxMultilingualTTS, "ResembleAI/chatterbox"))
    except Exception:
        pass
    try:
        from chatterbox.tts import ChatterboxTTS  # type: ignore
        candidates.append(("ChatterboxTTS", ChatterboxTTS, "ResembleAI/chatterbox"))
    except Exception:
        pass
    try:
        from chatterbox import Chatterbox  # type: ignore
        candidates.append(("Chatterbox", Chatterbox, None))
    except Exception:
        pass

    # 2) 대체 경로 (배포 변형/리네임 호환)
    try:
        from chatterbox_tts.mtl_tts import ChatterboxMultilingualTTS as _AltML  # type: ignore
        candidates.append(("ChatterboxMultilingualTTS", _AltML, "ResembleAI/chatterbox"))
    except Exception:
        pass
    try:
        from chatterbox_tts.tts import ChatterboxTTS as _AltTTS  # type: ignore
        candidates.append(("ChatterboxTTS", _AltTTS, "ResembleAI/chatterbox"))
    except Exception:
        pass

    # 중복 제거(클래스 기준)
    uniq = []
    seen = set()
    for name, cls, repo in candidates:
        key = getattr(cls, "__module__", "") + ":" + getattr(cls, "__name__", "")
        if key in seen:
            continue
        seen.add(key)
        uniq.append((name, cls, repo))
    return uniq


def _has_hangul(s: str) -> bool:
    return any(0xAC00 <= ord(ch) <= 0xD7A3 for ch in s)


def _has_latin(s: str) -> bool:
    return any(('A' <= ch <= 'Z') or ('a' <= ch <= 'z') for ch in s)


def _split_text_by_language(s: str) -> list[tuple[str, str]]:
    """
    간단 분할: 한글 연속 구간과 비-한글(라틴/기타) 구간을 분리.
    반환: [(segment_text, 'ko'|'en')]
    """
    import re
    parts = re.findall(r"[\uAC00-\uD7A3]+|[^\uAC00-\uD7A3]+", s)
    segments: list[tuple[str, str]] = []
    for p in parts:
        p_stripped = p.strip()
        if not p_stripped:
            continue
        lang = "ko" if _has_hangul(p_stripped) else "en"
        segments.append((p_stripped, lang))
    return segments

try:
    GLOBAL_SEED = int(os.environ.get("RAIKA_TTS_SEED", "1234"))
except Exception:
    GLOBAL_SEED = 1234

# 샘플링 스텝(속도/품질 트레이드오프) 환경변수
try:
    STEPS_KO_DEFAULT = int(os.environ.get("RAIKA_TTS_STEPS_KO", "100"))
except Exception:
    STEPS_KO_DEFAULT = 100
try:
    STEPS_EN_DEFAULT = int(os.environ.get("RAIKA_TTS_STEPS_EN", "60"))
except Exception:
    STEPS_EN_DEFAULT = 60
try:
    STEPS_OTHER_DEFAULT = int(os.environ.get("RAIKA_TTS_STEPS_OTHER", "90"))
except Exception:
    STEPS_OTHER_DEFAULT = 90

def _set_global_seed(seed: int) -> None:
    try:
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
    except Exception:
        pass

def _get_env_prompt_text(language: str) -> str | None:
    return None


def _get_sidecar_prompt_text(original_wav_path: str) -> str | None:
    return None


def _resolve_clone_prompt(language: str, original_wav_path: str | None) -> tuple[str | None, str | None]:
    return None, None


def _preprocess_korean_for_tts(text: str) -> str:
    """
    Apply Phonemizer for Korean text to convert it to IPA phonemes.
    """
    if not _phonemizer_available:
        print("[TTS] phonemizer not installed, skipping Korean Phonemizer.")
        return text
    try:
        # Resolve espeak-ng executable/library/data paths to avoid PATH issues on Windows
        default_roots = [
            r"C:\\Program Files\\eSpeak NG",
            r"C:\\Program Files (x86)\\eSpeak NG",
        ]
        espeak_bin = os.environ.get("PHONEMIZER_ESPEAK_PATH")
        espeak_lib = os.environ.get("PHONEMIZER_ESPEAK_LIBRARY")
        espeak_data = os.environ.get("ESPEAKNG_DATA_PATH") or os.environ.get("ESPEAK_DATA_PATH")

        for root in default_roots:
            if (not espeak_bin or not os.path.exists(espeak_bin)):
                cand = os.path.join(root, "espeak-ng.exe")
                if os.path.exists(cand):
                    espeak_bin = cand
            if (not espeak_lib or not os.path.exists(espeak_lib)):
                cand = os.path.join(root, "libespeak-ng.dll")
                if os.path.exists(cand):
                    espeak_lib = cand
            if (not espeak_data or not os.path.exists(espeak_data)):
                cand = os.path.join(root, "espeak-ng-data")
                if os.path.exists(cand):
                    espeak_data = cand

        if espeak_bin and os.path.exists(espeak_bin):
            os.environ["PHONEMIZER_ESPEAK_PATH"] = espeak_bin
            print(f"[TTS] Using espeak-ng exe: {espeak_bin}")
        else:
            print("[TTS] espeak-ng executable not found. Set PHONEMIZER_ESPEAK_PATH to espeak-ng.exe.")
        if espeak_lib and os.path.exists(espeak_lib):
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_lib
            print(f"[TTS] Using espeak-ng dll:  {espeak_lib}")
        if espeak_data and os.path.exists(espeak_data):
            os.environ["ESPEAKNG_DATA_PATH"] = espeak_data
            os.environ["ESPEAK_DATA_PATH"] = espeak_data  # some tools read this
            print(f"[TTS] Using espeak-ng data: {espeak_data}")

        # Convert to IPA phonemes (e.g., "안녕하세요" -> "annjʌŋɦasejo")
        phonemes = phonemize(text, language='ko', backend='espeak', strip=True)
        print(f"[TTS] Korean Phonemizer: '{text}' -> '{phonemes}'")
        return phonemes
    except Exception as e:
        print(f"[TTS] Korean Phonemizer failed: {e}. Verify espeak-ng exe/dll/data paths.")
        return text


def _get_tts():
    """Return the initialized TTS model instance."""
    # initialize_tts_model() 함수가 서버 시작 시 호출되므로,
    # 이 함수는 단순히 _tts 변수를 반환.
    # 만약 초기화 실패 시 무음 TTS가 _tts에 할당.
    if _tts is None:
         # 이론상 서버 시작 시 초기화되므로 이 경우는 발생하지 않아야 하지만, 안전장치로 추가
         print("[TTS] Warning: _get_tts() called but TTS model is not initialized. Initializing now...")
         initialize_tts_model()
    return _tts


def _try_init_multilingual() -> object | None:
    """
    멀티링궐 TTS를 강제로 초기화 시도. 성공 시 인스턴스, 실패 시 None.
    """
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # type: ignore
    except Exception:
        return None
    inst = None
    # 멀티링궐 레포 우선 시도 → 기본 레포 → device만
    for args in (
        {"repo_id": "ResembleAI/chatterbox-multilingual", "device": device},
        {"repo_id": "ResembleAI/chatterbox", "device": device},
        {"device": device},
    ):
        try:
            if "repo_id" in args:
                inst = ChatterboxMultilingualTTS.from_pretrained(args["repo_id"], device=args["device"])  # type: ignore
            else:
                inst = ChatterboxMultilingualTTS.from_pretrained(device=args["device"])  # type: ignore
            break
        except Exception:
            inst = None
            continue
    return inst

def _extract_audio_array(result) -> np.ndarray | None:
    """
    Chatterbox generate 결과로부터 오디오 파형을 최대한 유연하게 추출.
    지원: numpy array, torch.Tensor, (tuple/list) 내 배열, dict 키(audio|wav|waveform|samples|audio_values),
         bytes/bytearray(BYTES WAV), 파일 경로(str/Path).
    반환: float32 1D np.ndarray or None
    """
    try:
        import torch as _torch  # type: ignore
    except Exception:
        _torch = None

    def _to_np(x):
        try:
            if _torch is not None and isinstance(x, _torch.Tensor):
                x = x.detach().cpu().numpy()
        except Exception:
            pass
        try:
            arr = np.asarray(x, dtype=np.float32)
        except Exception:
            return None
        if arr.ndim == 0:
            return None
        if arr.ndim > 1:
            # (C, T) 또는 (T, C) 형태를 1채널로 축소
            try:
                if arr.shape[0] == 1:
                    arr = arr[0]
                elif arr.shape[-1] == 1:
                    arr = arr[..., 0]
                else:
                    # 여러 채널이면 평균
                    arr = np.mean(arr, axis=-1)
            except Exception:
                arr = arr.reshape(-1)
        return arr.astype(np.float32)

    # 직접 배열/텐서
    arr = _to_np(result)
    if arr is not None:
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # dict
    if isinstance(result, dict):
        for key in ("audio", "wav", "waveform", "samples", "audio_values", "audio_tensor"):
            if key in result:
                arr = _extract_audio_array(result[key])
                if isinstance(arr, np.ndarray) and arr.size > 0:
                    return arr
        return None

    # tuple/list: 가장 긴 파형 선택
    if isinstance(result, (list, tuple)):
        best = None
        best_len = 0
        for item in result:
            arr = _extract_audio_array(item)
            if isinstance(arr, np.ndarray) and arr.size > best_len:
                best = arr
                best_len = arr.size
        return best

    # bytes/bytearray: WAV 바이트로 가정
    if isinstance(result, (bytes, bytearray, _io.BytesIO)):
        try:
            bio = result if isinstance(result, _io.BytesIO) else _io.BytesIO(result)
            data, _sr = sf.read(bio, dtype='float32', always_2d=False)
            return _to_np(data)
        except Exception:
            return None

    # 파일 경로
    if isinstance(result, (str, Path)):
        try:
            p = Path(result)
            if p.exists():
                data, _sr = sf.read(str(p), dtype='float32', always_2d=False)
                return _to_np(data)
        except Exception:
            return None

    return None


def _apply_preemphasis(x: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    # 유지: 내부 간단 DSP 보정용(튜닝 후처리 아님)
    if x.size == 0:
        return x.astype(np.float32)
    y = np.copy(x).astype(np.float32)
    y[1:] = y[1:] - alpha * y[:-1]
    return y


def _apply_biquad_shelving(x: np.ndarray, sr: int, gain_db: float, cutoff_hz: float, high: bool, S: float = 0.6) -> np.ndarray:
    return x.astype(np.float32)


def _apply_peaking_eq(x: np.ndarray, sr: int, gain_db: float, center_hz: float, Q: float = 0.8) -> np.ndarray:
    return x.astype(np.float32)


def _apply_ko_lively_style(x: np.ndarray, sr: int = 16000) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


@functools.lru_cache(maxsize=8)
def _prepare_prompt_wav_cached(src_path: str) -> str:
    """
    스피커 프롬프트 WAV를 16k 모노로 강제 변환해 임시 캐시에 저장 후 경로 반환.
    원본이 이미 16k 모노면 원본 경로를 그대로 반환.
    변환 실패 시 원본 경로 반환.
    """
    try:
        if not os.path.exists(src_path):
            return src_path
        data, sr = sf.read(src_path, dtype='float32', always_2d=True)
        # 모노 다운믹스
        if data.shape[1] > 1:
            data = np.mean(data, axis=1, keepdims=True).astype(np.float32)
        # 16k 리샘플
        if sr != 16000:
            target_len = max(1, int(round(data.shape[0] * 16000 / float(sr))))
            data = signal.resample(data[:, 0], target_len).astype(np.float32)[:, None]
            sr = 16000
        # 이미 조건 만족하면 원본 사용
        if sr == 16000 and data.shape[1] == 1:
            return src_path
        # 임시 파일 저장
        base = os.path.basename(src_path)
        name, _ = os.path.splitext(base)
        tmp_dir = os.path.join(tempfile.gettempdir(), "raika_tts_prompts")
        os.makedirs(tmp_dir, exist_ok=True)
        out_path = os.path.join(tmp_dir, f"{name}_16k_mono.wav")
        sf.write(out_path, data.astype(np.float32).reshape(-1), 16000)
        return out_path
    except Exception:
        return src_path

def text_to_speech(text, speaker_wav, language=None, auto_play: bool | None = None):
    """
    Chatterbox로 텍스트를 음성으로 변환합니다.
    반환값: float32 numpy waveform (16kHz)
    """
    _set_global_seed(GLOBAL_SEED)
    model = _get_tts()
    # 한국어 명료도 확보: language_id 미지원 모델이면 멀티링궐 재초기화 시도
    try:
        import inspect as _inspect
        gen_params_boot = set(_inspect.signature(model.generate).parameters.keys())
    except Exception:
        gen_params_boot = set()
    if language == 'ko':
        need_reinit = False
        if not ({"language_id", "language", "lang"} & gen_params_boot):
            need_reinit = True
        try:
            cls_name = getattr(model.__class__, "__name__", "")
            if "Multilingual" not in cls_name:
                need_reinit = True
        except Exception:
            pass
        if need_reinit:
            reinit = _try_init_multilingual()
            if reinit is not None:
                model = reinit
                print("[TTS] Reinitialized to multilingual model for Korean support.")

    # 언어 자동 감지
    if language is None:
        language = detect_language(text)

    # Phonemizer 라이브러리 (한국어 발음 처리 관련) 사용 logic & speaker selection
    processed_text = text
    effective_language = language
    use_phonemizer_for_ko = False

    if language == 'ko':
        use_phonemizer_for_ko = os.environ.get("RAIKA_TTS_KO_USE_PHONEMIZER", "0") == "1"
        if use_phonemizer_for_ko:
            processed_text = _preprocess_korean_for_tts(text)
            effective_language = 'en'  # Treat IPA phonemes as a language the 'en' model understands
            print(f"[TTS] Phonemized Korean will be synthesized as language='en'")

    # Speaker selection: When phonemizing Korean, use the English voice for best results
    if language == 'ko' and use_phonemizer_for_ko:
        default_speaker = "./default_voice/Raika.wav"
    elif language == 'ko':
        default_speaker = "./default_voice/Raika_ko.wav"
    else:
        default_speaker = "./default_voice/Raika.wav"

    # 참조 프롬프트는 16k 모노 캐시본을 사용
    prompt_wav_src = speaker_wav if speaker_wav else default_speaker
    prompt_wav_path = _prepare_prompt_wav_cached(prompt_wav_src)

    print(f"Chatterbox TTS | lang={language} (effective: {effective_language}) | default_speaker={prompt_wav_path}")


    # 스피커 프롬프트는 경로 문자열을 기본으로 사용 (모델 호환성 우선)
    try:
        if not os.path.exists(prompt_wav_path):
            print(f"[TTS] speaker wav not found: {prompt_wav_path}")
    except Exception as e:
        print(f"[TTS] speaker wav check error: {e}")

    # 언어별 빠른 샘플링 스텝 설정
    if language == 'ko':
        steps_target = int(STEPS_KO_DEFAULT)
    elif language == 'en':
        steps_target = int(STEPS_EN_DEFAULT)
    else:
        steps_target = int(STEPS_OTHER_DEFAULT)

    # Chatterbox 호출 (generate 시그니처를 읽어 안전한 인자만 전달)
    import inspect as _inspect
    wav = None
    last_err = None
    try:
        gen_params = set(_inspect.signature(model.generate).parameters.keys())
    except Exception:
        gen_params = set()
    try:
        print(f"[TTS] generate() accepts params: {sorted(list(gen_params))[:20]}")
    except Exception:
        pass

    base_kwargs = {"text": processed_text}
    # 스피커/음성 클로당 파라미터 후보 중 지원되는 첫 키에 경로 문자열 전달
    # 단, 한국어에서 참조 프롬프트를 끄는 토글 지원: RAIKA_TTS_KO_DISABLE_PROMPT=1
    speaker_key_used = None
    ko_disable_prompt = os.environ.get("RAIKA_TTS_KO_DISABLE_PROMPT", "0") == "1"
    if not (language == 'ko' and ko_disable_prompt):
        for sk in (
            "audio_prompt_path",  # 공식 문서 명칭 우선
            "speaker_wav",
            "speaker_prompt",
            "prompt_wav",
            "reference_wav",
            "voice_sample",
            "speaker",
            "voice",
            "reference_audio",
            "clone_wav",
        ):
            if sk in gen_params:
                base_kwargs[sk] = prompt_wav_path
                speaker_key_used = sk
                break
        if speaker_key_used is not None:
            try:
                print(f"[TTS] using speaker key: {speaker_key_used} -> {prompt_wav_path}")
            except Exception:
                pass
    else:
        print("[TTS] ko_disable_prompt=1 → 참조 보이스 미사용")
            
    # 스텝 파라미터 후보 중 지원되는 첫 키 적용
    for sk in ("steps", "num_inference_steps", "inference_steps"):
        if sk in gen_params:
            base_kwargs[sk] = steps_target
            break
    # 표현력 파라미터 (있는 경우에만)
    # 언어별 기본값 허용: 한국어에선 cfg 낮춤, exaggeration은 중간값
    try:
        _cfg_env = float(os.environ.get("RAIKA_TTS_CFG", "0.5" if language == 'ko' else "0.25"))
    except Exception:
        _cfg_env = 0.5 if language == 'ko' else 0.25
    try:
        _ex_env = float(os.environ.get("RAIKA_TTS_EXAGGERATION", "0.5" if language == 'ko' else "0.35"))
    except Exception:
        _ex_env = 0.5 if language == 'ko' else 0.35
    try:
        _temp_env = float(os.environ.get("RAIKA_TTS_TEMPERATURE", "0.8" if language == 'ko' else "0.9"))
    except Exception:
        _temp_env = 0.8 if language == 'ko' else 0.9
    if "cfg" in gen_params:
        base_kwargs["cfg"] = _cfg_env
    if "cfg_weight" in gen_params:
        base_kwargs["cfg_weight"] = _cfg_env
    if "exaggeration" in gen_params:
        base_kwargs["exaggeration"] = _ex_env
    if "temperature" in gen_params:
        base_kwargs["temperature"] = _temp_env
    # 언어 파라미터 (있는 경우에만) — 지원되는 모든 키에 ko를 설정
    if isinstance(effective_language, str):
        for lk in ("language_id", "language", "lang"):
            if lk in gen_params:
                base_kwargs[lk] = effective_language

    # 1차: 안전 인자만으로 호출
    try:
        wav = model.generate(**base_kwargs)
    except Exception as e:
        last_err = e
        wav = None

    # 2차: 오디오 플래그가 있는 경우 활성화하여 재시도
    if wav is None:
        try:
            audio_flags = {}
            if "return_audio" in gen_params:
                audio_flags["return_audio"] = True
            if "return_wav" in gen_params:
                audio_flags["return_wav"] = True
            if "output" in gen_params:
                audio_flags["output"] = "audio"
            if "output_type" in gen_params:
                audio_flags["output_type"] = "audio"
            if audio_flags:
                wav = model.generate(**{**base_kwargs, **audio_flags})
        except Exception as e:
            last_err = e
            wav = None

    # 3차: 완전 최소 인자(text만)
    if wav is None:
        try:
            wav = model.generate(text=text)
        except Exception as e2:
            last_err = e2
            wav = None

    # 3.5차: 언어 토큰 프리픽스 폴백 (언어 파라미터가 없는 모델 전용)
    if wav is not None and language == 'ko' and not ({"language_id", "language", "lang"} & gen_params):
        try:
            use_token_fb = os.environ.get("RAIKA_TTS_KO_TOKEN_FALLBACK", "1") == "1"
        except Exception:
            use_token_fb = True
        if use_token_fb:
            token_prefixes = [
                "<|lang:ko|> ",
                "<|ko|> ",
                "[KO] ",
                "<|KOREAN|> ",
                "KOREAN: ",
                "한국어: ",
            ]
            for tp in token_prefixes:
                try:
                    fb_kwargs = {k: v for k, v in base_kwargs.items() if k not in (
                        "audio_prompt_path", "speaker_wav", "speaker_prompt", "prompt_wav",
                        "reference_wav", "voice_sample", "speaker", "voice", "reference_audio", "clone_wav",
                    )}
                    fb_kwargs["text"] = tp + processed_text
                    wav2 = model.generate(**fb_kwargs)
                    if wav2 is not None:
                        wav = wav2
                        print(f"[TTS] Applied KO token prefix: {tp.strip()}")
                        break
                except Exception:
                    continue

    # 4차: 한국어 전용 보수적 폴백 — 프롬프트 제거, 낮은 cfg/과장, 더 많은 스텝
    if wav is None and language == 'ko':
        try:
            fallback_kwargs = {"text": text}
            for lk in ("language_id", "language", "lang"):
                if lk in gen_params:
                    fallback_kwargs[lk] = "ko"
            for sk in ("steps", "num_inference_steps", "inference_steps"):
                if sk in gen_params:
                    fallback_kwargs[sk] = max(steps_target, 120)
                    break
            if "cfg" in gen_params:
                fallback_kwargs["cfg"] = 0.1
            if "exaggeration" in gen_params:
                fallback_kwargs["exaggeration"] = 0.25
            wav = model.generate(**fallback_kwargs)
            print("[TTS] Fallback KO: prompt off, steps↑, cfg↓, exaggeration↓")
        except Exception as e3:
            last_err = e3
            wav = None

    # 생성물 타입 로깅
    try:
        print(f"[TTS] generate type={type(wav)}")
        if isinstance(wav, dict):
            print(f"[TTS] generate dict keys={list(wav.keys())[:8]}")
        try:
            import torch as _torch
            if isinstance(wav, _torch.Tensor):
                print(f"[TTS] tensor shape={tuple(wav.shape)} dtype={wav.dtype}")
        except Exception:
            pass
    except Exception:
        pass

    if wav is None:
        # 무음 폴백 (init 시 SilentTTS가 생성되어 있을 수 있음)
        try:
            print(f"[TTS] generate failed. last_err={str(last_err)[:200] if last_err else 'None'}")
        except Exception:
            pass
        dur_sec = max(0.2, min(3.0, len(text) / 40.0))
        arr = np.zeros(int(16000 * dur_sec), dtype=np.float32)
    else:
        arr = _extract_audio_array(wav)
        if arr is None or getattr(arr, "size", 0) == 0:
            arr = np.zeros(1, dtype=np.float32)
        # 너무 짧으면 가청 확인용 비프 대체
        if getattr(arr, "size", 0) < 1600:
            try:
                print(f"[TTS] too short ({arr.size}), replace with beep")
            except Exception:
                pass
            t = np.linspace(0, 0.4, int(16000 * 0.4), endpoint=False, dtype=np.float32)
            arr = (0.15 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    # arr는 위에서 _extract_audio_array로 추출한 결과를 그대로 사용

    # NaN/Inf 정리
    try:
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    except Exception:
        pass

    # 샘플레이트를 16kHz로 강제 맞춤
    try:
        model_sr = int(getattr(model, "sr", getattr(model, "sample_rate", 16000)))
    except Exception:
        model_sr = 16000
    if model_sr != 16000 and arr.size > 0:
        new_len = max(1, int(round(len(arr) * 16000 / float(model_sr))))
        arr = signal.resample(arr, new_len).astype(np.float32)

    # 합성 결과 로그
    try:
        _peak = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
        print(f"[TTS] out_len={len(arr)} peak={_peak:.6f}")
    except Exception:
        pass

    # 무음/초저레벨 방지: 재시도 또는 간단 정규화
    try:
        peak = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
    except Exception:
        peak = 0.0

    if arr.size == 0 or peak < 1e-6:
        # 마지막 재시도: 프롬프트 제거 + 더 적은 스텝으로 속도 우선 재시도
        retry_wav = None
        step_keys = ("steps", "num_inference_steps", "inference_steps")
        for sk in step_keys:
            try:
                retry_wav = model.generate(text=text, **{sk: max(40, steps_target // 2)})
                break
            except Exception:
                continue
        if retry_wav is not None:
            try:
                import torch as _torch
                if isinstance(retry_wav, _torch.Tensor):
                    retry_wav = retry_wav.detach().cpu().numpy()
            except Exception:
                pass
            arr2 = np.asarray(retry_wav, dtype=np.float32)
            if arr2.ndim > 1:
                arr2 = arr2[..., 0].astype(np.float32)
            try:
                model_sr = int(getattr(model, "sr", getattr(model, "sample_rate", 16000)))
            except Exception:
                model_sr = 16000
            if model_sr != 16000 and arr2.size > 0:
                new_len = max(1, int(round(len(arr2) * 16000 / float(model_sr))))
                arr2 = signal.resample(arr2, new_len).astype(np.float32)
            arr = arr2
            try:
                peak = float(np.max(np.abs(arr))) if arr.size > 0 else 0.0
            except Exception:
                peak = 0.0

    # 여전히 무음이면 디버그 파일 저장 후 소극적 톤 생성(짧은 사인파)
    if arr.size == 0 or float(np.max(np.abs(arr)) if arr.size > 0 else 0.0) < 1e-6:
        # 디버그 파일 저장 (무음)
        try:
            dbg = os.path.join(os.getcwd(), "debug_tts_silent.wav")
            sf.write(dbg, np.zeros(16000, dtype=np.float32), 16000)
        except Exception:
            pass
        # 0.4초 440Hz 사인파 (사용자 확인용)
        t = np.linspace(0, 0.4, int(16000 * 0.4), endpoint=False, dtype=np.float32)
        arr = (0.15 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    # 출력 정규화 + 선택적 피치 시프트(여성 톤 유지)
    peak = float(np.max(np.abs(arr))) if arr.size > 0 else 1.0
    if peak > 0:
        arr = (arr / peak).astype(np.float32)
    # RAIKA_TTS_PITCH_SHIFT_SEMITONES_KO / _EN (정수 또는 실수)
    try:
        if language == 'ko':
            ps = float(os.environ.get("RAIKA_TTS_PITCH_SHIFT_SEMITONES_KO", "0"))
        elif language == 'en':
            ps = float(os.environ.get("RAIKA_TTS_PITCH_SHIFT_SEMITONES_EN", "0"))
        else:
            ps = float(os.environ.get("RAIKA_TTS_PITCH_SHIFT_SEMITONES_OTHER", "0"))
    except Exception:
        ps = 0.0
    if abs(ps) > 1e-3 and librosa is not None and arr.size > 0:
        try:
            # 16kHz 기준 피치 시프트 적용
            arr = librosa.effects.pitch_shift(arr.astype(np.float32), sr=16000, n_steps=ps).astype(np.float32)
        except Exception:
            pass

    # 항상 결과 파일 저장 (디버그)
    try:
        out_dbg = os.path.join(os.getcwd(), "debug_tts_last.wav")
        sf.write(out_dbg, arr.astype(np.float32), 16000)
    except Exception:
        pass

    # 자동 재생 (기본 비활성, 환경변수로 제어 가능)
    try:
        if auto_play is None:
            auto_play = os.environ.get("RAIKA_TTS_AUTO_PLAY", "0") == "1"
    except Exception:
        auto_play = False

    if auto_play:
        try:
            try:
                play_speed = float(os.environ.get("RAIKA_TTS_PLAY_SPEED", "1.0"))
            except Exception:
                play_speed = 1.0
            try:
                preserve = os.environ.get("RAIKA_TTS_PLAY_PRESERVE_PITCH", "0") == "1"
            except Exception:
                preserve = False
            play_wav(arr, speed_factor=play_speed, sample_rate=16000, preserve_pitch=preserve)
        except Exception:
            pass

    return arr.astype(np.float32)


def text_to_speech_mixed(text, speaker_wav=None, auto_play: bool | None = None):
    """
    혼합 텍스트(한글/영문 혼재)를 언어 구간별로 분할해 순차 합성 후 연결.
    반환: float32 numpy waveform(16kHz)
    """
    segments = _split_text_by_language(text)
    if not segments:
        return text_to_speech(text, speaker_wav=speaker_wav, language=None)

    waves = []
    for seg_text, lang in segments:
        # 아주 짧은 감탄/인사말은 명료도 좋으므로 약간 빠르게
        is_short = len(seg_text) <= 12
        # 구간 합성 시에는 즉시 재생을 비활성화하여 전체 연결 후 한 번만 재생
        wav = text_to_speech(seg_text, speaker_wav=speaker_wav, language=lang, auto_play=False)
        if is_short:
            # 재생 전에 약간의 여백을 둬서 연결 왜곡 방지
            waves.append(wav)
        else:
            waves.append(wav)

    if not waves:
        return np.zeros(0, dtype=np.float32)

    # 단순 연결 + 경계 페이드인/아웃으로 클릭/웅웅 억제
    def _xfade_concat(chunks: list[np.ndarray], sr: int = 16000, xfade_ms: float = 20.0) -> np.ndarray:
        if len(chunks) == 1:
            return chunks[0].astype(np.float32)
        xfade = int(sr * (xfade_ms / 1000.0))
        out = chunks[0].astype(np.float32)
        for nxt in chunks[1:]:
            nxt = nxt.astype(np.float32)
            if xfade > 0 and len(out) > xfade and len(nxt) > xfade:
                fade_out = np.linspace(1.0, 0.0, xfade, dtype=np.float32)
                fade_in = np.linspace(0.0, 1.0, xfade, dtype=np.float32)
                out[-xfade:] *= fade_out
                nxt[:xfade] *= fade_in
                out = np.concatenate([out, nxt], axis=0)
            else:
                out = np.concatenate([out, nxt], axis=0)
        return out

    out = _xfade_concat(waves, 16000, 20.0)

    # 자동 재생 (기본 비활성, 환경변수로 제어 가능)
    try:
        if auto_play is None:
            auto_play = os.environ.get("RAIKA_TTS_AUTO_PLAY", "0") == "1"
    except Exception:
        auto_play = False

    if auto_play:
        try:
            try:
                play_speed = float(os.environ.get("RAIKA_TTS_PLAY_SPEED", "1.0"))
            except Exception:
                play_speed = 1.0
            try:
                preserve = os.environ.get("RAIKA_TTS_PLAY_PRESERVE_PITCH", "0") == "1"
            except Exception:
                preserve = False
            play_wav(out, speed_factor=play_speed, sample_rate=16000, preserve_pitch=preserve)
        except Exception:
            pass

    return out

def _apply_simple_denoise(x: np.ndarray, sr: int) -> np.ndarray:
    # 고역/저역의 과도한 잡음을 줄이기 위한 간단한 하이패스 필터 + 소프트 클리핑
    # DC 오프셋 제거
    x = x - float(np.mean(x)) if x.size > 0 else x
    try:
        # 120Hz 하이패스 (웅웅거림 억제)
        cutoff = min(120.0, sr * 0.45)
        b, a = signal.butter(2, cutoff / (sr * 0.5), btype='highpass')
        y = signal.lfilter(b, a, x)
    except Exception:
        y = x
    # 소프트 클리핑으로 찌그러짐 감소
    gain = 1.0
    y = np.tanh(gain * y) / np.tanh(gain)
    return y.astype(np.float32)


def play_wav(wav_data, speed_factor=1.0, sample_rate: int = 16000, preserve_pitch: bool = False):
    # Convert list to numpy array
    wav_data_np = np.array(wav_data, dtype=np.float32)

    # 피치 보존 타임스트레치(가능하면 librosa), 아니면 단순 리샘플
    if preserve_pitch and librosa is not None and abs(speed_factor - 1.0) > 1e-3:
        try:
            resampled_data = librosa.effects.time_stretch(wav_data_np, rate=1.0 / speed_factor)
        except Exception:
            number_of_samples = round(len(wav_data_np) / speed_factor)
            resampled_data = signal.resample(wav_data_np, number_of_samples)
    else:
        number_of_samples = round(len(wav_data_np) / speed_factor)
        resampled_data = signal.resample(wav_data_np, number_of_samples)

    # 간단한 노이즈 저감
    resampled_data = _apply_simple_denoise(resampled_data, sample_rate)

    # Normalize the WAV data
    peak = np.max(np.abs(resampled_data)) if len(resampled_data) > 0 else 1.0
    if peak == 0:
        peak = 1.0
    resampled_data = np.int16((resampled_data / peak) * 32767)

    # Create a BytesIO stream
    byte_io = io.BytesIO()

    # Write the WAV data to the stream (16kHz 기본)
    write(byte_io, sample_rate, resampled_data)
    byte_io.seek(0)

    # Play the audio using simpleaudio (선택적)
    if sa is None:
        # simpleaudio 미설치 시에도 마지막 재생 데이터를 파일로 저장
        try:
            out_dbg = os.path.join(os.getcwd(), "debug_tts_last_play.wav")
            write(out_dbg, sample_rate, resampled_data)
        except Exception:
            pass
        return
    play_obj = sa.WaveObject(byte_io.read(), 1, 2, sample_rate).play()
    play_obj.wait_done()


def test_tts_main():
    """Chatterbox TTS 간단 테스트(기본 보이스 자동 선택)."""
    korean_text = "안녕하세요, 저는 Raika the wolfdog 입니다. 만나서 반갑습니다!"
    english_text = "Hello, I am Raika. Nice to meet you!"

    # 테스트 동안 Exaggeration = 1.1 임시 적용
    _prev_ex = os.environ.get("RAIKA_TTS_EXAGGERATION")
    os.environ["RAIKA_TTS_EXAGGERATION"] = "1.1"
    try:
        # print("[Test] Korean TTS (exaggeration=0.5)...")
        # ko_wav = text_to_speech(korean_text, speaker_wav=None, language="ko")
        # play_wav(ko_wav, speed_factor=1.00, sample_rate=16000, preserve_pitch=False)

        # print("[Test] English TTS (exaggeration=0.5)...")
        # en_wav = text_to_speech(english_text, speaker_wav=None, language="en")
        # play_wav(en_wav, speed_factor=1.00, sample_rate=16000, preserve_pitch=False)

        print("[Test] Korean TTS (exaggeration=1.1)...")
        ko_wav = text_to_speech(korean_text, speaker_wav=None, language="ko")
        play_wav(ko_wav, speed_factor=1.00, sample_rate=16000, preserve_pitch=False)

        print("[Test] English TTS (exaggeration=1.1)...")
        en_wav = text_to_speech(english_text, speaker_wav=None, language="en")
        play_wav(en_wav, speed_factor=1.00, sample_rate=16000, preserve_pitch=False)
    finally:
        if _prev_ex is None:
            try:
                del os.environ["RAIKA_TTS_EXAGGERATION"]
            except Exception:
                pass
        else:
            os.environ["RAIKA_TTS_EXAGGERATION"] = _prev_ex

if __name__ == "__main__":
    test_tts_main()
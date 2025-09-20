from .store import global_store
import re
from typing import Optional, Dict

from vosk import Model, KaldiRecognizer
import wave, json, os

from pydub import AudioSegment
import io
import subprocess

# -----------------------------
# Audio 전처리 (BytesIO → 16kHz mono wav)
# -----------------------------
def prepare_audio_for_vosk(org_file_path) -> io.BytesIO:
    """
    Streamlit BytesIO / UploadedFile → Vosk에서 쓸 수 있는 16kHz mono wav로 변환
    """
    process = subprocess.run(
        [
            "ffmpeg",
            "-i", org_file_path,
            "-ar", "16000",    # 16kHz
            "-ac", "1",        # mono
            "-f", "wav",
            "pipe:1"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )
    return io.BytesIO(process.stdout)

# -----------------------------
# Vosk 모델 로드
# -----------------------------
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"  # 모델 다운로드 후 경로
if not os.path.exists(VOSK_MODEL_PATH):
    raise FileNotFoundError("Vosk 모델을 먼저 다운로드하세요!")
vosk_model = Model(VOSK_MODEL_PATH)

def stt_vosk(user_audio_path: str) :
    """사용자 음성파일을 Vosk STT로 변환"""
    audio_stream = prepare_audio_for_vosk(user_audio_path)

    # Vosk recognizer 초기화
    model = Model(VOSK_MODEL_PATH)
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)

    # wav header skip 필요 → wave 모듈 사용 X, raw bytes 그대로 처리
    while True:
        data = audio_stream.read(4000)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)

    result = json.loads(rec.FinalResult())
    text = result.get("text", "").strip()
    words = result.get("result", [])  # 단어별 confidence

    conf_dict = {w["word"]: w.get("conf", 0) for w in words}
    return text, conf_dict

# Try import Coqui TTS (optional)
try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# 앱 시작 시 한 번만 로드
# 모델 선택: (사용 가능한 모델 이름은 환경에 따라 바꿔야 함)
# 예: LJSpeech → 미국 여성 화자 데이터셋 기반
# 발음은 전형적인 American English
tts_us_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
#tts_uk_model = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)

# 단순 function words 리스트 (빠져도 되는경우가 많은 단어들)
# 기능어 리스트
FUNCTION_WORDS = set([
    "a", "an", "the",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did",
    "have", "has", "had",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "to", "of", "in", "on", "at", "for", "with", "from", "by",
    "and", "but", "or", "so", "because",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"
])

# 축약(권장) 화이트리스트: 키는 canonical, 값은 허용되는 축약 패턴들(정규표현식)
CONTRACTION_WHITELIST = {
    "could have": [r"coulda", r"could've", r"could of"],
    "would have": [r"woulda", r"would've", r"would of"],
    "should have": [r"shoulda", r"should've", r"should of"],
    "going to": [r"gonna", r"gon?na"],
    "want to": [r"wanna"],
    "want a": [r"gimme", r"lemme", r"gonna"],  # 예시
    "let me": [r"lemme"],
    "give me": [r"gimme"],
    "I am": [r"I'm", r"Im", r"i'm"],
    "do not": [r"don't", r"dont"],
}

def score_content_word(w, user_tokens, conf_dict):
    """내용어 점수 계산 (가중치↑, confidence 기준↑)"""
    conf = conf_dict.get(w, 0)
    if w in user_tokens and conf >= 0.6:
        return 2.0
    elif conf >= 0.55:
        return 1.8
    elif conf >= 0.4:
        return 1.6
    else:
        return 0.0

def score_function_word(w, user_tokens, conf_dict):
    """기능어 점수 계산 (보너스, confidence 기준↑)"""
    conf = conf_dict.get(w, 0)
    gained = 0.0
    if w in user_tokens:
        # 축약 확인
        for base, patterns in CONTRACTION_WHITELIST.items():
            if w in base.split():
                if any(re.fullmatch(pat, ut) for ut in user_tokens for pat in patterns):
                    if conf >= 0.6:
                        return 2.5
        if conf >= 0.6:
            gained = 1.5
        elif conf >= 0.5:
            gained = 1.2
    elif conf >= 0.4:
        gained = 0.8
    return gained

# -----------------------------
# 간단한 청킹 함수
# -----------------------------
def chunk_sentence(text: str):
    """
    영어 문장을 원어민 리듬에 맞춰 대략적인 청크 단위로 분리
    """
    patterns = [r"\b(and|but|or|so|because)\b",
                r"\b(could have|would have|should have|going to|want to|let me|give me)\b",
                r"\b(in|on|at|for|with|from|by|to|of)\b"]

    text = text.lower()
    for p in patterns:
        text = re.sub(p, r"@@\1@@", text)

    raw_chunks = [c.strip() for c in text.split("@@") if c.strip()]
    chunks = [re.findall(r"[a-zA-Z']+", c) for c in raw_chunks]
    return chunks

  
def check_contraction(user_transcript: str, target_phrase: str) -> bool:
    """사용자가 허용된 축약형을 썼는지 확인"""
    if target_phrase in CONTRACTION_WHITELIST:
        for pattern in CONTRACTION_WHITELIST[target_phrase]:
            if re.search(pattern, user_transcript, re.IGNORECASE):
                return True
    return False

# -----------------------------
# TTS 생성 (길이 포함)
# -----------------------------
def tts_generate_us(text: str) -> bytes:
    """US tutor TTS → wav 바이트 리턴"""
    wav_path = "reference_us.wav"
    tts_us_model.tts_to_file(text=text, file_path=wav_path)
    with open(wav_path, "rb") as f:
        wav_bytes = f.read()

    seg = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
    duration_sec = len(seg) / 1000.0

    return wav_bytes, duration_sec
    
# -----------------------------
# Audio duration helper
# -----------------------------
def get_audio_duration(file_path: str) -> float:
    """wav 파일 길이(초)"""
    with wave.open(file_path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)   

def evaluate_pronunciation(target_text: str, user_audio_path: str, tutor_type: str = "us"):
    """
    전체 흐름: 사용자 오디오 → STT → 발음 평가 → 결과 반환
    """
    # 1) 튜터 참조 음성, 튜터 음성시간
    ref_audio,ref_duration = tts_generate_us(target_text) if tutor_type == "us" else None

    # 2) 사용자 음성 → STT
    user_transcript, conf_dict = stt_vosk(user_audio_path)
    user_tokens = re.findall(r"[a-zA-Z']+", user_transcript.lower())
    user_seg = AudioSegment.from_file(user_audio_path)
    # 사용자발화시간
    user_duration = len(user_seg) / 1000.0

    # 3) 청크화
    target_chunks = chunk_sentence(target_text)
    

    feedback = []
    score = 0
    total = 0

    # 4) 청크 비교
    for chunk in target_chunks:
        total += 2
        content_words = [w for w in chunk if w not in FUNCTION_WORDS and len(w) > 1]
        function_words = list(dict.fromkeys([w for w in chunk if w in FUNCTION_WORDS]))  # 중복 제거

        # 내용어 평가
        for w in content_words:
            total += 2.0  # 기준점수는 그대로
            gained = score_content_word(w, user_tokens, conf_dict)
            score += gained
            if gained == 2.0:
                feedback.append(f"내용어 '{w}'는 분명히 잘 들렸어요 👍")
            elif gained >= 1.5:
                feedback.append(f"내용어 '{w}'는 대체로 좋았지만 조금 더 또렷하면 완벽해요.")
            elif gained >= 1.0:
                feedback.append(f"내용어 '{w}'는 들리긴 했지만 약했어요.")
            else:
                feedback.append(f"내용어 '{w}' 발음을 놓친 것 같아요.")

        # 기능어 평가
        for w in function_words:
            gained = score_function_word(w, user_tokens, conf_dict)
            score += gained
            if gained >= 0.8:
                feedback.append(f"'{w}'를 축약해서 자연스럽게 말했네요 👌")
            elif gained >= 0.5:
                feedback.append(f"기능어 '{w}'는 무난히 발음했어요.")
            elif gained > 0:
                feedback.append(f"기능어 '{w}'는 조금 약했어요.")
            else:
                feedback.append(f"기능어 '{w}' 발음이 거의 안 들렸어요.")
    
    # 5) 속도 보너스 (20%)
    if user_duration <= ref_duration + 5:
        bonus = (score / total) * 0.2
        score += bonus
        feedback.append("⏱️ 발화 속도가 자연스러워서 추가 점수를 드립니다!")

    print("DEBUG total",total)
    print("DEBUG score",score)
    percentage = round((score / total) * 100, 1)

    # 최종 결과
    result = {
        "score": percentage,
        "feedback": feedback,
        "target_chunks": target_chunks,
        "reference_tts": ref_audio,   # US tutor 음성 (wav 바이트)
        "user_transcript": user_transcript,
        "user_duration": user_duration,
        "ref_duration": ref_duration
    }
    return result

    # store into global_store as well (synchronized)
    global_store.tts_us_comment = final_comment
    global_store.tts_us_audio = tts_bytes

"""
    out = {
        "comment": final_comment,
        "highlights": highlights,
        "tts_audio": tts_bytes
    }
    return out
"""

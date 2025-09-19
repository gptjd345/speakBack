from .store import global_store
import re
from typing import Optional, Dict
import tempfile
import difflib

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

# 단순 function words 리스트 (빠져도 되는경우가 많은 단어들)
FUNCTION_WORDS = {
    "a","an","the","to","of","in","on","at","for","and","but","or","so","if",
    "is","are","was","were","be","been","being","have","has","had","do","does","did",
    "that","this","these","those","with","by","as","from","about","into","over","after",
    "before","between","among","during","until","while","because","since","through"
}

# 유틸: 단어 정제
def tokenize_simple(text: str):
    text = text.strip().lower()
    # 따옴표/괄호 제거
    text = re.sub(r"[\"'()]", "", text)
    # 하이픈을 공백으로 바꾸기
    text = re.sub(r"[-/]", " ", text)
    # 비문자 숫자 제거(단어 내 숫자 유지하려면 수정)
    tokens = re.findall(r"\b[\w']+\b", text)
    return tokens

def matches_contraction(token: str, canonical_phrase: str) -> bool:
    token = token.lower()
    patterns = CONTRACTION_WHITELIST.get(canonical_phrase, [])
    for p in patterns:
        if re.fullmatch(p, token):
            return True
    return False

def is_function_word(token: str) -> bool:
    return token.lower() in FUNCTION_WORDS

def _segment_confidence_hint(asr_result: dict) -> Optional[float]:
    """
    asr_result 는 whisper.transcribe(...) 의 결과(사전)라고 가정.
    segments 가 있으면 각 segment의 avg_logprob 값을 사용해 proxy confidence 계산.
    (단순히 평균 - 클수록 신뢰)
    """
    try:
        segs = asr_result.get("segments", [])
        if not segs:
            return None
        avg = sum((s.get("avg_logprob", 0.0) or 0.0) for s in segs) / len(segs)
        # avg_logprob는 음수(높을수록 0에 가깝다). 그대로 반환
        return avg
    except Exception:
        return None
    
def generate_tts_bytes(text: str, voice: str = "us") -> bytes:
    """
    Coqui TTS 사용시 파일을 tmp로 만들고 바이트를 읽어서 반환.
    voice 인자는 향후 다중 목소리 선택용 placeholder.
    """
    if not TTS_AVAILABLE:
        return b""  # placeholder: 사용시엔 빈 바이트가 아닌 경고/플레이스홀더를 쓸 수 있음

    tts = tts_us_model
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name
    # TTS 라이브러리의 tts_to_file 사용
    tts.tts_to_file(text=text, file_path=tmp_path)
    with open(tmp_path, "rb") as f:
        data = f.read()
    return data    

# pronunciation_module.py
def evaluate_pronunciation(target_text: str, asr_text: str, asr_result: Optional[dict] = None) -> Dict:
    """
    메인 평가 함수
    - target_text: 사용자가 말하려던 문장
    - asr_text: ASR이 출력한 텍스트 (string)
    - asr_result: optional full ASR result dict (whisper result) -> confidence hint 사용
    반환:
        {
          "comment": str,
          "highlights": [ { "type":"missing|mismatch|ok|contraction", "target":..., "asr":... } ... ],
          "tts_audio": bytes
        }
    동작 요약:
      1) 단어 토큰화
      2) difflib로 align -> mismatch/missing 추출
      3) function word 여부로 중요도 결정
      4) contraction whitelist 허용/칭찬
      5) segment-level avg_logprob(옵션)으로 불확실성 반영
      6) TTS 생성(가능하면)
    """
   
    tgt_tokens = tokenize_simple(target_text)
    asr_tokens = tokenize_simple(asr_text or "")

    # alignment: sequence matcher produces opcodes
    sm = difflib.SequenceMatcher(a=tgt_tokens, b=asr_tokens)
    ops = sm.get_opcodes() # 양쪽이 얼마나 다른지를 비교해줌

    highlights = []
    critical_issues = []
    minor_issues = []

    # confidence proxy (avg_logprob); 작을수록 덜 신뢰(whisper 음성 log prob는 음수)
    seg_conf = _segment_confidence_hint(asr_result)

    for tag, i1, i2, j1, j2 in ops:
        # tag: 'equal', 'replace', 'delete', 'insert'
        tgt_chunk = tgt_tokens[i1:i2]
        asr_chunk = asr_tokens[j1:j2]
        if tag == "equal":
            for w in tgt_chunk:
                highlights.append({"type":"ok", "target":w, "asr":w})
            continue

        if tag == "replace":
            # target replaced by asr: could be contraction or mispronounce
            for tw, aw in zip(tgt_chunk, asr_chunk):
                # contraction check: if asr token matches allowed contraction for this tgt_chunk phrase
                combined_tgt = " ".join([tw])  # for single-word replacement, still works
                # check canonical phrases of length 2 as well (maybe target was two-word)
                contraction_matched = False
                # try two-word canonical if exists
                if (i1+1)<len(tgt_tokens):
                    two = tgt_tokens[i1:i1+2]
                    canonical2 = " ".join(two)
                    if matches_contraction(aw, canonical2):
                        highlights.append({"type":"contraction", "target":canonical2, "asr":aw})
                        contraction_matched = True
                if not contraction_matched and matches_contraction(aw, tw):
                    highlights.append({"type":"contraction", "target":tw, "asr":aw})
                    contraction_matched = True

                if contraction_matched:
                    # contraction is encouraged
                    continue

                # otherwise determine importance
                if is_function_word(tw):
                    highlights.append({"type":"minor_mismatch", "target":tw, "asr":aw})
                    minor_issues.append((tw, aw))
                else:
                    highlights.append({"type":"mismatch", "target":tw, "asr":aw})
                    critical_issues.append((tw, aw))

            # if lengths differ, note remaining
            if len(tgt_chunk) > len(asr_chunk):
                # some target words deleted/omitted
                omitted = tgt_chunk[len(asr_chunk):]
                for w in omitted:
                    if is_function_word(w):
                        highlights.append({"type":"ok_omitted", "target":w, "asr":""})
                    else:
                        highlights.append({"type":"missing", "target":w, "asr":""})
                        critical_issues.append((w, ""))
            elif len(asr_chunk) > len(tgt_chunk):
                extra = asr_chunk[len(tgt_chunk):]
                for w in extra:
                    highlights.append({"type":"extra", "target":"", "asr":w})
                    minor_issues.append(("", w))

        elif tag == "delete":
            # target words deleted in ASR
            for w in tgt_chunk:
                if is_function_word(w):
                    highlights.append({"type":"ok_omitted", "target":w, "asr":""})
                else:
                    highlights.append({"type":"missing", "target":w, "asr":""})
                    critical_issues.append((w, ""))
        elif tag == "insert":
            # ASR inserted words not in target (probably filler/recognition artifact)
            for w in asr_chunk:
                highlights.append({"type":"extra", "target":"", "asr":w})
                minor_issues.append(("", w))

    # Build human-readable comment with soft rules:
    comment_lines = []

    # If no critical issues -> praise
    if not critical_issues:
        comment_lines.append("Good — your message is understandable.")
    else:
        # list up to 3 critical items
        sample = critical_issues[:3]
        for t, a in sample:
            if a == "":
                comment_lines.append(f"The word **'{t}'** was not clearly heard — try to pronounce it a bit more distinctly.")
            else:
                comment_lines.append(f"It sounds like you said **'{a}'** instead of **'{t}'** — check that word.")
        # add general hint
        comment_lines.append("Focus on pronouncing content words (nouns/verbs) clearly; function words can be reduced.")

    # Mention contractions preference
    # If we detected contraction highlights, praise
    contractions_found = [h for h in highlights if h["type"] == "contraction"]
    if contractions_found:
        comment_lines.append("Nice natural contractions detected — in casual speech, contractions help flow. Keep it.")

    # If segment-level avg_logprob is very low, warn about overall clarity (low confidence)
    if seg_conf is not None:
        # seg_conf is avg_logprob (negative). Closer to 0 is better, very negative is bad.
        if seg_conf < -3.0:
            comment_lines.append("Audio was a bit unclear (low ASR confidence). Try speaking a bit louder or with less background noise.")
        elif seg_conf < -1.8:
            comment_lines.append("Some parts had low confidence — re-recording might help for clearer feedback.")

    # Final comment
    final_comment = " ".join(comment_lines)

    # TTS generation: create tutor audio for the canonical target_text but prefer contraction suggestions:
    # If contractions were found, use ASR text (to preserve student's chosen contraction) else use target_text
    tts_input_text = None
    if contractions_found:
        # use asr_text to keep student's contraction style, but ensure not empty
        tts_input_text = asr_text if asr_text.strip() else target_text
    else:
        # encourage contraction: try to produce a contracted version of target_text by replacing known phrases
        contracted = target_text
        for canonical, patterns in CONTRACTION_WHITELIST.items():
            pat = re.compile(r"\b" + re.escape(canonical) + r"\b", flags=re.IGNORECASE)
            # replace canonical phrase by its first whitelist short form if exists
            if pat.search(contracted):
                replacement = patterns[0]  # take first contraction suggestion (e.g., coulda)
                contracted = pat.sub(replacement, contracted)
        # prefer contracted if different
        tts_input_text = contracted if contracted != target_text else target_text

    # Generate TTS bytes (try Coqui)
    try:
        tts_bytes = generate_tts_bytes(tts_input_text, voice="us")
    except Exception:
        tts_bytes = b""

    # store into global_store as well (synchronized)
    global_store.tts_us_comment = final_comment
    global_store.tts_us_audio = tts_bytes

    out = {
        "comment": final_comment,
        "highlights": highlights,
        "tts_audio": tts_bytes
    }
    return out

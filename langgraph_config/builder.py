# 그래프 정의부
# langgraph_config/builder.py
from langgraph.graph import StateGraph, START, END
from .store import global_store
import whisper
import tempfile
import soundfile as sf
import io
import torch
from TTS.api import TTS

from .pronunciation_module import evaluate_pronunciation

from langsmith import trace
from langsmith.run_helpers import traceable

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

# 간단한 state 구조 (필요시 MessagesState 써도 됨)
class PipelineState(dict):
    @traceable
    def __merge__(self, other):
        merged = PipelineState(self)
        decisions = {}
        for k, v in other.items():
            if k in merged:
                decisions[k] = {"before": merged[k], "after": v, "action": "overwrite"}
            else:
                decisions[k] = {"before": None, "after": v, "action": "add"}
            merged[k] = v
        return {"decisions": decisions, "merged_state": dict(merged)}

# Whisper 모델 (한번 로드 후 재사용) -- 필요없을거같아서 지움
# ws_model = whisper.load_model("base")

# 한번만 로드해서 재사용
tts_us_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
#tts_uk_model = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False, gpu=False)

# ---------------- 노드 정의 ----------------
def audio_store_node(state):
    # 사용자 음성 데이터를 numpy + tensor 로 변환하고 Whisper STT 수행
    audioFile = global_store.audio_file
    if audioFile is None:
        state["err_txt"] = "[audio store ERROR] No audio data found"
        return state
    
    try:
        # 1) BytesIO → 임시 wav 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audioFile.read())
            tmp_path = tmp.name

        global_store.tmp_path = tmp_path

    except Exception as e:
        state["err_txt"] = f"[audio store ERROR] {e}"
        return state

    return state

def us_tutor_node(state: PipelineState):
    """미국 튜터 피드백 + 음성 생성"""
    target_text = global_store.target_text

    result = evaluate_pronunciation(
        target_text, 
        global_store.tmp_path,     # 사용자 임시 음성파일경로
        "us"                       # tutor type
    )

    # 디버그: 최종 결과 확인
    print("=== US Tutor Final Result ===")
    print("Score:\n", result["score"])
    print("Feedback:\n", result["feedback"])
    print("TTS Audio Bytes Length:", len(result["reference_tts"]))

    global_store.score = result["score"]
    global_store.us_feedback = result["feedback"]
    global_store.tts_us_audio = result["reference_tts"]
    global_store.user_transcript = result["user_transcript"]
    global_store.target_chunks = result["target_chunks"]
    global_store.user_duration = result["user_duration"]
    global_store.us_ref_duration = result["ref_duration"]

    return state

def uk_tutor_node(state: PipelineState):
    """영국 튜터 피드백"""

    # 실제 AI 평가 로직은 여기서 audio_np 기반
    global_store.tts_uk_comment = "[UK Tutor] Try softer vowels."
    global_store.tts_uk_audio = b"fake-uk-audio-bytes"

    print("=== [uk_tutor_node] Final State Snapshot ===")
    """
    for k, v in state.items():
        if isinstance(v, (bytes, bytearray)):
            print(f"{k}: <{len(v)} bytes>")
        else:
            print(f"{k}: {v}")
    """

    return state

def tts_node(state: PipelineState):
    # TTS는 이미 us/uk tutor에서 만든 걸 합쳐서 처리 가능
    state["tts_done"] = True

    print("=== [tts_node] Final State Snapshot ===")
          
    return state

def db_save_node(state: PipelineState):
    # DB 저장 시뮬레이션
    print("=== [DB Save Node] Final State Snapshot ===")
   
    return state

# ---------------- 그래프 빌더 ----------------
def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("audio_store", audio_store_node)
    graph.add_node("us_tutor", us_tutor_node)
    graph.add_node("uk_tutor", uk_tutor_node)
    graph.add_node("tts", tts_node)
    graph.add_node("db", db_save_node)

    graph.add_edge(START, "audio_store")      # 입력 → 오디오 저장/변환
    graph.add_edge("audio_store", "us_tutor")  # 오디오 → STT
    graph.add_edge("audio_store", "uk_tutor")
    graph.add_edge("us_tutor", "tts")
    graph.add_edge("uk_tutor", "tts")
    graph.add_edge("tts", "db")
    graph.add_edge("db", END)

    # CompiledStateGraph.invoke(state) 내부에서 새로운 상태 객체를 만들거나 덮어쓰는 과정이 있어서 바깥으로 제대로 전달되지 않는 거예요
    # 컴파일을 여기서 안하기로함
    return graph.compile()

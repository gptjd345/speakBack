import streamlit as st
from io import BytesIO
import base64
from audiorecorder import audiorecorder
from langgraph_config.graph_runner import run_pipeline
from dotenv import load_dotenv
load_dotenv()  # .env 파일 읽어서 환경변수 자동 등록

st.title("Pronunciation Coach 🎤")
# Target Text 입력
target_text = st.text_input("Enter the target sentence (for pronunciation practice)")
st.write("Upload your voice or record directly for corrections from US & UK tutors.")

# 선택: 업로드 vs 녹음
input_method = st.radio("Choose input method:", ["Upload Audio File", "Record Audio"])

name = st.text_input("Your Name")

# 상태 초기화
if "audio_file" not in st.session_state:
    st.session_state.audio_file = None
if "audio_name" not in st.session_state:
    st.session_state.audio_name = None

# ------------------------------
# 1️⃣ 오디오 업로드 선택
# ------------------------------
if input_method == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload your voice file", type=["wav", "mp3", "m4a"])
    if uploaded_file:
        st.session_state.audio_file = uploaded_file
        st.session_state.audio_name = uploaded_file.name

        st.markdown(f"**Uploaded File:** {st.session_state.audio_name} ✅")
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")

# ------------------------------
# 2️⃣ 브라우저 녹음 선택
# ------------------------------
elif input_method == "Record Audio":
    st.write("Click below to record your voice:")

    audio = audiorecorder("Start Recording 🎙️", "Stop Recording ⏹️")

    if len(audio) > 0:
        buf = BytesIO()
        audio.export(buf, format="wav")
        audio_bytes = buf.getvalue()

        audio_file = BytesIO(audio_bytes)
        audio_file.name = "recorded_audio.wav"

        st.session_state.audio_file = audio_file
        st.session_state.audio_name = audio_file.name

        # 디버깅용 출력
        st.write(f"DEBUG: audio byte size = {len(audio_bytes)}")

        st.audio(audio_bytes, format="audio/wav")
        st.markdown(f"**Recording Complete ✅** File: {st.session_state.audio_name}")

# ------------------------------
# 3️⃣ 전송 버튼
# ------------------------------
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("Send to LangGraph"):
        audio_file = st.session_state.audio_file
        audio_name = st.session_state.audio_name

        if name and target_text and audio_file:
            # 실제 LangGraph 처리 함수 호출 자리
            result = run_pipeline(audio_file, name, target_text)  # 🚀 LangGraph 실행
            st.write("### LangGraph Result")
            st.json(result) # 결과 dict 보여주기 

            print("Debug run_pipeline : ",result)

            # US TTS 음성 재생
            us_audio_bytes = result.get("us_audio")
            if us_audio_bytes:
                st.audio(us_audio_bytes, format="audio/wav")

            st.write("### UK Tutor Feedback")
            st.markdown(result.get("uk_comment", "No UK comment available"))

            # UK TTS 음성 재생 (가짜일 경우 빈 바이트 체크)
            uk_audio_bytes = result.get("uk_audio")
            if uk_audio_bytes:
                st.audio(uk_audio_bytes, format="audio/wav")

        else:
            st.warning("Please enter your name and upload/record an audio file!")


# langgraph_config/store.py
class GlobalStore:
    def __init__(self):
        self.target_text = None   # 사용자가 말하고자 하는 목표 문장  
        self.audio_file = None    # BytesIO 같은 파일 객체

        # 마이크 입력시 샘플링 주파수 32khz, 이고 whisper읽을때 16khz라 어차피 다시 샘플링해야함 
        self.tmp_path = None       # 원본파일 저장위치

        self.user_name = None      # 사용자 정보
        # 필요하면 더 추가 (예: 세션 ID 등)

        self.score = None # score
        self.us_feedback = None # us_feedback
        self.tts_us_audio = None # reference 참고용 음성
        self.user_transcript = None # user_transcript
        self.target_chunks = None # target 의 청크 묶음

        self.user_duration = None # 사용자 발화 시간
        self.us_ref_duration = None  # us tutor 발화시간

        self.tts_uk_audio = None   # uk tutor 가 만든 교정데이터
        self.tts_uk_comment = None # uk tutor 가 만든 교정데이터

# 전역 싱글톤처럼 import해서 씀
global_store = GlobalStore()


# langgraph_config/store.py
class GlobalStore:
    def __init__(self):
        self.target_text = None   # 사용자가 말하고자 하는 목표 문장  
        self.audio_file = None    # BytesIO 같은 파일 객체

        # 마이크 입력시 샘플링 주파수 32khz, 이고 whisper읽을때 16khz라 어차피 다시 샘플링해야함 
        #self.audio_np = None      # numpy array로 변환된 오디오
        # samplerate : 오디오 데이터를 초당 몇 개의 샘플로 쪼개서 기록했는지 나타냄
        #self.audio_sr = None      # 일반적인 음성 오디오는 16,000 Hz (16kHz)
        #self.audio_tensor = None  # tensor로 변경한 오디오
        self.tmp_path = None       # 원본파일 저장위치

        self.user_name = None      # 사용자 정보
        # 필요하면 더 추가 (예: 세션 ID 등)

        self.stt_text = None       # 사용자음성 데이터를 AI 가 텍스트화 한 참고데이터
        self.us_asr_result = None  # us asr 결과 분석
        self.uk_asr_result = None  # uk asr 결과 분석

        self.tts_us_audio = None   # us tutor 가 만든 교정데이터
        self.tts_us_comment = None # us tutor 가 만든 교정데이터
        self.us_highlights = None # us tutor의 하이라이트 

        self.tts_uk_audio = None   # uk tutor 가 만든 교정데이터
        self.tts_uk_comment = None # uk tutor 가 만든 교정데이터

# 전역 싱글톤처럼 import해서 씀
global_store = GlobalStore()


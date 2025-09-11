from langgraph.graph import StateGraph, MessagesState, START, END

def process_input(name: str, audio_file) -> str:
    """
    LangGraph 실행: 이름과 오디오 파일 처리
    audio_file은 Streamlit에서 업로드한 파일 객체(BytesIO 등)
    """
    builder = StateGraph(MessagesState)

    def process_node(state: MessagesState):
        # 파일명 추출 (Streamlit 업로드 객체에는 .name 속성이 있음)
        audio_name = getattr(state["messages"][-1][1], "name", "uploaded_audio.wav")
        user_name = state["messages"][-2][1]  # 직전에 들어온 name 값
        
        return {
            "messages": [
                ("ai", f"Name: {user_name}, Audio File: {audio_name} received ✅")
            ]
        }

    builder.add_node("process", process_node)
    
    builder.add_edge(START, "process")
    builder.add_edge("process", END)

    graph = builder.compile()

    result = graph.invoke({
        "messages": [
            ("user", name),          # 이름
            ("file", audio_file),    # 업로드된 파일 객체
        ]
    })
    return result["messages"][-1][1]


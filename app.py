import streamlit as st
import pandas as pd
import os
import glob
import datetime
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# -----------------------------------------------------------------------------
# [0] 설정 및 데이터 로드 (초고속 모드 적용)
# -----------------------------------------------------------------------------
st.set_page_config(page_title="KW-강의마스터 Pro", page_icon="🎓", layout="wide")
api_key = os.environ.get("GOOGLE_API_KEY", "")

# 세션 상태 초기화
if "global_log" not in st.session_state:
    st.session_state.global_log = [] 
if "timetable_result" not in st.session_state:
    st.session_state.timetable_result = "" 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 
if "current_menu" not in st.session_state:
    st.session_state.current_menu = "🤖 AI 학사 지식인"
if "timetable_chat_history" not in st.session_state:
    st.session_state.timetable_chat_history = []

def add_log(role, content, menu_context=None):
    timestamp = datetime.datetime.now().strftime("%H:%M")
    st.session_state.global_log.append({
        "role": role,
        "content": content,
        "time": timestamp,
        "menu": menu_context
    })

# 가상 강의평 데이터
def load_mock_reviews():
    data = {
        "과목명": ["C프로그래밍", "C프로그래밍", "대학수학1", "공학설계입문", "대학영어", "회로이론1", "대학물리학1"],
        "교수명": ["김코딩", "이자바", "이수학", "박설계", "Brown", "최전기", "나물리"],
        "강의평": [
            "과제 폭탄입니다. 살려주세요.",
            "천사 교수님. 학점 잘 주심.",
            "진도가 너무 빠름. 예습 필수.",
            "팀플 빌런 만나면 한 학기 망함.",
            "출석만 잘 하면 B+은 기본.",
            "시험이 족보에서 그대로 나옴.",
            "물포자도 이해하게 설명해주심."
        ],
        "시험정보": [
            "손코딩 시험", "실습 시험", "교재 연습문제 변형", "발표 비중 큼", "오픈북", "족보 암기 필수", "공식 암기 위주"
        ]
    }
    return pd.DataFrame(data)

REVIEW_DB = load_mock_reviews()

@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    """
    [초고속 로딩] 미리 만들어둔 텍스트 파일(cached_knowledge.txt)을 우선적으로 읽습니다.
    """
    cache_file = "data/cached_knowledge.txt"
    
    # 1. 미리 학습된 파일이 있는지 확인 (가장 빠름)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = f.read()
                if len(data) > 10: # 내용이 제대로 있는지 확인
                    return data, True # True = 캐시 사용됨
        except:
            pass # 읽기 실패하면 아래로 넘어감

    # 2. 캐시가 없으면 PDF 직접 파싱 (느림, 비상용)
    if not os.path.exists("data"):
        return "", False
        
    all_content = ""
    pdf_files = glob.glob("data/*.pdf")
    if not pdf_files:
        return "", False
        
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            pages = loader.load_and_split()
            filename = os.path.basename(pdf_file)
            all_content += f"\n\n--- [문서: {filename}] ---\n"
            for page in pages:
                all_content += page.page_content
        except: continue
    
    return all_content, False # False = 실시간 파싱됨

# 데이터 로드 실행
PRE_LEARNED_DATA, IS_CACHED = load_knowledge_base()

# 로딩 상태 표시 (사이드바)
if IS_CACHED:
    # 토스트 메시지는 앱 켤 때 한 번만 쓱 지나감
    st.toast("⚡ 미리 학습된 데이터를 사용하여 즉시 실행되었습니다!", icon="🚀")

# -----------------------------------------------------------------------------
# [1] AI 엔진
# -----------------------------------------------------------------------------
def get_llm():
    if not api_key: return None
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-09-2025", temperature=0)

def ask_ai(question):
    llm = get_llm()
    if not llm: return "⚠️ API Key 오류"
    try:
        chain = PromptTemplate.from_template(
            "문서 내용: {context}\n질문: {question}\n문서에 기반해 답변해줘."
        ) | llm
        return chain.invoke({"context": PRE_LEARNED_DATA, "question": question}).content
    except Exception as e: return str(e)

# 시간표 생성 함수 (HTML 컬러 테이블 + 선수과목 강조)
def generate_timetable_ai(major, grade, semester, target_credits, blocked_times_desc, requirements):
    llm = get_llm()
    if not llm: return "⚠️ API Key 오류"
    
    review_summary = REVIEW_DB.to_string()
    
    template = """
    너는 대학교 수강신청 전문가야. PDF 문서(시간표, 요람)를 분석해서 최적의 시간표를 짜줘.

    [학생 정보]
    - {major} {grade} {semester}
    - 목표: {target_credits}학점
    - 공강 필수 시간: {blocked_times} (이 시간은 수업 배치 절대 금지)
    - 추가요구: {requirements}

    [강의평 데이터]
    {review_data}

    [필수 지시사항]
    1. **모든 학년의 선수/후수 과목 철저 준수**:
       - 1학년뿐만 아니라, **2, 3, 4학년의 전공 연계성**을 반드시 확인해라.
       - 예시: "회로이론1(1학기) → 회로이론2(2학기)", "전자장1 → 전자장2", "종합설계(캡스톤) 이수 조건" 등.
       - 해당 학기({semester})에 들어야 다음 학기나 다음 학년에 문제가 없는 **'필수 선수 과목'**은 무조건 시간표에 넣어라.
       - 결과 설명에 "**[필수] 이 과목은 다음 단계인 OO과목 수강을 위해 꼭 들어야 합니다.**"라고 이유를 명시해라.
    
    2. **출력 형식 (HTML Table)**:
       - 마크다운 표 대신 **HTML `<table>` 태그**를 사용해라.
       - 각 수업마다 **서로 다른 파스텔톤 배경색**(`style="background-color: #..."`)을 적용해라.
       - 셀 내용: `<b>과목명</b><br><small>교수명</small>`
       - 행: 1교시~9교시 (시간 미포함), 열: 월~금
       - 표는 시각적으로 예쁘고 깔끔하게 만들어라.
    
    3. **공강 시간 처리**:
       - 공강으로 지정된 시간은 비워둬라.
       - 출력 시 "공강 필수 시간" 목록을 텍스트로 보여주지 마라. (결과물만 깔끔하게)

    [학습된 문서]
    {context}
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "major", "grade", "semester", "target_credits", "blocked_times", "requirements", "review_data"])
    chain = prompt | llm
    input_data = {
        "context": PRE_LEARNED_DATA,
        "major": major,
        "grade": grade,
        "semester": semester,
        "target_credits": target_credits,
        "blocked_times": blocked_times_desc,
        "requirements": requirements,
        "review_data": review_summary
    }
    return chain.invoke(input_data).content

def chat_with_timetable_ai(current_timetable, user_input):
    llm = get_llm()
    template = """
    너는 현재 시간표에 대한 상담을 해주는 AI 조교야.
    
    [현재 시간표 상태]
    {current_timetable}

    [사용자 입력]
    "{user_input}"

    [지시사항]
    사용자의 입력 의도를 파악해서 아래 두 가지 중 하나로 반응해.
    
    **Case 1. 시간표 수정 요청인 경우 (예: "1교시 빼줘", "교수 바꿔줘"):**
    - 시간표를 **재작성(HTML Table 형식 유지)**해줘.
    - 수정된 시간표를 출력하고, 무엇이 바뀌었는지 짧게 설명해.
    
    **Case 2. 과목에 대한 단순 질문인 경우 (예: "C프로그래밍 과제 많아?", "이거 선수과목 뭐야?"):**
    - **시간표를 다시 출력하지 말고**, 질문에 대한 **텍스트 답변**만 해.
    - 강의평 데이터나 학습된 지식을 활용해.
    
    답변 시작에 [수정] 또는 [답변] 태그를 붙여서 구분해줘.
    """
    prompt = PromptTemplate(template=template, input_variables=["current_timetable", "user_input"])
    chain = prompt | llm
    return chain.invoke({"current_timetable": current_timetable, "user_input": user_input}).content

# -----------------------------------------------------------------------------
# [2] UI 구성
# -----------------------------------------------------------------------------
def change_menu(menu_name):
    st.session_state.current_menu = menu_name

with st.sidebar:
    st.title("🗂️ 활동 로그")
    st.caption("클릭하면 해당 화면으로 이동합니다.")
    
    log_container = st.container(height=400)
    with log_container:
        if not st.session_state.global_log:
            st.info("기록 없음")
        else:
            for i, log in enumerate(reversed(st.session_state.global_log)):
                label = f"[{log['time']}] {log['content'][:15]}..."
                if st.button(label, key=f"log_btn_{i}", use_container_width=True):
                    if log['menu']:
                        change_menu(log['menu'])
                        st.rerun()

    st.divider()
    st.markdown("### ℹ️ 상태")
    if IS_CACHED:
        st.success("🚀 고속 모드 (Pre-learned)")
    elif PRE_LEARNED_DATA:
        st.warning("🐢 일반 모드 (PDF 실시간 분석)")
    else:
        st.error("데이터 없음")


menu = st.radio("기능 선택", ["🤖 AI 학사 지식인", "📅 스마트 시간표(수정가능)", "🔍 강의평 분석"], 
                horizontal=True, key="menu_radio", 
                index=["🤖 AI 학사 지식인", "📅 스마트 시간표(수정가능)", "🔍 강의평 분석"].index(st.session_state.current_menu))

if menu != st.session_state.current_menu:
    st.session_state.current_menu = menu
    st.rerun()

st.divider()

if st.session_state.current_menu == "🤖 AI 학사 지식인":
    st.subheader("🤖 무엇이든 물어보세요")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("질문 입력"):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        add_log("user", f"[지식인] {user_input}", "🤖 AI 학사 지식인")
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                response = ask_ai(user_input)
                st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

elif st.session_state.current_menu == "📅 스마트 시간표(수정가능)":
    st.subheader("📅 AI 맞춤형 시간표 설계")

    if st.session_state.timetable_result:
        st.markdown("### 🗓️ 내 시간표")
        st.markdown(st.session_state.timetable_result, unsafe_allow_html=True)
        st.divider()

    with st.expander("시간표 설정 열기/닫기", expanded=not bool(st.session_state.timetable_result)):
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("#### 1️⃣ 기본 정보")
            major = st.text_input("학과", "전자융합공학과")
            c1, c2 = st.columns(2)
            grade = c1.selectbox("학년", ["1학년", "2학년", "3학년", "4학년"])
            semester = c2.selectbox("학기", ["1학기", "2학기"])
            target_credit = st.number_input("목표 학점", 9, 24, 18)
            requirements = st.text_area("추가 요구사항", placeholder="예: 전공 필수 챙겨줘")

        with col2:
            st.markdown("#### 2️⃣ 공강 시간 설정")
            kw_times = {
                "1교시": "09:00~10:15", "2교시": "10:30~11:45", "3교시": "12:00~13:15",
                "4교시": "13:30~14:45", "5교시": "15:00~16:15", "6교시": "16:30~17:45",
                "7교시": "18:00~19:15", "8교시": "19:25~20:40", "9교시": "20:50~22:05"
            }
            schedule_index = [f"{k} ({v})" for k, v in kw_times.items()]
            schedule_data = pd.DataFrame(True, index=schedule_index, columns=["월", "화", "수", "목", "금"])
            edited_schedule = st.data_editor(
                schedule_data,
                column_config={
                    "월": st.column_config.CheckboxColumn("월", default=True),
                    "화": st.column_config.CheckboxColumn("화", default=True),
                    "수": st.column_config.CheckboxColumn("수", default=True),
                    "목": st.column_config.CheckboxColumn("목", default=True),
                    "금": st.column_config.CheckboxColumn("금", default=True),
                },
                height=360,
                use_container_width=True
            )

        if st.button("시간표 생성하기 ✨", type="primary", use_container_width=True):
            blocked_times = []
            for day in ["월", "화", "수", "목", "금"]:
                for idx, period_label in enumerate(edited_schedule.index):
                    if not edited_schedule.iloc[idx][day]:
                        blocked_times.append(f"{day}요일 {period_label}")
            blocked_desc = ", ".join(blocked_times) if blocked_times else "없음"
            with st.spinner("선수과목 확인 및 시간표 조합 중..."):
                result = generate_timetable_ai(major, grade, semester, target_credit, blocked_desc, requirements)
                st.session_state.timetable_result = result
                st.session_state.timetable_chat_history = []
                add_log("user", f"[시간표] {major} {grade} 생성", "📅 스마트 시간표(수정가능)")
                st.rerun()

    if st.session_state.timetable_result:
        st.subheader("💬 시간표 상담소")
        st.caption("시간표에 대해 질문하거나(Q&A), 수정을 요청(Refine)하세요.")
        for msg in st.session_state.timetable_chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)

        if chat_input := st.chat_input("예: 1교시 빼줘, 또는 대학수학1 꼭 들어야 해?"):
            st.session_state.timetable_chat_history.append({"role": "user", "content": chat_input})
            add_log("user", f"[상담] {chat_input}", "📅 스마트 시간표(수정가능)")
            with st.chat_message("user"):
                st.write(chat_input)
            with st.chat_message("assistant"):
                with st.spinner("분석 중..."):
                    response = chat_with_timetable_ai(st.session_state.timetable_result, chat_input)
                    if "[수정]" in response:
                        new_timetable = response.replace("[수정]", "").strip()
                        st.session_state.timetable_result = new_timetable
                        st.markdown(new_timetable, unsafe_allow_html=True)
                        st.session_state.timetable_chat_history.append({"role": "assistant", "content": "시간표를 수정했습니다. 위쪽 표를 확인해주세요."})
                        st.rerun()
                    else:
                        clean_response = response.replace("[답변]", "").strip()
                        st.markdown(clean_response)
                        st.session_state.timetable_chat_history.append({"role": "assistant", "content": clean_response})

elif st.session_state.current_menu == "🔍 강의평 분석":
    st.subheader("🔍 강의평 팩트체크")
    col1, col2 = st.columns(2)
    c_name = col1.selectbox("과목명", REVIEW_DB['과목명'].unique())
    p_name = col2.selectbox("교수명", REVIEW_DB[REVIEW_DB['과목명'] == c_name]['교수명'].unique())
    
    if "review_chat" not in st.session_state:
        st.session_state.review_chat = []

    if st.button("분석 시작"):
        reviews = REVIEW_DB[(REVIEW_DB['과목명']==c_name) & (REVIEW_DB['교수명']==p_name)]
        context = reviews.to_string()
        prompt = f"과목: {c_name}, 교수: {p_name}\n데이터: {context}\n이 강의의 장단점과 시험 스타일을 요약해줘."
        with st.spinner("분석 중..."):
            llm = get_llm()
            res = llm.invoke(prompt).content
            st.session_state.review_chat = [{"role": "assistant", "content": res, "context": context}]
            add_log("user", f"[강의평] {c_name}", "🔍 강의평 분석")

    for msg in st.session_state.review_chat:
        if "role" in msg:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    if st.session_state.review_chat:
        if q_input := st.chat_input("질문 입력"):
            st.session_state.review_chat.append({"role": "user", "content": q_input})
            with st.chat_message("user"):
                st.write(q_input)
            with st.chat_message("assistant"):
                context_data = st.session_state.review_chat[0].get("context", "")
                llm = get_llm()
                ans = llm.invoke(f"데이터: {context_data}\n질문: {q_input}\n답변해.").content
                st.write(ans)
                st.session_state.review_chat.append({"role": "assistant", "content": ans})

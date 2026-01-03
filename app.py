import streamlit as st
import pandas as pd
import os
import glob
import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# -----------------------------------------------------------------------------
# [0] 설정 및 데이터 로드
# -----------------------------------------------------------------------------
st.set_page_config(page_title="KW-강의마스터 Pro", page_icon="🎓", layout="wide")
api_key = os.environ.get("GOOGLE_API_KEY", "")

# 세션 상태 초기화 (대화 로그 및 데이터 유지용)
if "global_log" not in st.session_state:
    st.session_state.global_log = [] # 사이드바 표시용 로그
if "timetable_result" not in st.session_state:
    st.session_state.timetable_result = "" # 생성된 시간표 저장
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] # 학사 지식인 대화

def add_log(role, content):
    """사이드바 로그에 메시지 추가"""
    timestamp = datetime.datetime.now().strftime("%H:%M")
    st.session_state.global_log.append({
        "role": role,
        "content": content,
        "time": timestamp
    })

# 가상 강의평 데이터
def load_mock_reviews():
    data = {
        "과목명": ["C프로그래밍", "C프로그래밍", "대학수학1", "공학설계입문", "대학영어", "회로이론1"],
        "교수명": ["김코딩", "이자바", "이수학", "박설계", "Brown", "최전기"],
        "강의평": [
            "과제 폭탄입니다. 살려주세요.",
            "천사 교수님. 학점 잘 주심.",
            "진도가 너무 빠름. 예습 필수.",
            "팀플 빌런 만나면 한 학기 망함.",
            "출석만 잘 하면 B+은 기본.",
            "시험이 족보에서 그대로 나옴."
        ],
        "시험정보": [
            "손코딩 시험", "실습 시험", "교재 연습문제 변형", "발표 비중 큼", "오픈북", "족보 암기 필수"
        ]
    }
    return pd.DataFrame(data)

REVIEW_DB = load_mock_reviews()

@st.cache_resource(show_spinner="문서 학습 중...")
def load_knowledge_base():
    all_content = ""
    if not os.path.exists("data"):
        os.makedirs("data")
        return ""
    pdf_files = glob.glob("data/*.pdf")
    if not pdf_files:
        return ""
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            pages = loader.load_and_split()
            filename = os.path.basename(pdf_file)
            all_content += f"\n\n--- [문서: {filename}] ---\n"
            for page in pages:
                all_content += page.page_content
        except: continue
    return all_content

PRE_LEARNED_DATA = load_knowledge_base()

# -----------------------------------------------------------------------------
# [1] AI 엔진 (생성 및 수정 기능 포함)
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

# 시간표 생성 함수
def generate_timetable_ai(major, grade, semester, target_credits, blocked_times_desc, requirements):
    llm = get_llm()
    if not llm: return "⚠️ API Key 오류"
    
    review_summary = REVIEW_DB.to_string()
    
    template = """
    너는 '수강신청 마스터'야. PDF 문서(시간표)와 강의평을 참고해 최적의 시간표를 짜줘.

    [학생 정보]
    - {major} {grade} {semester}
    - 목표: {target_credits}학점
    - **공강 필수 시간(이 시간은 절대 수업 넣지마)**: {blocked_times}
    - 추가요구: {requirements}

    [강의평 데이터]
    {review_data}

    [지시사항]
    1. 실제 PDF 내 개설 과목과 시간을 매칭해.
    2. 결과는 **마크다운 표**로 출력해. (행: 1~9교시, 열: 월~금)
    3. 셀 내용: "과목명<br>(교수명)"
    4. 강의평이 안 좋은 과목이 포함되면 경고 문구(⚠️)를 띄워줘.

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

# 시간표 수정(꼬리 질문) 함수
def refine_timetable_ai(current_timetable, user_request):
    llm = get_llm()
    template = """
    너는 현재 시간표를 수정해주는 조교야.
    
    [현재 시간표]
    {current_timetable}

    [사용자의 수정 요청]
    "{user_request}"

    [지시사항]
    1. 사용자의 요청을 반영하여 시간표를 **재작성**해줘.
    2. 마크다운 표 형식을 유지해.
    3. 수정된 부분에 대해서는 짧게 코멘트를 달아줘.
    """
    prompt = PromptTemplate(template=template, input_variables=["current_timetable", "user_request"])
    chain = prompt | llm
    return chain.invoke({"current_timetable": current_timetable, "user_request": user_request}).content

# -----------------------------------------------------------------------------
# [2] UI 구성
# -----------------------------------------------------------------------------

# --- [사이드바] 대화 로그 표시 ---
with st.sidebar:
    st.title("🗂️ 활동 로그")
    st.caption("AI와의 대화 내역이 여기에 저장됩니다.")
    
    log_container = st.container(height=400)
    with log_container:
        if not st.session_state.global_log:
            st.info("아직 대화 내역이 없습니다.")
        else:
            for log in reversed(st.session_state.global_log):
                with st.chat_message(log["role"]):
                    st.write(f"**[{log['time']}]** {log['content']}")
    
    st.divider()
    st.markdown("### ℹ️ 학습된 데이터")
    try:
        pdf_count = len(glob.glob("data/*.pdf"))
        st.success(f"📚 PDF 문서 {pdf_count}개 연동됨")
    except:
        st.error("데이터 폴더 확인 필요")


# --- [메인 화면] 탭 메뉴 ---
menu = st.radio("기능 선택", ["🤖 AI 학사 지식인", "📅 스마트 시간표(수정가능)", "🔍 강의평 분석"], horizontal=True)
st.divider()

# 1. AI 학사 지식인 (일반 Q&A)
if menu == "🤖 AI 학사 지식인":
    st.subheader("🤖 무엇이든 물어보세요")
    
    # 기존 대화 출력
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 입력창
    if user_input := st.chat_input("질문 입력 (예: 졸업 요건이 뭐야?)"):
        # 유저 메시지 표시 및 저장
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        add_log("user", f"[지식인] {user_input}")
        with st.chat_message("user"):
            st.markdown(user_input)

        # AI 응답
        with st.chat_message("assistant"):
            with st.spinner("문서 검색 중..."):
                response = ask_ai(user_input)
                st.markdown(response)
        
        # AI 메시지 저장
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        add_log("assistant", f"[지식인 답변] {response[:30]}...")

# 2. 스마트 시간표 (생성 + 꼬리 질문 수정)
elif menu == "📅 스마트 시간표(수정가능)":
    st.subheader("📅 AI 맞춤형 시간표 설계")

    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("#### 1️⃣ 기본 정보 입력")
        major = st.text_input("학과", "전자융합공학과")
        c1, c2 = st.columns(2)
        grade = c1.selectbox("학년", ["1학년", "2학년", "3학년", "4학년"])
        semester = c2.selectbox("학기", ["1학기", "2학기"])
        target_credit = st.number_input("목표 학점", 9, 24, 18)
        requirements = st.text_area("추가 요구사항", placeholder="예: 전공 필수 위주로, 아침 수업 싫음")

    with col2:
        st.markdown("#### 2️⃣ 공강 시간 설정 (Click)")
        st.caption("✅ 체크된 시간은 '수업 가능', ⬜ 체크 해제한 시간은 '공강(수업 없음)'입니다.")
        
        # 시간표 Grid 생성 (기본값 True = 수업 가능)
        schedule_data = pd.DataFrame(
            True,
            index=[f"{i}교시" for i in range(1, 10)],
            columns=["월", "화", "수", "목", "금"]
        )
        
        # 데이터 에디터로 공강 선택 UI 구현
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

    # 생성 버튼
    if st.button("시간표 생성하기 ✨", type="primary", use_container_width=True):
        # 공강 시간 분석 (False인 값들 찾기)
        blocked_times = []
        for day in ["월", "화", "수", "목", "금"]:
            for period in edited_schedule.index:
                if not edited_schedule.loc[period, day]: # 체크 해제된 경우
                    blocked_times.append(f"{day}요일 {period}")
        
        blocked_desc = ", ".join(blocked_times) if blocked_times else "없음"
        add_log("user", f"[시간표 생성] {major} {grade}, 공강: {blocked_desc}")

        with st.spinner("시간표 조합 중..."):
            result = generate_timetable_ai(major, grade, semester, target_credit, blocked_desc, requirements)
            st.session_state.timetable_result = result # 결과 세션 저장
            add_log("assistant", "[시간표 생성 완료]")

    # 결과 표시 및 수정(꼬리 질문) 영역
    if st.session_state.timetable_result:
        st.divider()
        st.markdown("### 🗓️ 생성된 시간표")
        st.markdown(st.session_state.timetable_result, unsafe_allow_html=True)
        
        st.info("💡 시간표가 마음에 들지 않나요? 아래 채팅창에 수정 요청을 해보세요.")
        
        # 수정 요청 채팅창
        if refine_input := st.chat_input("수정 요청 (예: 화요일 1교시 수업 빼줘, C프로그래밍 교수님 바꿔줘)"):
            add_log("user", f"[시간표 수정] {refine_input}")
            with st.chat_message("user"):
                st.write(refine_input)
            
            with st.chat_message("assistant"):
                with st.spinner("시간표 수정 중..."):
                    new_result = refine_timetable_ai(st.session_state.timetable_result, refine_input)
                    st.session_state.timetable_result = new_result # 결과 덮어쓰기
                    st.markdown(new_result, unsafe_allow_html=True)
                    st.rerun() # 화면 갱신해서 수정된 시간표를 위로 올림

# 3. 강의평 분석
elif menu == "🔍 강의평 분석":
    st.subheader("🔍 강의평 팩트체크")
    
    col1, col2 = st.columns(2)
    c_name = col1.selectbox("과목명", REVIEW_DB['과목명'].unique())
    p_name = col2.selectbox("교수명", REVIEW_DB[REVIEW_DB['과목명'] == c_name]['교수명'].unique())
    
    # 강의평 분석용 챗 세션 키
    if "review_chat" not in st.session_state:
        st.session_state.review_chat = []

    if st.button("분석 시작"):
        # 초기 분석 수행
        reviews = REVIEW_DB[(REVIEW_DB['과목명']==c_name) & (REVIEW_DB['교수명']==p_name)]
        context = reviews.to_string()
        
        prompt = f"과목: {c_name}, 교수: {p_name}\n데이터: {context}\n이 강의의 장단점과 시험 스타일을 요약해줘."
        
        with st.spinner("분석 중..."):
            llm = get_llm()
            res = llm.invoke(prompt).content
            st.session_state.review_chat = [{"role": "assistant", "content": res, "context": context}]
            add_log("user", f"[강의평] {c_name} 분석 요청")

    # 대화 표시
    for msg in st.session_state.review_chat:
        if "role" in msg:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    # 강의평 관련 꼬리 질문
    if st.session_state.review_chat:
        if q_input := st.chat_input("더 궁금한 점이 있나요? (예: 과제 진짜 많아?)"):
            st.session_state.review_chat.append({"role": "user", "content": q_input})
            add_log("user", f"[강의평 질문] {q_input}")
            with st.chat_message("user"):
                st.write(q_input)
            
            with st.chat_message("assistant"):
                # 이전 맥락(강의평 데이터)을 포함하여 질문
                context_data = st.session_state.review_chat[0].get("context", "")
                llm = get_llm()
                ans = llm.invoke(f"강의평 데이터: {context_data}\n질문: {q_input}\n데이터에 기반해서 답변해.").content
                st.write(ans)
                st.session_state.review_chat.append({"role": "assistant", "content": ans})
                add_log("assistant", f"[강의평 답변] {ans[:20]}...")

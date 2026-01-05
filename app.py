import streamlit as st
import pandas as pd
import os
import glob
import datetime
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# -----------------------------------------------------------------------------
# [0] 설정 및 데이터 로드
# -----------------------------------------------------------------------------
st.set_page_config(page_title="KW-강의마스터 Pro", page_icon="🎓", layout="wide")

# API Key 로드
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = os.environ.get("GOOGLE_API_KEY", "")

if not api_key:
    st.error("🚨 **Google API Key가 설정되지 않았습니다.**")
    st.stop()

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

# HTML 코드 정제 함수
def clean_html_output(text):
    cleaned = text.strip()
    if cleaned.startswith("```html"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.replace("```html", "").replace("```", "").strip()

# ★ 재시도(Retry) 로직 ★
def run_with_retry(func, *args, **kwargs):
    max_retries = 5
    delays = [1, 2, 4, 8, 16]
    for i in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                if i < max_retries - 1:
                    time.sleep(delays[i])
                    continue
            raise e

# PDF 데이터 로드
@st.cache_resource(show_spinner="PDF 문서를 분석 중입니다...")
def load_knowledge_base():
    if not os.path.exists("data"):
        return ""
    pdf_files = glob.glob("data/*.pdf")
    if not pdf_files:
        return ""
    all_content = ""
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            pages = loader.load_and_split()
            filename = os.path.basename(pdf_file)
            all_content += f"\n\n--- [문서: {filename}] ---\n"
            for page in pages:
                all_content += page.page_content
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
            continue
    return all_content

PRE_LEARNED_DATA = load_knowledge_base()

# -----------------------------------------------------------------------------
# [1] AI 엔진
# -----------------------------------------------------------------------------
def get_llm():
    if not api_key: return None
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-09-2025", temperature=0)

def ask_ai(question):
    llm = get_llm()
    if not llm: return "⚠️ API Key 오류"
    def _execute():
        chain = PromptTemplate.from_template(
            "문서 내용: {context}\n질문: {question}\n문서에 기반해 답변해줘. 답변할 때 근거가 되는 문서의 원문 내용을 반드시 \" \" (쌍따옴표) 안에 인용해서 포함해줘."
        ) | llm
        return chain.invoke({"context": PRE_LEARNED_DATA, "question": question}).content
    try:
        return run_with_retry(_execute)
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            return "⚠️ **잠시만요!** 사용량이 많아 AI가 숨을 고르고 있습니다. 1분 뒤에 다시 시도해주세요."
        return f"❌ AI 오류: {str(e)}"

# 공통 프롬프트 지시사항
COMMON_TIMETABLE_INSTRUCTION = """
[★★★ 핵심 알고리즘: 2단계 검증 프로세스 (Strict Verification) ★★★]

1. **Step 1: 요람(Curriculum) 기반 '수강 대상' 리스트 확정**:
   - 먼저 PDF 요람 문서에서 **'{major} {grade} {semester}'**에 배정된 **'표준 이수 과목' 목록**을 추출하세요.
   - **주의:** 'MSC 필수', '공학인증 필수'라고 적혀 있어도, 이 학기(예: 1학년 1학기) 표에 없으면 리스트에 넣지 마세요.

2. **Step 2: 시간표 데이터와 정밀 대조 (Exact Match Only)**:
   - Step 1에서 확정된 리스트에 있는 과목만 시간표 데이터에서 찾으세요.
   - **과목명 완전 일치 필수**: 예: '대학물리학1' vs '대학물리및실험1' 구분.
   - **학년 불일치 자동 탈락 (Red Line)**: 사용자가 '1학년'인데 대상이 '2'면 제외.

3. **출력 형식 (세로형 HTML Table)**:
   - 반드시 **HTML `<table>` 태그**를 사용해라.
   - **행(Row): 1교시 ~ 9교시** (행 머리글에 시간 포함: 1교시 (09:00~10:15) 등)
   - **열(Column): 월, 화, 수, 목, 금, 토, 일** (7일 모두 표시)
   - **스타일 규칙**:
     - `table` 태그에 `width="100%"` 속성을 주어라.
     - **같은 과목은 반드시 같은 배경색**을 사용해라. (파스텔톤 권장)
     - **수업이 없는 빈 시간(공강)은 반드시 흰색 배경**으로 둬라.
     - 셀 내용: `<b>과목명</b><br><small>교수명 (대상학년)</small>`

4. **온라인 및 원격 강의 처리 (필수)**:
   - 강의 시간이 **'온라인', '원격', 'Cyber', '시간 미지정'** 등이면 **시간표 표(Table)에 넣지 말고**, 표 아래 `### 💻 온라인/원격 강의 목록` 섹션에 리스트로 출력.

5. **출력 순서 고정**:
   - 1순위: HTML 시간표 표
   - 2순위: "### 💻 온라인/원격 강의 목록"
   - 3순위: "### ✅ 필수 과목 검증"
   - 4순위: "### ⚠️ 배치 실패 목록"
"""

# 시간표 생성 함수
def generate_timetable_ai(major, grade, semester, target_credits, blocked_times_desc, requirements):
    llm = get_llm()
    if not llm: return "⚠️ API Key 오류"
    def _execute():
        template = """
        너는 대학교 수강신청 전문가야. 오직 제공된 [학습된 문서]의 텍스트 데이터에 기반해서만 시간표를 짜줘.

        [학생 정보]
        - 소속: {major}
        - 학년/학기: {grade} {semester}
        - 목표: {target_credits}학점
        - 공강 필수 시간: {blocked_times} (이 시간은 수업 배치 절대 금지)
        - 추가요구: {requirements}

        """ + COMMON_TIMETABLE_INSTRUCTION + """

        [추가 지시사항]
        - **HTML 코드를 마크다운 코드 블록(```html)으로 감싸지 마라.** 그냥 Raw HTML 텍스트로 출력해라.

        [학습된 문서]
        {context}
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "major", "grade", "semester", "target_credits", "blocked_times", "requirements"])
        chain = prompt | llm
        input_data = {
            "context": PRE_LEARNED_DATA,
            "major": major,
            "grade": grade,
            "semester": semester,
            "target_credits": target_credits,
            "blocked_times": blocked_times_desc,
            "requirements": requirements
        }
        return chain.invoke(input_data).content
    try:
        response_content = run_with_retry(_execute)
        return clean_html_output(response_content)
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            return "⚠️ **사용량 초과**: 잠시 후 다시 시도해주세요."
        return f"❌ AI 오류: {str(e)}"

# [수정됨] 상담 함수: 필요한 변수들을 명확히 전달하도록 수정
def chat_with_timetable_ai(current_timetable, user_input, major, grade, semester):
    llm = get_llm()
    def _execute():
        template = """
        너는 현재 시간표에 대한 상담을 해주는 AI 조교야.
        
        [현재 시간표 상태]
        {current_timetable}

        [사용자 입력]
        "{user_input}"

        [학생 정보]
        - 소속: {major}
        - 학년/학기: {grade} {semester}

        [지시사항]
        사용자의 입력 의도를 파악해서 아래 두 가지 중 하나로 반응해.
        
        **Case 1. 시간표 수정 요청인 경우 (예: "1교시 빼줘", "교수 바꿔줘"):**
        - 시간표를 **재작성**해줘.
        """ + COMMON_TIMETABLE_INSTRUCTION + """
        - **HTML 코드를 마크다운 코드 블록(```html)으로 감싸지 마라.** Raw HTML로 출력해.
        
        **Case 2. 과목에 대한 단순 질문인 경우 (예: "이거 선수과목 뭐야?"):**
        - **시간표를 다시 출력하지 말고**, 질문에 대한 **텍스트 답변**만 해.
        - **답변할 때 근거가 되는 문서의 원문 내용을 반드시 " " (쌍따옴표) 안에 인용해서 포함해줘.**
        
        답변 시작에 [수정] 또는 [답변] 태그를 붙여서 구분해줘.

        [학습된 문서]
        {context}
        """
        # input_variables에 COMMON_TIMETABLE_INSTRUCTION 내부의 변수(major, grade, semester)도 포함됨
        prompt = PromptTemplate(template=template, input_variables=["current_timetable", "user_input", "major", "grade", "semester", "context"])
        chain = prompt | llm
        
        # [핵심] invoke 호출 시 빠진 변수가 없도록 모두 전달
        return chain.invoke({
            "current_timetable": current_timetable, 
            "user_input": user_input,
            "major": major,
            "grade": grade,
            "semester": semester,
            "context": PRE_LEARNED_DATA
        }).content
    
    try:
        response_content = run_with_retry(_execute)
        if "[수정]" in response_content:
            parts = response_content.split("[수정]", 1)
            if len(parts) > 1:
                return "[수정]" + clean_html_output(parts[1])
            else:
                return clean_html_output(response_content)
        return response_content
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            return "⚠️ **사용량 초과**: 잠시 후 다시 시도해주세요."
        return f"❌ AI 오류: {str(e)}"

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
    if PRE_LEARNED_DATA:
         st.success(f"✅ PDF 문서 학습 완료")
    else:
        st.error("⚠️ 데이터 폴더에 PDF 파일이 없습니다.")

menu = st.radio("기능 선택", ["🤖 AI 학사 지식인", "📅 스마트 시간표(수정가능)"], 
                horizontal=True, key="menu_radio", 
                index=["🤖 AI 학사 지식인", "📅 스마트 시간표(수정가능)"].index(st.session_state.current_menu))

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
    timetable_area = st.empty()
    if st.session_state.timetable_result:
        with timetable_area.container():
            st.markdown("### 🗓️ 내 시간표")
            st.markdown(st.session_state.timetable_result, unsafe_allow_html=True)
            st.divider()

    with st.expander("시간표 설정 열기/닫기", expanded=not bool(st.session_state.timetable_result)):
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("#### 1️⃣ 기본 정보")
            kw_departments = [
                "전자융합공학과", "전자공학과", "전자통신공학과", "전기공학과", 
                "전자재료공학과", "로봇학부", "컴퓨터정보공학부", "소프트웨어학부", 
                "정보융합학부", "건축학과", "건축공학과", "화학공학과", "환경공학과"
            ]
            major = st.selectbox("학과", kw_departments, key="tt_major")
            c1, c2 = st.columns(2)
            grade = c1.selectbox("학년", ["1학년", "2학년", "3학년", "4학년"], key="tt_grade")
            semester = c2.selectbox("학기", ["1학기", "2학기"], key="tt_semester")
            target_credit = st.number_input("목표 학점", 9, 24, 18, key="tt_credit")
            requirements = st.text_area("추가 요구사항", placeholder="예: 전공 필수 챙겨줘", key="tt_req")

        with col2:
            st.markdown("#### 2️⃣ 공강 시간 설정")
            st.info("✅ **체크된 시간**: 수업 가능 (기본)  \n⬜ **체크 해제**: 공강 (수업 배정 안 함)")
            kw_times = {
                "1교시": "09:00~10:15", "2교시": "10:30~11:45", "3교시": "12:00~13:15",
                "4교시": "13:30~14:45", "5교시": "15:00~16:15", "6교시": "16:30~17:45",
                "7교시": "18:00~19:15", "8교시": "19:25~20:40", "9교시": "20:50~22:05"
            }
            schedule_index = [f"{k} ({v})" for k, v in kw_times.items()]
            if "init_schedule_df" not in st.session_state:
                st.session_state.init_schedule_df = pd.DataFrame(True, index=schedule_index, columns=["월", "화", "수", "목", "금"])
            edited_schedule = st.data_editor(
                st.session_state.init_schedule_df,
                column_config={
                    "월": st.column_config.CheckboxColumn("월", default=True),
                    "화": st.column_config.CheckboxColumn("화", default=True),
                    "수": st.column_config.CheckboxColumn("수", default=True),
                    "목": st.column_config.CheckboxColumn("목", default=True),
                    "금": st.column_config.CheckboxColumn("금", default=True),
                },
                height=360,
                use_container_width=True,
                key="tt_editor"
            )

        if st.button("시간표 생성하기 ✨", type="primary", use_container_width=True):
            blocked_times = []
            for day in ["월", "화", "수", "목", "금"]:
                for idx, period_label in enumerate(edited_schedule.index):
                    if not edited_schedule.iloc[idx][day]:
                        blocked_times.append(f"{day}요일 {period_label}")
            blocked_desc = ", ".join(blocked_times) if blocked_times else "없음"
            with st.spinner("선수과목 확인 및 시간표 조합 중... (최대 1분 소요될 수 있습니다)"):
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
                    # [수정됨] 필요한 변수들 전달 확인
                    response = chat_with_timetable_ai(st.session_state.timetable_result, chat_input, major, grade, semester)
                    if "[수정]" in response:
                        new_timetable = response.replace("[수정]", "").strip()
                        new_timetable = clean_html_output(new_timetable) 
                        st.session_state.timetable_result = new_timetable
                        with timetable_area.container():
                            st.markdown("### 🗓️ 내 시간표")
                            st.markdown(new_timetable, unsafe_allow_html=True)
                            st.divider()
                        success_msg = "시간표를 수정했습니다. 위쪽 표가 업데이트 되었습니다."
                        st.write(success_msg)
                        st.session_state.timetable_chat_history.append({"role": "assistant", "content": success_msg})
                    else:
                        clean_response = response.replace("[답변]", "").strip()
                        st.markdown(clean_response)
                        st.session_state.timetable_chat_history.append({"role": "assistant", "content": clean_response})

import streamlit as st
import pandas as pd
import os
import glob
import datetime
import time
import base64
import re  # 정규표현식 사용
import json # JSON 처리를 위한 라이브러리
import hashlib # 색상 생성을 위한 해시 라이브러리
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Firebase 라이브러리 (Admin SDK)
import firebase_admin
from firebase_admin import credentials, firestore

# -----------------------------------------------------------------------------
# [0] 설정 및 데이터 로드
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="KW-Plan: AI 학사 설계",
    page_icon="🦄",
    layout="wide"
)

# 스타일 설정
def set_style():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(180deg, #FFFFFF 0%, #FFF0F5 100%) !important;
            background-attachment: fixed !important;
        }
        h1 { color: #8A1538 !important; font-family: 'Pretendard', sans-serif; font-weight: 800; }
        
        div.row-widget.stRadio > div { justify-content: center; gap: 15px; }
        div.row-widget.stRadio > div[role="radiogroup"] > label {
            background-color: white; border: 2px solid #E9ECEF; padding: 10px 20px;
            border-radius: 12px; font-weight: bold; color: #495057;
            box-shadow: 0 2px 4px rgba(0,0,0,0.03); transition: all 0.2s;
        }
        div.row-widget.stRadio > div[role="radiogroup"] > label:hover,
        div.row-widget.stRadio > div[role="radiogroup"] > label[data-checked="true"] {
            border-color: #8A1538; background-color: #FFF5F7; color: #8A1538;
        }

        /* 채팅 입력창 디자인 */
        [data-testid="stChatInput"] { background-color: transparent !important; border-color: transparent !important; }
        [data-testid="stChatInput"] > div { background-color: transparent !important; border-color: transparent !important; box-shadow: none !important; }
        [data-testid="stBottom"] { background-color: transparent !important; border: none !important; box-shadow: none !important; }
        
        textarea[data-testid="stChatInputTextArea"] {
            background-color: #FFFFFF !important; border: 2px solid #8A1538 !important;
            border-radius: 30px !important; min-height: 50px !important; height: 50px !important;
            padding-top: 12px !important; padding-bottom: 12px !important; padding-right: 50px !important;
            box-shadow: 0 4px 12px rgba(138, 21, 56, 0.1) !important; color: #333333 !important;
        }
        
        [data-testid="stChatInputSubmitButton"] {
            background-color: transparent !important; color: #8A1538 !important;
            position: absolute !important; top: 50% !important; right: 10px !important;
            transform: translateY(-50%) !important; border: none !important; z-index: 99 !important;
        }

        footer { visibility: hidden; }
        @media only screen and (max-width: 600px) {
            .main .block-container { padding-left: 0.2rem !important; padding-right: 0.2rem !important; }
        }
        div[data-testid="stMarkdownContainer"] table { width: 100% !important; table-layout: fixed !important; }
        </style>
    """, unsafe_allow_html=True)

set_style()

# API Key 로드
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = os.environ.get("GOOGLE_API_KEY", "")

if not api_key:
    st.error("🚨 **Google API Key가 설정되지 않았습니다.**")
    st.stop()

# 세션 상태 초기화
if "global_log" not in st.session_state: st.session_state.global_log = [] 
if "timetable_result" not in st.session_state: st.session_state.timetable_result = "" 
if "chat_history" not in st.session_state: st.session_state.chat_history = [] 
if "current_menu" not in st.session_state: st.session_state.current_menu = "🤖 AI 학사 지식인"
if "menu_radio" not in st.session_state: st.session_state["menu_radio"] = "🤖 AI 학사 지식인"
if "timetable_chat_history" not in st.session_state: st.session_state.timetable_chat_history = []
if "graduation_analysis_result" not in st.session_state: st.session_state.graduation_analysis_result = ""
if "graduation_chat_history" not in st.session_state: st.session_state.graduation_chat_history = []
if "user" not in st.session_state: st.session_state.user = None
if "current_timetable_meta" not in st.session_state: st.session_state.current_timetable_meta = {}
if "selected_syllabus" not in st.session_state: st.session_state.selected_syllabus = None

def add_log(role, content, menu_context=None):
    timestamp = datetime.datetime.now().strftime("%H:%M")
    st.session_state.global_log.append({"role": role, "content": content, "time": timestamp, "menu": menu_context})

def run_with_retry(func, *args, **kwargs):
    max_retries = 5
    delays = [1, 2, 4, 8, 16]
    for i in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "503" in error_msg:
                if i < max_retries - 1:
                    time.sleep(delays[i])
                    continue
            raise e

# -----------------------------------------------------------------------------
# [Firebase Manager]
# -----------------------------------------------------------------------------
class FirebaseManager:
    def __init__(self):
        self.db = None
        self.is_initialized = False
        self.init_firestore()

    def init_firestore(self):
        if "firebase_service_account" in st.secrets:
            try:
                if not firebase_admin._apps:
                    cred_info = dict(st.secrets["firebase_service_account"])
                    cred = credentials.Certificate(cred_info)
                    firebase_admin.initialize_app(cred)
                self.db = firestore.client()
                self.is_initialized = True
            except Exception:
                pass

    def login(self, email, password):
        if not self.is_initialized: return None, "Firebase 연결 실패"
        try:
            users_ref = self.db.collection('users')
            query = users_ref.where('email', '==', email).where('password', '==', password).stream()
            for doc in query:
                user_data = doc.to_dict()
                user_data['localId'] = doc.id
                return user_data, None
            return None, "이메일 또는 비밀번호 불일치"
        except Exception as e: return None, str(e)

    def signup(self, email, password):
        if not self.is_initialized: return None, "Firebase 연결 실패"
        try:
            users_ref = self.db.collection('users')
            existing = list(users_ref.where('email', '==', email).stream())
            if existing: return None, "이미 가입된 이메일"
            new_ref = users_ref.document()
            data = {"email": email, "password": password, "created_at": firestore.SERVER_TIMESTAMP}
            new_ref.set(data)
            data['localId'] = new_ref.id
            return data, None
        except Exception as e: return None, str(e)

    def save_data(self, collection, doc_id, data):
        if not self.is_initialized or not st.session_state.user: return False
        try:
            uid = st.session_state.user['localId']
            doc_ref = self.db.collection('users').document(uid).collection(collection).document(doc_id)
            data['updated_at'] = firestore.SERVER_TIMESTAMP
            doc_ref.set(data)
            return True
        except: return False

    def load_collection(self, collection):
        if not self.is_initialized or not st.session_state.user: return []
        try:
            uid = st.session_state.user['localId']
            docs = self.db.collection('users').document(uid).collection(collection).order_by('updated_at', direction=firestore.Query.DESCENDING).stream()
            return [{"id": doc.id, **doc.to_dict()} for doc in docs]
        except: return []

fb_manager = FirebaseManager()

# PDF 로드
@st.cache_resource(show_spinner="PDF 문서를 분석 중입니다...")
def load_knowledge_base():
    if not os.path.exists("data"): return ""
    pdf_files = glob.glob("data/*.pdf")
    if not pdf_files: return ""
    all_content = ""
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            pages = loader.load_and_split()
            filename = os.path.basename(pdf_file)
            all_content += f"\n\n--- [문서: {filename}] ---\n"
            for page in pages: all_content += page.page_content
        except: continue
    return all_content

PRE_LEARNED_DATA = load_knowledge_base()

# -----------------------------------------------------------------------------
# [1] AI 엔진 및 로직 함수
# -----------------------------------------------------------------------------
def get_llm():
    if not api_key: return None
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-09-2025", temperature=0)

def get_pro_llm():
    if not api_key: return None
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-09-2025", temperature=0)

def ask_ai(question):
    llm = get_llm()
    if not llm: return "⚠️ API Key 오류"
    history_context = ""
    if "chat_history" in st.session_state:
        for msg in st.session_state.chat_history[-10:]:
            role = "사용자" if msg["role"] == "user" else "AI"
            history_context += f"{role}: {msg['content']}\n"

    def _execute():
        template = """
        너는 광운대학교 학사 정보를 안내하는 AI 조교야.
        [이전 대화 내역] {history}
        [문서 내용] {context}
        [현재 질문] {question}
        답변 가이드: 문서 내용을 근거로 " " 안에 인용하며 답변해.
        """
        prompt = PromptTemplate(template=template, input_variables=["history", "context", "question"])
        chain = prompt | llm
        return chain.invoke({"history": history_context, "context": PRE_LEARNED_DATA, "question": question}).content
    try: return run_with_retry(_execute)
    except Exception as e: return f"❌ AI 오류: {str(e)}"

# =============================================================================
# 시간 충돌 감지 로직 (온라인 강의 예외 처리 유지)
# =============================================================================
def check_time_conflict(new_course, current_schedule):
    new_slots = new_course.get('time_slots', [])
    
    # 온라인 강의(시간미정 또는 빈 리스트)는 충돌 검사 제외
    if not new_slots or new_slots == ["시간미정"] or not isinstance(new_slots, list):
        return False, None

    new_slots_set = set(new_slots)

    for existing in current_schedule:
        existing_slots = existing.get('time_slots', [])
        # 기존 강의가 온라인이면 충돌 아님
        if not existing_slots or existing_slots == ["시간미정"]:
            continue
            
        existing_slots_set = set(existing_slots)
        overlap = new_slots_set & existing_slots_set
        if overlap:
            return True, existing['name']
    return False, None

# =============================================================================
# HTML 시간표 렌더러 (색상 구분 유지)
# =============================================================================
def render_interactive_timetable(schedule_list):
    days = ["월", "화", "수", "목", "금"]
    table_grid = {i: {d: {"text": "", "color": "#ffffff"} for d in days} for i in range(1, 10)}
    online_courses = []

    # 파스텔톤 색상 팔레트 정의
    pastel_colors = [
        "#FFEBEE", "#E3F2FD", "#F3E5F5", "#E8F5E9", "#FFF3E0", 
        "#FBE9E7", "#E0F7FA", "#FFF8E1", "#F1F8E9", "#E1F5FE",
        "#FCE4EC", "#E8EAF6", "#E0F2F1", "#FFECB3", "#D7CCC8"
    ]

    # 과목명 -> 색상 매핑 함수 (Hash 사용)
    def get_color_for_course(name):
        hash_val = int(hashlib.sha256(name.encode('utf-8')).hexdigest(), 16)
        return pastel_colors[hash_val % len(pastel_colors)]

    for course in schedule_list:
        slots = course.get('time_slots', [])
        c_name = course['name']
        c_prof = course['professor']
        
        # 색상 결정
        bg_color = get_color_for_course(c_name)

        if not slots or slots == ["시간미정"] or not isinstance(slots, list):
            online_courses.append({"name": c_name, "color": bg_color})
            continue

        for slot in slots:
            if len(slot) < 2: continue
            day_char = slot[0] # "월"
            try:
                period = int(slot[1:]) # "3"
                if day_char in days and 1 <= period <= 9:
                    content = f"<b>{c_name}</b><br><small>{c_prof}</small>"
                    # 그리드에 내용과 색상 저장
                    table_grid[period][day_char] = {"text": content, "color": bg_color}
            except: pass

    html = """
    <table border="1" width="100%" style="border-collapse: collapse; text-align: center; font-size: 12px; border-color: #ddd;">
        <tr style="background-color: #f8f9fa;">
            <th width="10%">교시</th><th width="18%">월</th><th width="18%">화</th><th width="18%">수</th><th width="18%">목</th><th width="18%">금</th>
        </tr>
    """
    
    for i in range(1, 10):
        html += f"<tr><td style='background-color: #f8f9fa; font-weight:bold;'>{i}</td>"
        for day in days:
            cell = table_grid[i][day]
            # 셀 별 고유 색상 적용
            bg_color = cell["color"]
            content = cell["text"]
            border_style = "border: 1px solid #ddd;"
            html += f"<td style='background-color: {bg_color}; {border_style} height: 45px; vertical-align: middle;'>{content}</td>"
        html += "</tr>"

    if online_courses:
        # 온라인 강의도 색상 박스로 표시
        online_html_list = []
        for oc in online_courses:
            online_html_list.append(
                f"<span style='background-color:{oc['color']}; padding:2px 6px; border-radius:4px; margin-right:5px;'>{oc['name']}</span>"
            )
        online_text = " ".join(online_html_list)
        html += f"<tr><td style='background-color: #f8f9fa;'><b>온라인</b></td><td colspan='5' style='text-align: left; padding: 8px;'>{online_text}</td></tr>"
        
    html += "</table>"
    return html

# =============================================================================
# [핵심 수정] AI 후보군 추출 (학번 전달 및 교양/MSC 로직 전면 수정)
# =============================================================================
def get_course_candidates_json(major, grade, semester, student_id, diagnosis_text=""):
    llm = get_llm()
    if not llm: return []

    # 프롬프트: 교양 학년 제한 해제 + MSC 학번별 체계 참조 지시
    prompt_template = """
    너는 [대학교 수강신청 자료집 정밀 분석기]이다. 
    제공된 문서를 바탕으로 **{major} {student_id} ({grade} {semester})** 학생이 수강 가능한 과목 리스트를 JSON으로 추출하라.
    
    [분석 기준 및 검증 절차]
    1. **MSC 및 전공 기초 (필수 탐색):** - 문서 내의 **[학과별 교육과정표]** 또는 **[MSC 지정 현황]** 페이지를 찾아라.
       - **{major} {student_id}** 기준, 1학년 또는 해당 학기에 반드시 들어야 하는 MSC(수학/과학/전산) 필수 과목을 찾아 **Classification="MSC필수", Priority="High"**로 설정하라.
       - 예: 미분적분학, 대학물리, C프로그래밍, 화학 등이 해당될 수 있음.
       
    2. **교양 과목 (학년 제한 해제):**
       - **중요:** 교양 과목(균형교양, 핵심교양, 일반교양 등)은 학정번호 앞자리가 학년을 의미하더라도, **타 학년이 수강 가능하므로 절대 필터링하지 말고 모두 포함하라.**
       - 단, 문서의 **[수강신청 유의사항]**을 확인하여 "동일 영역/난이도 중복 수강 불가" 같은 제약이 있다면 `reason` 필드에 경고를 적어라.

    3. **전공 과목:**
       - 해당 학과, 해당 학년의 전공 필수/선택 과목을 모두 포함하라.

    [JSON 출력 필드 작성 규칙]
    - classification: "전공필수", "전공선택", "MSC필수", "교양필수", "균형교양", "일반교양" 중 택 1
    - priority: 필수/MSC/재수강="High", 전공="Medium", 그 외="Normal"
    - reason: 팩트 위주 기재 (예: "MSC필수 | 3학점", "균형교양(자연) | 동일난이도 주의")

    [입력 정보]
    - 학과: {major}
    - 학번/학년/학기: {student_id} / {grade} {semester}
    - 진단 결과(재수강): {diagnosis_context}

    **오직 JSON 리스트만 출력하라.**
    [문서 데이터]
    {context}
    """
    
    def _execute():
        chain = PromptTemplate.from_template(prompt_template) | llm
        return chain.invoke({
            "major": major, "grade": grade, "semester": semester,
            "student_id": student_id, # 학번 정보 전달
            "diagnosis_context": diagnosis_text, "context": PRE_LEARNED_DATA
        }).content

    try:
        response = run_with_retry(_execute)
        cleaned_json = response.replace("```json", "").replace("```", "").strip()
        if not cleaned_json.startswith("["):
             start = cleaned_json.find("[")
             end = cleaned_json.rfind("]")
             if start != -1 and end != -1:
                 cleaned_json = cleaned_json[start:end+1]
        return json.loads(cleaned_json)
    except Exception as e:
        print(f"JSON Parsing Error: {e}")
        return []

def chat_with_timetable_ai(current_timetable, user_input, major, grade, semester):
    llm = get_llm()
    def _execute():
        template = """
        [현재 시간표] {current_timetable}
        [사용자 입력] "{user_input}"
        [학생 정보] {major} {grade} {semester}
        문서 내용을 바탕으로 답변해.
        [학습된 문서] {context}
        """
        prompt = PromptTemplate(template=template, input_variables=["current_timetable", "user_input", "major", "grade", "semester", "context"])
        chain = prompt | llm
        return chain.invoke({"current_timetable": current_timetable, "user_input": user_input, "major": major, "grade": grade, "semester": semester, "context": PRE_LEARNED_DATA}).content
    try: return run_with_retry(_execute)
    except Exception as e: return f"❌ AI 오류: {str(e)}"

# =============================================================================
# [섹션] 성적 및 진로 진단 분석 함수
# =============================================================================
def analyze_graduation_requirements(uploaded_images):
    llm = get_pro_llm()
    if not llm: return "⚠️ API Key 오류"
    def encode_image(image_file):
        image_file.seek(0)
        return base64.b64encode(image_file.read()).decode("utf-8")
    image_messages = []
    for img_file in uploaded_images:
        base64_image = encode_image(img_file)
        image_messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

    def _execute():
        prompt = """
        당신은 [대기업 인사담당자 출신 취업 컨설턴트]입니다. 성적표와 문서를 바탕으로 분석하세요.
        [[SECTION:GRADUATION]] ### 🎓 1. 졸업 요건 진단
        [[SECTION:GRADES]] ### 📊 2. 성적 정밀 분석 (전공 평점, 재수강 필요 등)
        [[SECTION:CAREER]] ### 💼 3. AI 커리어 솔루션 (대기업 직무 매칭)
        """
        content_list = [{"type": "text", "text": prompt}]
        content_list.extend(image_messages)
        content_list.append({"type": "text", "text": f"\n\n{PRE_LEARNED_DATA}"})
        message = HumanMessage(content=content_list)
        response = llm.invoke([message])
        return response.content
    try: return run_with_retry(_execute)
    except Exception as e: return f"❌ AI 오류: {str(e)}"

def chat_with_graduation_ai(current_analysis, user_input):
    llm = get_llm()
    def _execute():
        template = """
        당신은 독설적인 'AI 취업 컨설턴트'입니다.
        [진단 결과] {current_analysis}
        [질문] "{user_input}"
        현실적이고 직설적으로 답변하세요.
        [참고] {context}
        """
        prompt = PromptTemplate(template=template, input_variables=["current_analysis", "user_input", "context"])
        chain = prompt | llm
        return chain.invoke({"current_analysis": current_analysis, "user_input": user_input, "context": PRE_LEARNED_DATA}).content
    try: return run_with_retry(_execute)
    except Exception as e: return f"❌ AI 오류: {str(e)}"

# -----------------------------------------------------------------------------
# [2] UI 구성
# -----------------------------------------------------------------------------
def change_menu(menu_name):
    st.session_state.current_menu = menu_name

with st.sidebar:
    st.title("🗂️ 활동 로그")
    if st.session_state.user is None:
        with st.expander("🔐 로그인 / 회원가입", expanded=True):
            auth_mode = st.radio("모드 선택", ["로그인", "회원가입"], horizontal=True)
            email = st.text_input("이메일")
            password = st.text_input("비밀번호", type="password")
            if st.button(auth_mode):
                if not email or not password: st.error("정보 입력 필요")
                else:
                    if not fb_manager.is_initialized: st.error("Firebase 미설정")
                    else:
                        with st.spinner("처리 중..."):
                            if auth_mode == "로그인": user, err = fb_manager.login(email, password)
                            else: user, err = fb_manager.signup(email, password)
                            if user:
                                st.session_state.user = user
                                st.success(f"환영합니다! ({user['email']})")
                                st.rerun()
                            else: st.error(f"오류: {err}")
    else:
        st.info(f"👤 **{st.session_state.user['email']}**님")
        if st.button("로그아웃"):
            st.session_state.clear()
            st.session_state["menu_radio"] = "🤖 AI 학사 지식인" 
            st.rerun()
    
    st.divider()
    st.subheader("⚙️ 시스템 관리자 모드")
    if st.button("📡 학교 서버 데이터 동기화"):
        with st.spinner("서버 동기화 중..."):
            time.sleep(2)
            st.cache_resource.clear()
            st.success("동기화 완료")
            st.rerun()
            
    st.divider()
    log_container = st.container(height=300)
    with log_container:
        if not st.session_state.global_log: st.info("기록 없음")
        else:
            for i, log in enumerate(reversed(st.session_state.global_log)):
                label = f"[{log['time']}] {log['content'][:15]}..."
                if st.button(label, key=f"log_btn_{i}", use_container_width=True):
                    if log['menu']:
                        st.session_state.current_menu = log['menu']
                        st.session_state["menu_radio"] = log['menu'] 
                        st.rerun()
    st.divider()
    if PRE_LEARNED_DATA: st.success(f"✅ PDF 문서 학습 완료")
    else: st.error("⚠️ 데이터 폴더에 PDF 파일이 없습니다.")

# 메인 UI
st.markdown("<h1 style='text-align: center; color: #8A1538;'>🦄 Kwangwoon AI Planner</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #666;'>광운대학교 학생을 위한 지능형 수강설계 에이전트</h5>", unsafe_allow_html=True)
st.write("") 

_, col_center, _ = st.columns([1, 4, 1])
with col_center:
    menu = st.radio(
        "메뉴 선택",
        options=["🤖 AI 학사 지식인", "📅 스마트 시간표(수정가능)", "📈 성적 및 진로 진단"],
        index=0, horizontal=True, key="menu_radio", label_visibility="collapsed"
    )

if menu != st.session_state.current_menu:
    st.session_state.current_menu = menu
    st.rerun()

st.divider()

if st.session_state.current_menu == "🤖 AI 학사 지식인":
    st.subheader("🤖 무엇이든 물어보세요")
    if st.session_state.user and fb_manager.is_initialized:
        with st.expander("💾 대화 내용 관리"):
            c1, c2 = st.columns(2)
            if c1.button("저장"):
                fb_manager.save_data('chat_history', str(int(time.time())), {"history": st.session_state.chat_history})
                st.toast("저장 완료")
            saved = fb_manager.load_collection('chat_history')
            if saved:
                sel = c2.selectbox("불러오기", saved, format_func=lambda x: datetime.datetime.fromtimestamp(int(x['id'])).strftime('%m-%d %H:%M'), label_visibility="collapsed")
                if c2.button("로드"):
                    st.session_state.chat_history = sel['history']
                    st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if user_input := st.chat_input("질문 입력"):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        add_log("user", f"[지식인] {user_input}", "🤖 AI 학사 지식인")
        with st.chat_message("user"): st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("답변 중..."):
                resp = ask_ai(user_input)
                st.markdown(resp)
        st.session_state.chat_history.append({"role": "assistant", "content": resp})

elif st.session_state.current_menu == "📅 스마트 시간표(수정가능)":
    st.subheader("📅 AI 스마트 시간표 빌더")
    if "candidate_courses" not in st.session_state: st.session_state.candidate_courses = []
    if "my_schedule" not in st.session_state: st.session_state.my_schedule = []

    with st.expander("🛠️ 수강신청 설정", expanded=not bool(st.session_state.candidate_courses)):
        kw_departments = [
            "전자공학과", "전자통신공학과", "전자융합공학과", "전기공학과", "전자재료공학과", "반도체시스템공학부", "로봇학부",
            "컴퓨터정보공학부", "소프트웨어학부", "정보융합학부", "지능형로봇학과", "건축학과", "건축공학과", "화학공학과", "환경공학과",
            "수학과", "전자바이오물리학과", "화학과", "스포츠융합과학과", "정보콘텐츠학과", "국어국문학과", "영어산업학과", 
            "미디어커뮤니케이션학부", "산업심리학과", "동북아문화산업학부", "행정학과", "법학부", "국제학부", "자산관리학과",
            "경영학부", "국제통상학부", "자율전공학부(자연)", "자율전공학부(인문)"
        ]
        
        # [UI 수정] 학번 선택 추가 (4단 컬럼)
        c1, c2, c3, c4 = st.columns(4)
        major = c1.selectbox("학과", kw_departments, key="tt_major")
        grade = c2.selectbox("학년", ["1학년", "2학년", "3학년", "4학년"], key="tt_grade")
        semester = c3.selectbox("학기", ["1학기", "2학기"], key="tt_semester")
        student_id = c4.selectbox("학번 (입학년도)", ["25학번", "24학번", "23학번", "22학번", "21학번 이전"], key="tt_student_id")
        
        use_diagnosis = st.checkbox("☑️ 성적 진단 결과 반영", value=True)
        
        if st.button("🚀 강의 목록 불러오기 (AI Scan)", type="primary", use_container_width=True):
            diag_text = ""
            if use_diagnosis and st.session_state.graduation_analysis_result: diag_text = st.session_state.graduation_analysis_result
            elif use_diagnosis and st.session_state.user: 
                saved = fb_manager.load_collection('graduation_diagnosis')
                if saved: diag_text = saved[0]['result']
            
            with st.spinner(f"수강신청 자료집에서 {major} {student_id} 교육과정을 분석 중입니다..."):
                # [함수 호출 수정] student_id 추가 전달
                candidates = get_course_candidates_json(major, grade, semester, student_id, diag_text)
                if candidates:
                    st.session_state.candidate_courses = candidates
                    st.session_state.my_schedule = [] 
                    st.rerun()
                else: st.error("강의 추출 실패")

    if st.session_state.candidate_courses:
        st.divider()
        col_left, col_right = st.columns([1, 1.4], gap="medium")
        with col_left:
            st.subheader("📚 강의 선택")
            st.caption("담은 과목은 목록에서 사라집니다.")
            with st.container(height=600, border=True):
                tab1, tab2, tab3 = st.tabs(["🔥 필수/재수강/MSC", "🏫 전공선택", "🧩 교양/기타"])
                
                def draw_course_row(course, key_prefix):
                    if any(c['name'] == course['name'] for c in st.session_state.my_schedule): return
                    
                    priority = course.get('priority', 'Normal')
                    reason_bg = "#ffebee" if priority == 'High' else "#e3f2fd" if priority == 'Medium' else "#f1f3f5"
                    
                    with st.container(border=True):
                        c_info, c_btn = st.columns([0.85, 0.15])
                        with c_info:
                            time_str = ', '.join(course['time_slots']) if course['time_slots'] else "시간미정"
                            st.markdown(f"<div style='line-height:1.2;'><span style='font-weight:bold; font-size:16px;'>{course['name']}</span> <span style='font-size:13px; color:#555;'>({course['credits']}학점) | {course['professor']} | {time_str}</span></div>", unsafe_allow_html=True)
                            if course.get('reason') or course.get('classification'):
                                tag = course.get('classification', course.get('reason'))
                                st.markdown(f"<div style='background-color:{reason_bg}; padding:2px 8px; border-radius:4px; font-size:12px; margin-top:4px; display:inline-block;'>💡 {tag}</div>", unsafe_allow_html=True)
                        with c_btn:
                            st.write("") 
                            if st.button("➕", key=f"ad_{key_prefix}_{course['id']}", type="primary"):
                                conflict, conflict_name = check_time_conflict(course, st.session_state.my_schedule)
                                if conflict: st.toast(f"⚠️ 시간 충돌! '{conflict_name}'", icon="🚫")
                                else:
                                    st.session_state.my_schedule.append(course)
                                    st.rerun()

                # 탭 분류 필터링 로직 강화 (교양필수 키워드 추가)
                must_keywords = ['필수', 'MSC', '기초', '핵심', '공통', '교양필수']
                
                must_list = [
                    c for c in st.session_state.candidate_courses 
                    if c.get('priority') == 'High' 
                    or '재수강' in c.get('reason', '')
                    or any(k in c.get('classification', '') for k in must_keywords)
                ]
                
                major_list = [
                    c for c in st.session_state.candidate_courses 
                    if (c.get('priority') == 'Medium' or '전공' in c.get('classification', '')) 
                    and c not in must_list
                ]
                
                other_list = [
                    c for c in st.session_state.candidate_courses 
                    if c not in must_list and c not in major_list
                ]

                with tab1:
                    if not must_list: st.info("해당 과목 없음")
                    for c in must_list: draw_course_row(c, "must")
                with tab2:
                    if not major_list: st.info("해당 과목 없음")
                    for c in major_list: draw_course_row(c, "mj")
                with tab3:
                    if not other_list: st.info("해당 과목 없음")
                    for c in other_list: draw_course_row(c, "ot")

        with col_right:
            st.subheader("🗓️ 내 시간표")
            if st.session_state.my_schedule:
                with st.expander("📋 신청 내역 관리", expanded=True):
                    for idx, added in enumerate(st.session_state.my_schedule):
                        cols = st.columns([0.8, 0.2])
                        cols[0].markdown(f"**{added['name']}**")
                        if cols[1].button("❌", key=f"del_{idx}"):
                            st.session_state.my_schedule.pop(idx)
                            st.rerun()
            
            html_table = render_interactive_timetable(st.session_state.my_schedule)
            st.markdown(html_table, unsafe_allow_html=True)
            st.divider()
            if st.button("💾 이대로 저장하기", use_container_width=True):
                if not st.session_state.my_schedule: st.error("과목 선택 필요")
                else:
                    data = {"result": html_table, "major": major, "grade": grade, "name": f"{major} {grade}", "created_at": datetime.datetime.now()}
                    if fb_manager.save_data('timetables', str(int(time.time())), data):
                        st.toast("저장 완료!", icon="✅")
                    else: st.warning("로그인 필요")

elif st.session_state.current_menu == "📈 성적 및 진로 진단":
    st.subheader("📈 성적 및 진로 정밀 진단")
    st.markdown("**성적표 이미지를 업로드하세요.** AI가 졸업 요건, 성적, 커리어를 분석합니다.")
    
    if st.session_state.user and fb_manager.is_initialized:
        with st.expander("📂 지난 진단 결과"):
            saved = fb_manager.load_collection('graduation_diagnosis')
            if saved:
                sel = st.selectbox("선택", saved, format_func=lambda x: datetime.datetime.fromtimestamp(int(x['id'])).strftime('%m-%d %H:%M'))
                if st.button("불러오기"):
                    st.session_state.graduation_analysis_result = sel['result']
                    st.rerun()

    upl = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if upl and st.button("진단 시작 🚀", type="primary"):
        with st.spinner("분석 중..."):
            res = analyze_graduation_requirements(upl)
            st.session_state.graduation_analysis_result = res
            st.session_state.graduation_chat_history = []
            add_log("user", "[진단] 이미지 분석", "📈 성적 및 진로 진단")
            st.rerun()

    if st.session_state.graduation_analysis_result:
        st.divider()
        res = st.session_state.graduation_analysis_result
        try:
            p1 = res.split("[[SECTION:GRADUATION]]")[1] if "[[SECTION:GRADUATION]]" in res else res
            sec_grad = p1.split("[[SECTION:GRADES]]")[0]
            p2 = p1.split("[[SECTION:GRADES]]")[1] if "[[SECTION:GRADES]]" in p1 else ""
            sec_grade = p2.split("[[SECTION:CAREER]]")[0]
            sec_career = p2.split("[[SECTION:CAREER]]")[1] if "[[SECTION:CAREER]]" in p2 else ""
        except: sec_grad, sec_grade, sec_career = res, "", ""

        t1, t2, t3 = st.tabs(["🎓 졸업 요건", "📊 성적 분석", "💼 커리어 솔루션"])
        with t1: st.markdown(sec_grad)
        with t2: st.markdown(sec_grade)
        with t3: st.markdown(sec_career)
        
        st.divider()
        if st.button("☁️ 결과 저장"):
            if fb_manager.save_data('graduation_diagnosis', str(int(time.time())), {"result": res}): st.toast("저장 완료")
        
        st.subheader("💬 컨설턴트와의 대화")
        for msg in st.session_state.graduation_chat_history:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        if ci := st.chat_input("질문 입력"):
            st.session_state.graduation_chat_history.append({"role": "user", "content": ci})
            add_log("user", f"[진단상담] {ci}", "📈 성적 및 진로 진단")
            with st.chat_message("user"): st.write(ci)
            with st.chat_message("assistant"):
                with st.spinner("분석 중..."):
                    rsp = chat_with_graduation_ai(res, ci)
                    if "[수정]" in rsp:
                        st.session_state.graduation_analysis_result = rsp.replace("[수정]", "").strip()
                        st.rerun()
                    else: st.markdown(rsp)
            st.session_state.graduation_chat_history.append({"role": "assistant", "content": rsp})

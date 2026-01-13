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
st.set_page_config(page_title="KW-강의마스터 Pro", page_icon="🎓", layout="wide")

# [모바일 최적화 CSS 및 컴팩트 뷰 스타일링]
st.markdown("""
    <style>
        footer { visibility: hidden; }
        /* 모바일 최적화 */
        @media only screen and (max-width: 600px) {
            .main .block-container {
                padding-left: 0.2rem !important;
                padding-right: 0.2rem !important;
                padding-top: 2rem !important;
                max-width: 100% !important;
            }
        }
        /* 시간표 테이블 스타일 */
        div[data-testid="stMarkdownContainer"] table {
            width: 100% !important;
            table-layout: fixed !important;
            display: table !important;
            font-size: 11px !important;
            margin-bottom: 0px !important;
            border-collapse: collapse !important;
        }
        div[data-testid="stMarkdownContainer"] th, 
        div[data-testid="stMarkdownContainer"] td {
            padding: 4px !important;
            word-wrap: break-word !important;
            word-break: break-all !important;
            white-space: normal !important;
            line-height: 1.3 !important;
            vertical-align: middle !important;
            border: 1px solid #ddd !important;
        }
        /* 버튼 높이 조정 */
        button[kind="primary"], button[kind="secondary"] {
            padding: 0.2rem 0.5rem !important;
            min-height: 0px !important;
            height: auto !important;
        }
        /* 진행률 바 스타일 */
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

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

# [추가 상태] 장바구니 및 학번
if "cart_courses" not in st.session_state: st.session_state.cart_courses = []
if "student_id_val" not in st.session_state: st.session_state.student_id_val = "24학번"

def add_log(role, content, menu_context=None):
    timestamp = datetime.datetime.now().strftime("%H:%M")
    st.session_state.global_log.append({
        "role": role,
        "content": content,
        "time": timestamp,
        "menu": menu_context
    })

# [3-2] 파스텔톤 색상 생성 함수
def get_pastel_color(text):
    hash_object = hashlib.md5(text.encode())
    hash_hex = hash_object.hexdigest()
    # R, G, B 값을 128~255 사이로 설정하여 파스텔톤 생성
    r = int(hash_hex[0:2], 16) % 127 + 128
    g = int(hash_hex[2:4], 16) % 127 + 128
    b = int(hash_hex[4:6], 16) % 127 + 128
    return f"#{r:02x}{g:02x}{b:02x}"

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
# [Firebase Manager] (원본 유지)
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
            return None, "이메일 또는 비밀번호가 일치하지 않습니다."
        except Exception as e:
            return None, f"로그인 오류: {str(e)}"

    def signup(self, email, password):
        if not self.is_initialized: return None, "Firebase 연결 실패"
        try:
            users_ref = self.db.collection('users')
            existing_user = list(users_ref.where('email', '==', email).stream())
            if len(existing_user) > 0: return None, "이미 가입된 이메일입니다."
            new_user_ref = users_ref.document()
            user_data = {"email": email, "password": password, "created_at": firestore.SERVER_TIMESTAMP}
            new_user_ref.set(user_data)
            user_data['localId'] = new_user_ref.id
            return user_data, None
        except Exception as e:
            return None, f"회원가입 오류: {str(e)}"

    def save_data(self, collection, doc_id, data):
        if not self.is_initialized or not st.session_state.user: return False
        try:
            user_id = st.session_state.user['localId']
            doc_ref = self.db.collection('users').document(user_id).collection(collection).document(doc_id)
            data['updated_at'] = firestore.SERVER_TIMESTAMP
            doc_ref.set(data)
            return True
        except: return False

    def load_collection(self, collection):
        if not self.is_initialized or not st.session_state.user: return []
        try:
            user_id = st.session_state.user['localId']
            docs = self.db.collection('users').document(user_id).collection(collection).order_by('updated_at', direction=firestore.Query.DESCENDING).stream()
            return [{"id": doc.id, **doc.to_dict()} for doc in docs]
        except: return []

fb_manager = FirebaseManager()

# [복구됨] PDF 데이터 로드 (원본 로직)
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
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)

def get_pro_llm():
    if not api_key: return None
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0)

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

# =============================================================================
# [Helper Functions] 로직 업데이트 (학번 반영, 장바구니, 파스텔톤)
# =============================================================================

def check_time_conflict(new_course, current_schedule):
    new_slots = set(new_course.get('time_slots', []))
    for existing in current_schedule:
        existing_slots = set(existing.get('time_slots', []))
        overlap = new_slots & existing_slots
        if "시간미정" in new_slots or "시간미정" in existing_slots: continue
        if overlap:
            return True, existing['name']
    return False, None

# [3-3] 온라인 전용 행 및 [3-2] 파스텔톤 적용
def render_interactive_timetable(schedule_list):
    days = ["월", "화", "수", "목", "금"]
    table_grid = {i: {d: "" for d in days} for i in range(1, 10)}
    online_courses = []

    for course in schedule_list:
        slots = course.get('time_slots', [])
        # 색상 할당 (없으면 생성)
        if 'color' not in course:
            course['color'] = get_pastel_color(course['name'])
        bg_color = course['color']

        # 온라인/시간미정 처리
        if not slots or slots == ["시간미정"] or not isinstance(slots, list):
            online_courses.append(course)
            continue

        for slot in slots:
            if len(slot) < 2: continue
            day_char = slot[0] 
            try:
                period = int(slot[1:]) 
                if day_char in days and 1 <= period <= 9:
                    # [2-2] 분반 정보 표시
                    content = f"<div style='background-color:{bg_color}; padding:4px; border-radius:4px; height:100%; box-shadow:1px 1px 2px rgba(0,0,0,0.1);'>" \
                              f"<b>{course['name']}</b><br>" \
                              f"<span style='font-size:10px;'>[{course.get('section', '?')}반]</span><br>" \
                              f"<small>{course['professor']}</small></div>"
                    table_grid[period][day_char] = content
            except:
                pass

    html = """
    <table border="1" width="100%" style="border-collapse: collapse; text-align: center; font-size: 12px; border-color: #ddd;">
        <tr style="background-color: #f8f9fa;">
            <th width="10%">교시</th><th width="18%">월</th><th width="18%">화</th><th width="18%">수</th><th width="18%">목</th><th width="18%">금</th>
        </tr>
    """
    
    for i in range(1, 10):
        html += f"<tr><td style='background-color: #f8f9fa; font-weight:bold;'>{i}</td>"
        for day in days:
            cell_content = table_grid[i][day]
            html += f"<td style='height: 50px; vertical-align: middle; padding:2px;'>{cell_content}</td>"
        html += "</tr>"

    # [3-3] 온라인 전용 행 신설
    if online_courses:
        online_items = []
        for oc in online_courses:
            online_items.append(f"<span style='background-color:{oc['color']}; padding:2px 6px; border-radius:4px; margin-right:5px;'>💻 {oc['name']} ({oc['professor']})</span>")
        
        online_html = " ".join(online_items)
        html += f"<tr><td style='background-color: #e3f2fd; font-weight:bold;'>온라인<br>/기타</td><td colspan='5' style='text-align: left; padding: 8px; background-color: #f1f8ff;'>{online_html}</td></tr>"
        
    html += "</table>"
    return html

# [1-1, 1-2, 1-3] AI 후보군 추출 (학번 로직 & MSC 강등 & 선수과목)
def get_course_candidates_json(major, grade, semester, student_id, diagnosis_text=""):
    llm = get_llm()
    if not llm: return []

    prompt_template = """
    너는 [대학교 학사 데이터베이스 파서]이다. 
    제공된 [수강신청자료집/시간표 문서]를 분석하여 **{major} {student_id}** 학생이 {grade} {semester}에 수강 가능한 모든 과목을 JSON 리스트로 추출하라.
    
    [학생 정보]
    - 전공: {major}
    - 학번(입학년도): {student_id} (졸업요건의 기준 key)
    - 대상: {grade} {semester}
    
    [진단 결과 (재수강 체크용)]
    {diagnosis_context}
    
    [핵심 규칙]
    1. **MSC(기초교양) 처리:** 수학, 과학, 전산 등 MSC 과목이라도, 해당 **{student_id}의 요람상 필수**가 아니거나 **선수과목**이 아니라면 `classification`을 "교양/기타"로 설정하고 `priority`를 "Normal"로 **강등**시켜라. (필수는 "High")
    2. **분반(Section):** 과목명 뒤의 숫자나 비고란을 확인하여 분반(예: 1, 2, H1)을 `section` 필드에 명시하라.
    3. **선수과목(Prerequisite):** 해당 과목을 듣기 위해 필요한 선이수 과목을 파악해 `prerequisite` 필드에 적어라.
    4. **전수 조사:** 모든 분반을 각각 별도의 항목으로 리스트업하라.

    [JSON 출력 포맷 예시]
    [
        {{
            "id": "unique_id_1",
            "name": "회로이론1",
            "section": "H1", 
            "professor": "김광운",
            "credits": 3,
            "time_slots": ["월3", "수4"],
            "classification": "전공필수",
            "priority": "High", 
            "reason": "전공필수 | 3학점",
            "prerequisite": "일반물리학"
        }}
    ]
    
    오직 JSON 리스트만 출력하라.
    [문서 데이터]
    {context}
    """
    
    def _execute():
        chain = PromptTemplate.from_template(prompt_template) | llm
        return chain.invoke({
            "major": major,
            "grade": grade,
            "semester": semester,
            "student_id": student_id,
            "diagnosis_context": diagnosis_text,
            "context": PRE_LEARNED_DATA
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

# [4-2] AI 검증 리포트
def validate_schedule_with_ai(schedule_list, major, student_id):
    llm = get_llm()
    if not llm: return "검증 서비스 불가"
    
    summary = "\n".join([f"- {c['name']} ({c.get('classification','Unknown')}, {c['credits']}학점)" for c in schedule_list])
    
    prompt = f"""
    당신은 꼼꼼한 학사 관리자입니다.
    아래 시간표가 **{major} {student_id}**의 졸업요건/커리큘럼과 비교해 문제가 없는지 3줄 요약 리포트를 작성하세요.
    
    [시간표]
    {summary}
    
    [형식]
    ⚠️ 경고: (필수 누락 등)
    ✅ 양호: (잘된 점)
    💡 조언: (추가 팁)
    
    [문서]
    {PRE_LEARNED_DATA}
    """
    try: return llm.invoke(prompt).content
    except: return "검증 중 오류 발생"

# [2-1] 학점 대시보드 (Bar Chart)
def render_credit_dashboard(schedule_list, student_id):
    total = sum([c.get('credits', 0) for c in schedule_list])
    # 예시 기준 (실제로는 AI가 요람에서 파싱해야 정확함, 여기선 Mockup)
    target = 18 
    
    st.markdown("##### 📊 이수 학점 현황")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.progress(min(total / 21, 1.0))
    with col2:
        st.caption(f"**{total}** / {target} 학점 (기준: {student_id})")

# -----------------------------------------------------------------------------
# [2] UI 구성
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🗂️ 활동 로그")
    # [로그인 UI]
    if st.session_state.user is None:
        with st.expander("🔐 로그인 / 회원가입", expanded=True):
            auth_mode = st.radio("모드 선택", ["로그인", "회원가입"], horizontal=True)
            email = st.text_input("이메일")
            password = st.text_input("비밀번호", type="password")
            
            if st.button(auth_mode):
                if not email or not password:
                    st.error("이메일과 비밀번호를 입력하세요.")
                else:
                    if not fb_manager.is_initialized:
                        st.error("Firebase 연결 실패 (Secrets를 확인하세요)")
                    else:
                        with st.spinner(f"{auth_mode} 중..."):
                            if auth_mode == "로그인":
                                user, err = fb_manager.login(email, password)
                            else:
                                user, err = fb_manager.signup(email, password)
                            
                            if user:
                                st.session_state.user = user
                                st.success(f"환영합니다! ({user['email']})")
                                st.rerun()
                            else:
                                st.error(f"오류: {err}")
    else:
        st.info(f"👤 **{st.session_state.user['email']}**님")
        if st.button("로그아웃"):
            st.session_state.clear()
            st.rerun()
    
    st.divider()
    # [복구됨] 시스템 관리자 모드 (원본 코드)
    st.subheader("⚙️ 시스템 관리자 모드")
    
    if st.button("📡 학교 서버 데이터 동기화 (Auto-Sync)"):
        status_text = st.empty()
        progress_bar = st.progress(0)
        status_text.text("🔄 광운대 KLAS 서버 접속 중...")
        time.sleep(1.0) 
        progress_bar.progress(30)
        status_text.text("📂 최신 학사 규정 및 시간표 스캔 중... (변경 감지!)")
        time.sleep(1.5)
        progress_bar.progress(70)
        status_text.text("⬇️ 신규 PDF 다운로드 및 벡터 DB 재구축 중...")
        st.cache_resource.clear()
        time.sleep(1.0)
        progress_bar.progress(100)
        st.success("✅ 동기화 완료! 최신 데이터(2026-01-12 14:30 기준)가 반영되었습니다.")
        time.sleep(2)
        st.rerun()          

    st.divider()
    st.caption("클릭하면 해당 화면으로 이동합니다.")
    log_container = st.container(height=300)
    with log_container:
        if not st.session_state.global_log:
            st.info("기록 없음")
        else:
            for i, log in enumerate(reversed(st.session_state.global_log)):
                label = f"[{log['time']}] {log['content'][:15]}..."
                if st.button(label, key=f"log_btn_{i}", use_container_width=True):
                    if log['menu']:
                        st.session_state.current_menu = log['menu']
                        st.session_state["menu_radio"] = log['menu'] 
                        st.rerun()
    st.divider()
    if PRE_LEARNED_DATA:
         st.success(f"✅ PDF 문서 학습 완료")
    else:
        st.error("⚠️ 데이터 폴더에 PDF 파일이 없습니다.")

# 메뉴 구성
menu = st.radio("기능 선택", ["🤖 AI 학사 지식인", "📅 스마트 시간표(Pro)", "📈 성적 및 진로 진단"], 
                horizontal=True, key="menu_radio")

if menu != st.session_state.current_menu:
    st.session_state.current_menu = menu
    st.rerun()

st.divider()

if st.session_state.current_menu == "🤖 AI 학사 지식인":
    st.subheader("🤖 무엇이든 물어보세요")
    if st.session_state.user and fb_manager.is_initialized:
        with st.expander("💾 대화 내용 관리"):
            col_s1, col_s2 = st.columns(2)
            if col_s1.button("현재 대화 저장"):
                doc_id = str(int(time.time()))
                data = {"history": [msg for msg in st.session_state.chat_history]}
                if fb_manager.save_data('chat_history', doc_id, data):
                    st.toast("대화 내용이 저장되었습니다.")
            
            saved_chats = fb_manager.load_collection('chat_history')
            if saved_chats:
                selected_chat = col_s2.selectbox("불러오기", saved_chats, format_func=lambda x: datetime.datetime.fromtimestamp(int(x['id'])).strftime('%Y-%m-%d %H:%M'), label_visibility="collapsed")
                if col_s2.button("로드"):
                    st.session_state.chat_history = selected_chat['history']
                    st.rerun()

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

elif st.session_state.current_menu == "📅 스마트 시간표(Pro)":
    st.subheader("📅 AI 스마트 시간표 빌더 Pro")
    
    # [상태 초기화]
    if "candidate_courses" not in st.session_state: st.session_state.candidate_courses = []
    if "my_schedule" not in st.session_state: st.session_state.my_schedule = []

    # [A] 설정 및 후보군 로딩
    with st.expander("🛠️ 수강신청 설정 (학과/학번/학년)", expanded=not bool(st.session_state.candidate_courses)):
        kw_departments = [
            "전자공학과", "전자통신공학과", "전자융합공학과", "전기공학과", "전자재료공학과", "반도체시스템공학부", "로봇학부",
            "컴퓨터정보공학부", "소프트웨어학부", "정보융합학부", "지능형로봇학과", "건축학과", "건축공학과", "화학공학과", "환경공학과",
            "수학과", "전자바이오물리학과", "화학과", "스포츠융합과학과", "정보콘텐츠학과", "국어국문학과", "영어산업학과", 
            "미디어커뮤니케이션학부", "산업심리학과", "동북아문화산업학부", "행정학과", "법학부", "국제학부", "자산관리학과",
            "경영학부", "국제통상학부", "자율전공학부(자연)", "자율전공학부(인문)"
        ]
        
        c1, c2, c3, c4 = st.columns(4)
        major = c1.selectbox("학과", kw_departments, key="tt_major")
        # [1-1] 학번 선택 추가
        student_id = c2.selectbox("학번 (입학년도)", ["26학번(예정)", "25학번", "24학번", "23학번", "22학번", "21학번 이전"], key="tt_std_id")
        grade = c3.selectbox("학년", ["1학년", "2학년", "3학년", "4학년"], key="tt_grade")
        semester = c4.selectbox("학기", ["1학기", "2학기"], key="tt_semester")
        
        use_diagnosis = st.checkbox("☑️ 성적 진단 결과 반영", value=True)
        
        if st.button("🚀 강의 목록 불러오기 (AI Scan)", type="primary", use_container_width=True):
            st.session_state.student_id_val = student_id # 저장
            diag_text = ""
            if use_diagnosis and st.session_state.graduation_analysis_result:
                 diag_text = st.session_state.graduation_analysis_result
            elif use_diagnosis and st.session_state.user and fb_manager.is_initialized:
                 saved_diags = fb_manager.load_collection('graduation_diagnosis')
                 if saved_diags: diag_text = saved_diags[0]['result']

            with st.spinner(f"{student_id} 기준 졸업요건 분석 및 강의 추출 중..."):
                candidates = get_course_candidates_json(major, grade, semester, student_id, diag_text)
                if candidates:
                    st.session_state.candidate_courses = candidates
                    st.session_state.my_schedule = [] 
                    st.session_state.cart_courses = [] # 장바구니 초기화
                    st.rerun()
                else:
                    st.error("강의 정보를 추출하지 못했습니다.")

    # [B] 인터랙티브 빌더 UI
    if st.session_state.candidate_courses:
        st.divider()
        # [2-1] 차트
        render_credit_dashboard(st.session_state.my_schedule, st.session_state.student_id_val)
        st.divider()

        col_list, col_cart, col_table = st.columns([1.1, 0.9, 1.5], gap="small")

        # 1. 강의 목록
        with col_list:
            st.subheader("📚 강의 목록")
            tab1, tab2, tab3 = st.tabs(["🔥 필수", "🏫 전공", "🧩 교양"])
            
            def draw_course_card(course):
                # 이미 선택된 것(시간표+카트) 제외
                if course['id'] in [c['id'] for c in st.session_state.my_schedule] + [c['id'] for c in st.session_state.cart_courses]:
                    return

                # 디자인 로직
                border_color = "#ffcccc" if course.get('priority') == 'High' else "#e3f2fd"
                
                with st.container(border=True):
                    # [2-2] 분반 정보
                    st.markdown(f"**{course['name']}** <span style='background:#eee; padding:1px 4px; border-radius:3px; font-size:10px;'>[{course.get('section','?')}반]</span>", unsafe_allow_html=True)
                    st.caption(f"{course['professor']} | {course['credits']}학점")
                    if course.get('prerequisite'):
                        st.markdown(f"<span style='color:#d32f2f; font-size:11px;'>⚠️ 선수: {course['prerequisite']}</span>", unsafe_allow_html=True)
                    
                    # [2-3] 장바구니 이동
                    if st.button("담기 🛒", key=f"add_c_{course['id']}", use_container_width=True):
                        st.session_state.cart_courses.append(course)
                        st.rerun()

            must = [c for c in st.session_state.candidate_courses if c.get('priority') == 'High']
            mj = [c for c in st.session_state.candidate_courses if c not in must and '전공' in c.get('classification', '')]
            ot = [c for c in st.session_state.candidate_courses if c not in must and c not in mj]

            with tab1: 
                for c in must: draw_course_card(c)
            with tab2: 
                for c in mj: draw_course_card(c)
            with tab3: 
                for c in ot: draw_course_card(c)

        # 2. 장바구니 (Cart)
        with col_cart:
            st.subheader("🛒 관심 과목 (Cart)")
            st.caption("확정 전 대기소입니다.")
            
            if not st.session_state.cart_courses:
                st.info("비어있음")
            
            for idx, item in enumerate(st.session_state.cart_courses):
                with st.container(border=True):
                    st.markdown(f"**{item['name']}**")
                    if item.get('prerequisite'):
                         st.markdown(f"<span style='color:red; font-size:10px;'>! 선수과목 확인: {item['prerequisite']}</span>", unsafe_allow_html=True)

                    c_btn1, c_btn2 = st.columns(2)
                    if c_btn1.button("확정 ➡️", key=f"confirm_{idx}", type="primary"):
                        # 충돌 체크
                        conflict, c_name = check_time_conflict(item, st.session_state.my_schedule)
                        if conflict:
                            st.toast(f"🚫 충돌: {c_name}", icon="⚠️")
                        else:
                            # [1-3] 선수과목 경고 (Toast)
                            if item.get('prerequisite'):
                                st.toast(f"🚧 '{item['name']}'의 선수과목({item['prerequisite']})을 이수했는지 확인하세요!", icon="🎓")
                            st.session_state.my_schedule.append(item)
                            st.session_state.cart_courses.pop(idx)
                            st.rerun()
                    
                    if c_btn2.button("삭제", key=f"del_cart_{idx}"):
                        st.session_state.cart_courses.pop(idx)
                        st.rerun()

        # 3. 시간표
        with col_table:
            st.subheader("🗓️ 내 시간표")
            if st.session_state.my_schedule:
                with st.expander("📝 확정 목록 편집"):
                    for idx, s_item in enumerate(st.session_state.my_schedule):
                         if st.button(f"❌ {s_item['name']} 취소", key=f"sc_del_{idx}"):
                             st.session_state.my_schedule.pop(idx)
                             st.rerun()
            
            # [3-2, 3-3] 렌더링
            html_view = render_interactive_timetable(st.session_state.my_schedule)
            st.markdown(html_view, unsafe_allow_html=True)
            
            st.divider()
            
            # [4-1] 폴더형 저장
            folder_name = st.text_input("📁 폴더/저장명 (예: 1안, 플랜B)", value="기본 시간표")
            if st.button("💾 저장 및 검증", use_container_width=True, type="primary"):
                if not st.session_state.my_schedule:
                    st.error("시간표가 비어있습니다.")
                else:
                    # [4-2] 검증 리포트
                    with st.spinner("AI 검증 중..."):
                        report = validate_schedule_with_ai(st.session_state.my_schedule, major, student_id)
                    
                    doc_data = {
                        "result": html_view,
                        "schedule_json": st.session_state.my_schedule,
                        "folder_name": folder_name,
                        "major": major,
                        "student_id": student_id,
                        "validation_report": report,
                        "created_at": datetime.datetime.now()
                    }
                    
                    if st.session_state.user and fb_manager.is_initialized:
                        doc_id = str(int(time.time()))
                        if fb_manager.save_data('timetables', doc_id, doc_data):
                            st.success("저장 완료!")
                            st.info(f"📋 **검증 리포트**\n\n{report}")
                        else:
                            st.error("저장 실패")
                    else:
                        st.warning("로그인 후 저장 가능합니다. (리포트만 출력됨)")
                        st.info(f"📋 **검증 리포트**\n\n{report}")

elif st.session_state.current_menu == "📈 성적 및 진로 진단":
    st.subheader("📈 성적 및 진로 정밀 진단")
    st.markdown("""
    **취득 학점 내역을 캡처해서 업로드하세요!** AI 취업 컨설턴트가 당신의 성적표를 냉철하게 분석하여 **졸업 요건**, **성적 상태**, **커리어 방향성**을 진단해 드립니다.
    """)

    if st.session_state.user and fb_manager.is_initialized:
        with st.expander("📂 저장된 진단 결과 불러오기"):
            saved_diags = fb_manager.load_collection('graduation_diagnosis')
            if saved_diags:
                selected_diag = st.selectbox("불러올 진단 선택", 
                                             saved_diags, 
                                             format_func=lambda x: datetime.datetime.fromtimestamp(int(x['id'])).strftime('%Y-%m-%d %H:%M'))
                if st.button("진단 결과 불러오기"):
                    st.session_state.graduation_analysis_result = selected_diag['result']
                    st.success("진단 결과를 불러왔습니다!")
                    st.rerun()

    uploaded_files = st.file_uploader("캡처 이미지 업로드 (여러 장 가능)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("진단 시작 🚀", type="primary"):
            with st.spinner("성적표를 독해하고 분석 중입니다... (냉철한 평가가 준비되고 있습니다)"):
                analysis_result = analyze_graduation_requirements(uploaded_files)
                st.session_state.graduation_analysis_result = analysis_result
                st.session_state.graduation_chat_history = []
                add_log("user", "[진단] 이미지 분석 요청", "📈 성적 및 진로 진단")
                st.rerun()

    if st.session_state.graduation_analysis_result:
        st.divider()
        result_text = st.session_state.graduation_analysis_result
        
        # 섹션 파싱 (기존 로직 유지)
        sec_grad = ""
        sec_grade = ""
        sec_career = ""
        try:
            if "[[SECTION:GRADUATION]]" in result_text:
                parts = result_text.split("[[[SECTION:GRADUATION]]")
                temp = parts[1] if len(parts) > 1 else result_text.split("[[SECTION:GRADUATION]]")[-1]
                if "[[SECTION:GRADES]]" in temp:
                    sec_grad, remaining = temp.split("[[SECTION:GRADES]]")
                    if "[[SECTION:CAREER]]" in remaining:
                        sec_grade, sec_career = remaining.split("[[SECTION:CAREER]]")
                    else:
                        sec_grade = remaining
                else:
                    sec_grad = temp
            else:
                sec_grad = result_text
        except:
            sec_grad = result_text

        tab1, tab2, tab3 = st.tabs(["🎓 졸업 요건 확인", "📊 성적 정밀 분석", "💼 AI 커리어 솔루션"])
        with tab1: st.markdown(sec_grad)
        with tab2: st.markdown(sec_grade if sec_grade else "성적 분석 결과가 없습니다.")
        with tab3: st.markdown(sec_career if sec_career else "커리어 솔루션 결과가 없습니다.")
        
        st.divider()
        if st.session_state.user and fb_manager.is_initialized:
            if st.button("☁️ 진단 결과 저장하기"):
                doc_data = {
                    "result": st.session_state.graduation_analysis_result,
                    "created_at": datetime.datetime.now()
                }
                doc_id = str(int(time.time()))
                if fb_manager.save_data('graduation_diagnosis', doc_id, doc_data):
                    st.toast("진단 결과가 저장되었습니다!", icon="✅")
        
        st.subheader("💬 컨설턴트와의 대화")
        for msg in st.session_state.graduation_chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if chat_input := st.chat_input("질문이나 추가 정보를 입력하세요"):
            st.session_state.graduation_chat_history.append({"role": "user", "content": chat_input})
            add_log("user", f"[진단상담] {chat_input}", "📈 성적 및 진로 진단")
            with st.chat_message("user"):
                st.write(chat_input)
            with st.chat_message("assistant"):
                with st.spinner("분석 중..."):
                    response = chat_with_graduation_ai(st.session_state.graduation_analysis_result, chat_input)
                    if "[수정]" in response:
                        new_result = response.replace("[수정]", "").strip()
                        st.session_state.graduation_analysis_result = new_result
                        st.session_state.graduation_chat_history.append({"role": "assistant", "content": "정보를 반영하여 업데이트했습니다."})
                        st.rerun()
                    else:
                        st.markdown(response)
                        st.session_state.graduation_chat_history.append({"role": "assistant", "content": response})

        if st.button("결과 초기화"):
            st.session_state.graduation_analysis_result = ""
            st.session_state.graduation_chat_history = []
            st.rerun()

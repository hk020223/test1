import streamlit as st
import pandas as pd
import os
import glob
import datetime
import time
import base64
import re
import json
import random
import plotly.graph_objects as go # 시각화(방사형 차트)
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Firebase 라이브러리
import firebase_admin
from firebase_admin import credentials, firestore

# -----------------------------------------------------------------------------
# [0] 설정 및 스타일
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="KW-Plan Pro: AI 수강 설계",
    page_icon="🦄",
    layout="wide"
)

# 파스텔톤 색상 팔레트 (과목별 자동 배색용)
PASTEL_COLORS = [
    "#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF", 
    "#E2F0CB", "#FFDAC1", "#FF9AA2", "#B5EAD7", "#C7CEEA",
    "#F8C8DC", "#FFD1DC", "#E0BBE4", "#957DAD", "#D291BC"
]

def get_color_for_course(course_name):
    """과목명에 기반하여 고정된 파스텔 색상을 반환 (해시 활용)"""
    idx = sum(ord(c) for c in course_name) % len(PASTEL_COLORS)
    return PASTEL_COLORS[idx]

def set_style():
    st.markdown("""
        <style>
        /* 기본 테마: 연한 버건디 틴트 */
        .stApp {
            background: linear-gradient(180deg, #FFFFFF 0%, #FFF0F5 100%) !important;
            background-attachment: fixed !important;
        }
        h1, h2, h3 {
            color: #8A1538 !important;
            font-family: 'Pretendard', sans-serif;
        }
        
        /* [UI] 우측 시간표 Sticky 처리 (화면 스크롤 시 따라옴) */
        div[data-testid="column"]:nth-of-type(2) {
            position: sticky;
            top: 2rem;
            height: fit-content;
            z-index: 99;
        }

        /* [UI] 탭 스타일 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 4px 4px 0 0;
            padding-top: 10px;
            padding-bottom: 10px;
        }

        /* [UI] 버튼 스타일 */
        div.stButton > button {
            border-radius: 8px;
            font-weight: bold;
        }

        /* [UI] 카드 스타일 (강의 리스트) */
        .course-card {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            margin-bottom: 10px;
            border: 1px solid #eee;
            transition: 0.2s;
        }
        .course-card:hover {
            border-color: #8A1538;
            transform: translateY(-2px);
        }
        
        /* 모바일 최적화 */
        @media only screen and (max-width: 600px) {
            div[data-testid="column"]:nth-of-type(2) {
                position: relative; /* 모바일에서는 Sticky 해제 */
            }
        }
        </style>
    """, unsafe_allow_html=True)

set_style()

# API Key 및 Firebase 설정
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = os.environ.get("GOOGLE_API_KEY", "")

if not api_key:
    st.error("🚨 **Google API Key가 설정되지 않았습니다.**")
    st.stop()

# -----------------------------------------------------------------------------
# [State] 세션 상태 관리 (v3.0 추가: Cart, StudentID)
# -----------------------------------------------------------------------------
if "global_log" not in st.session_state: st.session_state.global_log = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "current_menu" not in st.session_state: st.session_state.current_menu = "🤖 AI 학사 지식인"
if "menu_radio" not in st.session_state: st.session_state["menu_radio"] = "🤖 AI 학사 지식인"
if "user" not in st.session_state: st.session_state.user = None

# v3.0 Data States
if "candidate_courses" not in st.session_state: st.session_state.candidate_courses = []
if "cart_courses" not in st.session_state: st.session_state.cart_courses = [] # 장바구니
if "my_schedule" not in st.session_state: st.session_state.my_schedule = [] # 확정 시간표
if "student_id" not in st.session_state: st.session_state.student_id = "25학번" # 학번 기본값
if "graduation_analysis_result" not in st.session_state: st.session_state.graduation_analysis_result = ""
if "graduation_chat_history" not in st.session_state: st.session_state.graduation_chat_history = []

def add_log(role, content, menu_context=None):
    timestamp = datetime.datetime.now().strftime("%H:%M")
    st.session_state.global_log.append({
        "role": role, "content": content, "time": timestamp, "menu": menu_context
    })

def run_with_retry(func, *args, **kwargs):
    max_retries = 3
    for i in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                time.sleep(2 ** i)
                continue
            raise e

# -----------------------------------------------------------------------------
# [Backend] Firebase Manager
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
            except Exception: pass

    def login(self, email, password):
        if not self.is_initialized: return None, "DB 연결 실패"
        try:
            users_ref = self.db.collection('users')
            query = users_ref.where('email', '==', email).where('password', '==', password).stream()
            for doc in query:
                user_data = doc.to_dict(); user_data['localId'] = doc.id
                return user_data, None
            return None, "계정 정보 불일치"
        except Exception as e: return None, str(e)

    def signup(self, email, password):
        if not self.is_initialized: return None, "DB 연결 실패"
        try:
            users_ref = self.db.collection('users')
            if len(list(users_ref.where('email', '==', email).stream())) > 0:
                return None, "이미 존재하는 이메일"
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
            self.db.collection('users').document(uid).collection(collection).document(doc_id).set({
                **data, "updated_at": firestore.SERVER_TIMESTAMP
            })
            return True
        except: return False

    def load_collection(self, collection):
        if not self.is_initialized or not st.session_state.user: return []
        try:
            uid = st.session_state.user['localId']
            docs = self.db.collection('users').document(uid).collection(collection).order_by('updated_at', direction=firestore.Query.DESCENDING).stream()
            return [{"id": d.id, **d.to_dict()} for d in docs]
        except: return []

fb_manager = FirebaseManager()

# -----------------------------------------------------------------------------
# [AI] 데이터 로드 및 Gemini 엔진
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="학사 문서(요람) 분석 중...")
def load_knowledge_base():
    if not os.path.exists("data"): return ""
    all_content = ""
    for pdf in glob.glob("data/*.pdf"):
        try:
            loader = PyPDFLoader(pdf)
            for page in loader.load_and_split():
                all_content += page.page_content
        except: continue
    return all_content

PRE_LEARNED_DATA = load_knowledge_base()

def get_llm():
    if not api_key: return None
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-09-2025", temperature=0)

# [v3.0] AI 파서: 학번, MSC, 선수과목 로직 반영
def get_course_candidates_json(major, grade, semester, student_id, diagnosis_text=""):
    llm = get_llm()
    if not llm: return []
    
    prompt_template = """
    너는 [대학교 수강신청 데이터베이스 파서]이다. 
    제공된 문서를 분석하여 **{major} {student_id} {grade} {semester}** 학생이 수강 가능한 **모든 정규 개설 과목**을 JSON 리스트로 추출하라.
    
    [입력 정보]
    - 전공: {major} / 학번: {student_id} (졸업요건 Key)
    - 대상: {grade} {semester}
    - 진단결과: {diagnosis_context}
    
    [v3.0 엄격한 분류 규칙]
    1. **MSC 및 필수 과목 판별:** {student_id} 기준의 졸업 요람을 확인하여, 해당 학과/학번의 필수 과목인지 체크하라. 필수가 아니면 '전공선택' 또는 '교양'으로 분류.
    2. **선수과목 경고:** '회로이론2' 처럼 선수과목이 필요한 경우, reason 필드에 "[선수과목 주의]"라고 명시하라.
    3. **Priority:** 전공필수/재수강=High, 전공선택=Medium, 교양=Normal.
    4. **분반 정보:** 문서에 분반(A반, H1반 등)이 있다면 name 뒤에 붙이거나 reason에 적어라.

    [JSON 출력 포맷]
    [
        {{
            "id": "unique_id_1",
            "name": "자료구조",
            "professor": "이광운",
            "credits": 3,
            "time_slots": ["월3", "수4"],
            "classification": "전공필수",
            "priority": "High", 
            "reason": "전공필수 | 3학점 | [분반: A]",
            "category": "major" 
        }},
        {{
            "id": "unique_id_2",
            "name": "공학수학1",
            "professor": "김수학",
            "credits": 3,
            "time_slots": ["화1", "목1"],
            "classification": "MSC필수",
            "priority": "High",
            "reason": "MSC | 3학점",
            "category": "msc"
        }}
    ]
    **오직 JSON만 출력.**
    [문서 데이터]
    {context}
    """
    
    def _execute():
        chain = PromptTemplate.from_template(prompt_template) | llm
        return chain.invoke({
            "major": major, "grade": grade, "semester": semester, 
            "student_id": student_id, "diagnosis_context": diagnosis_text, 
            "context": PRE_LEARNED_DATA
        }).content

    try:
        response = run_with_retry(_execute)
        cleaned = response.replace("```json", "").replace("```", "").strip()
        if "[" in cleaned and "]" in cleaned:
            cleaned = cleaned[cleaned.find("["):cleaned.rfind("]")+1]
        return json.loads(cleaned)
    except Exception as e:
        print(f"Parsing Error: {e}")
        return []

# -----------------------------------------------------------------------------
# [Functions] v3.0 시각화 및 로직
# -----------------------------------------------------------------------------
# [v3.0] 방사형 차트 (이수 밸런스)
def draw_radar_chart(cart_list):
    # 분류별 학점 계산
    categories = {"전공(Major)": 0, "MSC/기초": 0, "교양(Gen-Ed)": 0}
    
    for c in cart_list:
        cls = c.get('classification', '')
        if '전공' in cls: categories["전공(Major)"] += c.get('credits', 3)
        elif 'MSC' in cls or '수학' in cls or '과학' in cls: categories["MSC/기초"] += c.get('credits', 3)
        else: categories["교양(Gen-Ed)"] += c.get('credits', 3)
    
    # Chart 생성
    fig = go.Figure(data=go.Scatterpolar(
        r=list(categories.values()),
        theta=list(categories.keys()),
        fill='toself',
        line_color='#8A1538'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 18])), # 최대 18학점 기준
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=30),
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# [v3.0] 시간표 렌더러 (파스텔톤 + 온라인 행 분리 + 이미지 다운로드 준비)
def render_timetable_v3(schedule_list):
    days = ["월", "화", "수", "목", "금"]
    table_grid = {i: {d: None for d in days} for i in range(1, 10)} # 1~9교시
    online_courses = []

    for course in schedule_list:
        slots = course.get('time_slots', [])
        # 슬롯 없는 경우 온라인 처리
        if not slots or slots == ["시간미정"] or not isinstance(slots, list):
            online_courses.append(course)
            continue
        
        # 슬롯 파싱 (예: "월3")
        for slot in slots:
            if len(slot) < 2: continue
            day = slot[0]
            try:
                period = int(slot[1:])
                if day in days and 1 <= period <= 9:
                    table_grid[period][day] = course
            except: pass

    # HTML 생성
    html = """
    <style>
        table.timetable { width: 100%; border-collapse: collapse; text-align: center; font-size: 11px; font-family: sans-serif; }
        .timetable th { background-color: #f8f9fa; border: 1px solid #ddd; padding: 5px; color: #555; }
        .timetable td { border: 1px solid #ddd; height: 50px; vertical-align: middle; padding: 2px; }
        .cell-content { border-radius: 4px; padding: 4px; font-weight: bold; color: #333; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }
    </style>
    <div id="capture_target" style="padding: 10px; background: white; border-radius: 8px;">
        <h3 style="text-align:center; color:#8A1538; margin-bottom:10px;">2026-1학기 시간표</h3>
        <table class="timetable">
            <thead>
                <tr>
                    <th width="10%">교시</th><th width="18%">월</th><th width="18%">화</th><th width="18%">수</th><th width="18%">목</th><th width="18%">금</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for i in range(1, 10):
        html += f"<tr><td style='background-color:#f8f9fa;'><b>{i}</b></td>"
        for day in days:
            course = table_grid[i][day]
            if course:
                color = get_color_for_course(course['name'])
                cell = f"<div class='cell-content' style='background-color:{color};'>{course['name']}<br><span style='font-size:9px; font-weight:normal;'>{course['professor']}</span></div>"
                html += f"<td>{cell}</td>"
            else:
                html += "<td></td>"
        html += "</tr>"

    # [v3.0] 온라인 강의 전용 Row
    if online_courses:
        online_html = ""
        for oc in online_courses:
            color = get_color_for_course(oc['name'])
            online_html += f"<span style='display:inline-block; background-color:{color}; padding:2px 6px; border-radius:4px; margin-right:5px; font-size:11px;'>💻 {oc['name']}</span>"
        html += f"<tr><td style='background-color:#f1f3f5;'><b>Online</b></td><td colspan='5' style='text-align:left; padding:8px;'>{online_html}</td></tr>"

    html += "</tbody></table></div>"
    return html

# 시간 충돌 체크
def check_conflict(new_course, current_schedule):
    new_slots = set(new_course.get('time_slots', []))
    for existing in current_schedule:
        existing_slots = set(existing.get('time_slots', []))
        if new_slots & existing_slots:
            return True, existing['name']
    return False, None

# -----------------------------------------------------------------------------
# [UI] 메인 어플리케이션
# -----------------------------------------------------------------------------
def main():
    # 사이드바 (로그인 & 메뉴)
    with st.sidebar:
        st.title("🦄 KW-Plan Pro")
        if not st.session_state.user:
            with st.expander("🔐 로그인 / 회원가입", expanded=True):
                mode = st.radio("모드", ["로그인", "회원가입"], horizontal=True, label_visibility="collapsed")
                email = st.text_input("이메일")
                pw = st.text_input("비밀번호", type="password")
                if st.button("실행"):
                    if not fb_manager.is_initialized: st.error("DB 설정 오류")
                    elif mode == "로그인":
                        u, e = fb_manager.login(email, pw)
                        if u: st.session_state.user = u; st.rerun()
                        else: st.error(e)
                    else:
                        u, e = fb_manager.signup(email, pw)
                        if u: st.session_state.user = u; st.rerun()
                        else: st.error(e)
        else:
            st.info(f"🎓 {st.session_state.user['email']}")
            if st.button("로그아웃"): st.session_state.clear(); st.rerun()
        
        st.divider()
        st.caption("Navigation")
        if st.button("📅 수강신청 마스터 (v3.0)", use_container_width=True):
             st.session_state.current_menu = "📅 수강신청 마스터 (v3.0)"
             st.rerun()
        if st.button("🤖 AI 학사 지식인", use_container_width=True):
             st.session_state.current_menu = "🤖 AI 학사 지식인"
             st.rerun()
        if st.button("📈 성적/졸업 진단", use_container_width=True):
             st.session_state.current_menu = "📈 성적/졸업 진단"
             st.rerun()

    # 상단 헤더
    st.markdown("<h1 style='text-align: center; color: #8A1538;'>🦄 KW-Course Master Pro <span style='font-size:16px; color:gray;'>v3.0</span></h1>", unsafe_allow_html=True)
    st.divider()

    # --------------------------------------------------------------------------
    # MENU 1: 📅 수강신청 마스터 (v3.0 Core)
    # --------------------------------------------------------------------------
    if st.session_state.current_menu == "📅 수강신청 마스터 (v3.0)":
        
        # [Step 1] 설정 패널 (학번 추가됨)
        with st.expander("🛠️ 1단계: 수강 환경 설정", expanded=not bool(st.session_state.candidate_courses)):
            c1, c2, c3, c4 = st.columns(4)
            major = c1.selectbox("학과", ["전자융합공학과", "컴퓨터정보공학부", "소프트웨어학부", "전기공학과", "로봇학부", "경영학부", "법학부"])
            student_id = c2.selectbox("학번 (입학년도)", ["26학번", "25학번", "24학번", "23학번", "22학번", "21학번 이전"])
            grade = c3.selectbox("학년", ["1학년", "2학년", "3학년", "4학년"])
            semester = c4.selectbox("학기", ["1학기", "2학기"])
            
            if st.button("🚀 AI 데이터 로드 (요람 분석)", type="primary", use_container_width=True):
                with st.spinner(f"📘 {major} {student_id} 기준 졸업 요건 및 개설 과목 분석 중..."):
                    diag = st.session_state.graduation_analysis_result
                    res = get_course_candidates_json(major, grade, semester, student_id, diag)
                    if res:
                        st.session_state.candidate_courses = res
                        st.session_state.cart_courses = []
                        st.session_state.my_schedule = []
                        st.session_state.student_id = student_id
                        st.success(f"✅ {len(res)}개 과목 로드 완료!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("데이터 로드 실패")

        if st.session_state.candidate_courses:
            # [Step 2] 메인 작업 공간 (좌: 검색/Cart, 우: Sticky Timetable)
            col_left, col_right = st.columns([1.2, 1])

            with col_left:
                # [Visual] 방사형 차트
                st.subheader("📊 이수 밸런스 체크")
                current_selection = st.session_state.cart_courses + st.session_state.my_schedule
                if current_selection:
                    fig = draw_radar_chart(current_selection)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("과목을 담으면 분석 차트가 표시됩니다.")

                st.divider()

                # [Cart System] 장바구니 vs 검색
                tab_search, tab_cart = st.tabs(["🔍 강의 검색", f"🛒 장바구니 ({len(st.session_state.cart_courses)})"])
                
                with tab_search:
                    # 필터링
                    filter_opt = st.radio("분류 필터", ["전체", "필수/MSC", "전공", "교양"], horizontal=True, label_visibility="collapsed")
                    
                    filtered = st.session_state.candidate_courses
                    if filter_opt == "필수/MSC": filtered = [c for c in filtered if c.get('priority') == 'High']
                    elif filter_opt == "전공": filtered = [c for c in filtered if '전공' in c.get('classification','')]
                    elif filter_opt == "교양": filtered = [c for c in filtered if '교양' in c.get('classification','')]

                    for course in filtered:
                        # 이미 담은건 패스
                        if course in st.session_state.cart_courses or course in st.session_state.my_schedule: continue
                        
                        with st.container():
                            st.markdown(f"""
                            <div class="course-card">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <div>
                                        <div style="font-size:16px; font-weight:bold; color:#333;">{course['name']} <span style="font-size:12px; color:#888;">{course.get('professor','')}</span></div>
                                        <div style="font-size:12px; color:#666;">{course.get('classification')} | {course.get('credits')}학점</div>
                                        <div style="font-size:11px; color:#8A1538;">{course.get('reason','')}</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            if st.button("담기 🔽", key=f"add_{course['id']}", use_container_width=True):
                                st.session_state.cart_courses.append(course)
                                st.rerun()

                with tab_cart:
                    if not st.session_state.cart_courses:
                        st.info("장바구니가 비어있습니다.")
                    else:
                        st.success("시간표로 옮길 과목을 선택(확정)하세요.")
                        for idx, c in enumerate(st.session_state.cart_courses):
                            cc1, cc2 = st.columns([3, 1])
                            cc1.markdown(f"**{c['name']}** ({c['time_slots']})")
                            if cc2.button("확정 ▶️", key=f"confirm_{idx}"):
                                conflict, conflict_name = check_conflict(c, st.session_state.my_schedule)
                                if conflict:
                                    st.error(f"충돌: {conflict_name}")
                                else:
                                    st.session_state.my_schedule.append(c)
                                    st.session_state.cart_courses.pop(idx)
                                    st.rerun()
                            
                            # 삭제 버튼
                            if st.button("삭제 🗑️", key=f"del_cart_{idx}"):
                                st.session_state.cart_courses.pop(idx)
                                st.rerun()

            with col_right:
                st.subheader(f"🗓️ 확정 시간표 ({sum(c['credits'] for c in st.session_state.my_schedule)}학점)")
                
                # [v3.0] HTML 렌더링
                html_code = render_timetable_v3(st.session_state.my_schedule)
                st.components.v1.html(html_code, height=500, scrolling=True)

                # [Control]
                c_btn1, c_btn2 = st.columns(2)
                if c_btn1.button("🔄 초기화", use_container_width=True):
                    st.session_state.my_schedule = []
                    st.rerun()
                
                # [v3.0] 저장 및 이미지 다운로드
                if c_btn2.button("💾 클라우드 저장", use_container_width=True):
                    if not st.session_state.user: st.warning("로그인 필요")
                    else:
                        doc_id = str(int(time.time()))
                        # 검증 리포트 생성
                        validation_msg = "✅ 검증 완료"
                        if sum(c['credits'] for c in st.session_state.my_schedule) < 15:
                            validation_msg = "⚠️ 15학점 미만입니다."
                        
                        data = {
                            "name": f"{major} {grade} (Plan A)",
                            "folder": "2026-1학기", # v3.0 폴더 기능
                            "result_html": html_code,
                            "validation": validation_msg,
                            "credits": sum(c['credits'] for c in st.session_state.my_schedule)
                        }
                        if fb_manager.save_data("timetables", doc_id, data):
                            st.toast("저장 및 검증 완료!")
                
                # [v3.0] 이미지 다운로드 (Data URI 활용)
                b64 = base64.b64encode(html_code.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="timetable.html" style="text-decoration:none; display:inline-block; width:100%; background-color:#4CAF50; color:white; padding:8px; text-align:center; border-radius:8px; font-weight:bold;">🖼️ 이미지/HTML 다운로드</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.caption("다운로드된 파일을 열어 '인쇄->PDF 저장' 하세요.")

    # --------------------------------------------------------------------------
    # MENU 2: 🤖 AI 학사 지식인 (기존 유지)
    # --------------------------------------------------------------------------
    elif st.session_state.current_menu == "🤖 AI 학사 지식인":
        st.subheader("🤖 무엇이든 물어보세요 (학사규정/졸업요건)")
        
        # 채팅 UI
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
            
        if user_input := st.chat_input("질문 입력 (예: 25학번 전자융합공학과 졸업요건 알려줘)"):
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            add_log("user", f"[지식인] {user_input}", "AI 지식인")
            with st.chat_message("user"): st.markdown(user_input)
            
            with st.chat_message("assistant"):
                with st.spinner("규정집 찾는 중..."):
                    llm = get_llm()
                    prompt = PromptTemplate.from_template("문서: {context}\n질문: {question}\n근거를 인용해서 답변해.")
                    chain = prompt | llm
                    response = chain.invoke({"context": PRE_LEARNED_DATA, "question": user_input}).content
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # --------------------------------------------------------------------------
    # MENU 3: 📈 성적/졸업 진단 (기존 유지)
    # --------------------------------------------------------------------------
    elif st.session_state.current_menu == "📈 성적/졸업 진단":
        st.subheader("📈 성적표 진단 및 커리어 코칭")
        uploaded = st.file_uploader("성적표 이미지 업로드", accept_multiple_files=True)
        if uploaded and st.button("분석 시작"):
            st.info("이미지 분석 기능은 Vision API 토큰이 필요합니다. (데모 모드)")
            # 실제 구현 시 get_pro_llm() 사용

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import os
import glob
import datetime
import time
import base64
import json
import requests
import uuid
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# LangChain & AI
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

# -----------------------------------------------------------------------------
# [0] 설정 및 상수 정의
# -----------------------------------------------------------------------------
st.set_page_config(page_title="KW-강의마스터 Pro", page_icon="🎓", layout="wide")

# 광운대학교 전체 학과 리스트 (상수)
ALL_DEPARTMENTS = [
    "전자융합공학과", "전자공학과", "전자통신공학과", "전기공학과", "전자재료공학과", "로봇학부",
    "소프트웨어학부", "컴퓨터정보공학부", "정보융합학부",
    "건축학과", "건축공학과", "화학공학과", "환경공학과",
    "수학과", "전자바이오물리학과", "화학과", "스포츠융합과학과",
    "국어국문학과", "영어산업학과", "미디어커뮤니케이션학부", "산업심리학과", "동북아문화산업학부",
    "행정학과", "법학부", "국제학부",
    "경영학부", "국제통상학부"
]
ALL_DEPARTMENTS.sort()

# CSS 스타일
st.markdown("""
    <style>
        footer { visibility: hidden; }
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
        .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #ff4b4b; }
        /* 모바일 최적화 */
        @media only screen and (max-width: 600px) {
            .main .block-container { padding-top: 2rem !important; }
            div[data-testid="stMarkdownContainer"] table { font-size: 10px !important; }
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
if "user" not in st.session_state: st.session_state.user = None
if "global_log" not in st.session_state: st.session_state.global_log = []
if "shared_context" not in st.session_state: st.session_state.shared_context = "" # 탭 간 맥락 공유용
if "grade_json_data" not in st.session_state: st.session_state.grade_json_data = None
if "graduation_json_data" not in st.session_state: st.session_state.graduation_json_data = None # 졸업요건 데이터 (시각화용)
if "timetable_result" not in st.session_state: st.session_state.timetable_result = ""
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "timetable_chat_history" not in st.session_state: st.session_state.timetable_chat_history = []
if "graduation_chat_history" not in st.session_state: st.session_state.graduation_chat_history = []
if "bookmarks" not in st.session_state: st.session_state.bookmarks = [] # Q&A 보관함

# -----------------------------------------------------------------------------
# [Firebase Manager] 데이터 저장/로드 및 인증
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

    def auth_user(self, email, password, mode="login"):
        if "FIREBASE_WEB_API_KEY" not in st.secrets: return None, "API Key 설정 필요"
        api_key_fb = st.secrets["FIREBASE_WEB_API_KEY"].strip()
        endpoint = "signInWithPassword" if mode == "login" else "signUp"
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:{endpoint}?key={api_key_fb}"
        payload = {"email": email, "password": password, "returnSecureToken": True}
        try:
            res = requests.post(url, json=payload)
            data = res.json()
            if "error" in data: return None, data["error"]["message"]
            return data, None
        except Exception as e: return None, str(e)

    # 데이터 저장 (성적, 졸업요건 등)
    def save_user_data(self, collection, doc_id, data):
        if not self.is_initialized or not st.session_state.user: return False
        try:
            user_id = st.session_state.user['localId']
            self.db.collection('users').document(user_id).collection(collection).document(doc_id).set(data)
            return True
        except: return False
    
    # 데이터 로드 (단일 문서)
    def load_user_data(self, collection, doc_id):
        if not self.is_initialized or not st.session_state.user: return None
        try:
            user_id = st.session_state.user['localId']
            doc = self.db.collection('users').document(user_id).collection(collection).document(doc_id).get()
            return doc.to_dict() if doc.exists else None
        except: return None

    # 보관함(Bookmarks) 추가
    def add_bookmark(self, question, answer, tag):
        if not self.is_initialized or not st.session_state.user: return False
        try:
            user_id = st.session_state.user['localId']
            data = {
                "question": question,
                "answer": answer,
                "tag": tag,
                "created_at": firestore.SERVER_TIMESTAMP
            }
            self.db.collection('users').document(user_id).collection('bookmarks').add(data)
            return True
        except: return False

    # 보관함 로드
    def load_bookmarks(self):
        if not self.is_initialized or not st.session_state.user: return []
        try:
            user_id = st.session_state.user['localId']
            docs = self.db.collection('users').document(user_id).collection('bookmarks').order_by('created_at', direction=firestore.Query.DESCENDING).stream()
            return [{"id": doc.id, **doc.to_dict()} for doc in docs]
        except: return []

fb_manager = FirebaseManager()

# -----------------------------------------------------------------------------
# [AI 엔진]
# -----------------------------------------------------------------------------
def get_llm(): 
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)

def get_pro_llm(): 
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=api_key)

@st.cache_resource
def load_knowledge_base():
    if not os.path.exists("data"): return ""
    pdf_files = glob.glob("data/*.pdf")
    content = ""
    for f in pdf_files:
        try: content += f"\n\n--- [{os.path.basename(f)}] ---\n" + "".join([p.page_content for p in PyPDFLoader(f).load()])
        except: pass
    return content

PRE_LEARNED_DATA = load_knowledge_base()

def clean_json_output(text):
    text = text.strip()
    if text.startswith("```json"): text = text[7:]
    elif text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    return text.strip()

# -----------------------------------------------------------------------------
# [핵심 기능 1] 성적표 분석 (JSON 추출)
# -----------------------------------------------------------------------------
def analyze_grades_structure(uploaded_images):
    llm = get_pro_llm()
    image_messages = []
    for img_file in uploaded_images:
        img_file.seek(0)
        b64 = base64.b64encode(img_file.read()).decode("utf-8")
        image_messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    
    prompt = """
    성적표 이미지를 분석하여 **반드시 유효한 JSON 형식**으로만 출력하세요. 마크다운 금지.
    
    {
        "student_info": {"admission_year": "2024", "major": "전자공학과"},
        "courses": [
            {"year": "2024", "semester": "1", "type": "전필", "name": "회로이론1", "grade": "A+", "score": 4.5},
            ...
        ],
        "strength_keywords": ["회로설계", "임베디드"],
        "weakness_analysis": "전공 기초는 튼튼하나 SW 관련 프로젝트 경험이 부족함."
    }
    """
    msg = HumanMessage(content=[{"type": "text", "text": prompt}] + image_messages)
    try:
        res = llm.invoke([msg]).content
        return json.loads(clean_json_output(res))
    except: return None

# -----------------------------------------------------------------------------
# [핵심 기능 2] 졸업 요건 분석 (JSON + 리포트)
# -----------------------------------------------------------------------------
def analyze_graduation_json(uploaded_images):
    llm = get_pro_llm()
    image_messages = []
    for img_file in uploaded_images:
        img_file.seek(0)
        b64 = base64.b64encode(img_file.read()).decode("utf-8")
        image_messages.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    
    prompt = """
    졸업 요건을 진단하여 **JSON 데이터**와 **분석 리포트(Text)** 두 가지를 모두 포함한 JSON으로 출력하세요.
    [학사 문서]를 참고하여 정확히 계산하세요.
    
    출력 형식:
    {
        "chart_data": {
            "total": {"earned": 100, "required": 130},
            "major_req": {"earned": 15, "required": 21},
            "major_sel": {"earned": 30, "required": 54},
            "liberal": {"earned": 20, "required": 30}
        },
        "report_text": "### 🎓 졸업 요건 진단 결과\n\n..."
    }
    """
    msg = HumanMessage(content=[{"type": "text", "text": prompt}] + image_messages + [{"type": "text", "text": f"\n[학사 문서]\n{PRE_LEARNED_DATA}"}])
    try:
        res = llm.invoke([msg]).content
        return json.loads(clean_json_output(res))
    except: return None

# -----------------------------------------------------------------------------
# [핵심 기능 3] AI 도구 (시간표, 커리어)
# -----------------------------------------------------------------------------
def consult_career_path(job_role, grade_json, context):
    llm = get_llm()
    search = DuckDuckGoSearchRun()
    try: search_res = search.invoke(f"{job_role} 신입 채용 기술 스택 자격요건")
    except: search_res = "검색 불가"
    
    template = """
    당신은 냉철한 채용 담당자입니다.
    [지원자 스펙] {student_data}
    [시장 요구사항] {search_result}
    [학교 커리큘럼] {context}
    
    지원자의 부족한 점(Skill Gap)을 지적하고, 학교 강의 중 무엇을 들어야 할지 구체적으로 추천하세요.
    """
    prompt = PromptTemplate(template=template, input_variables=["student_data", "search_result", "context"])
    return (prompt | llm).invoke({"student_data": json.dumps(grade_json), "search_result": search_res, "context": context}).content

def generate_timetable_ai(major, grade, semester, target, blocked, req, shared_ctx):
    llm = get_llm()
    template = """
    수강신청 전문가로서 시간표를 작성하세요. 출력은 HTML Table 형식입니다.
    [학생 정보] {major} {grade} {semester}, 목표 {target}학점
    [공강 시간] {blocked}
    [추가 요구] {req}
    
    ★★★ [이전 상담 맥락 반영 필수] ★★★
    "{shared_ctx}"
    위 맥락에서 언급된 부족한 역량을 채울 수 있는 과목을 우선 배치하세요.
    
    [학습 문서] {context}
    """
    prompt = PromptTemplate(template=template, input_variables=["major", "grade", "semester", "target", "blocked", "req", "shared_ctx", "context"])
    res = (prompt | llm).invoke({
        "major": major, "grade": grade, "semester": semester, "target": target, 
        "blocked": blocked, "req": req, "shared_ctx": shared_ctx, "context": PRE_LEARNED_DATA
    }).content
    return res.replace("```html", "").replace("```", "").strip()

# -----------------------------------------------------------------------------
# [UI] 메인 앱
# -----------------------------------------------------------------------------
# 1. 사이드바 (로그인, 보관함)
with st.sidebar:
    st.title("🗂️ 내비게이션")
    
    # 로그인 처리
    if st.session_state.user is None:
        with st.expander("🔐 로그인 / 회원가입", expanded=True):
            mode = st.radio("모드", ["로그인", "회원가입"], horizontal=True, label_visibility="collapsed")
            email = st.text_input("이메일")
            pw = st.text_input("비밀번호", type="password")
            if st.button("실행"):
                u, e = fb_manager.auth_user(email, pw, "login" if mode == "로그인" else "signup")
                if u:
                    st.session_state.user = u
                    # [데이터 복원]
                    grade_data = fb_manager.load_user_data('grade_data', 'latest')
                    if grade_data: st.session_state.grade_json_data = grade_data
                    grad_data = fb_manager.load_user_data('graduation_data', 'latest')
                    if grad_data: st.session_state.graduation_json_data = grad_data
                    
                    st.success("로그인 성공! 이전 데이터를 불러왔습니다.")
                    time.sleep(1)
                    st.rerun()
                else: st.error(e)
    else:
        st.info(f"👋 {st.session_state.user['email']}님")
        if st.button("로그아웃"):
            st.session_state.user = None
            st.session_state.grade_json_data = None
            st.session_state.graduation_json_data = None
            st.rerun()

    # 보관함 (Bookmarks)
    if st.session_state.user:
        st.divider()
        st.subheader("📂 Q&A 보관함")
        bookmarks = fb_manager.load_bookmarks()
        if not bookmarks: st.caption("저장된 내용이 없습니다.")
        for bm in bookmarks:
            with st.expander(f"📌 {bm['question'][:15]}..."):
                st.write(f"**Q:** {bm['question']}")
                st.write(f"**A:** {bm['answer']}")
                st.caption(f"Tag: {bm['tag']}")

# 2. 메인 탭 구성
st.title("🎓 KW-강의마스터 Pro")

tabs = st.tabs(["📈 성적 및 진로 진단", "📅 스마트 시간표", "🤖 AI 학사 지식인"])

# -----------------------------------------------------------------------------
# TAB 1: 성적 및 진로 진단 (도넛 차트 + 맥락 공유)
# -----------------------------------------------------------------------------
with tabs[0]:
    sub_tabs = st.tabs(["📊 성적 분석", "🎓 졸업 요건 확인", "🚀 AI 커리어 솔루션"])
    
    # 1-1. 성적 분석
    with sub_tabs[0]:
        st.markdown("##### 📄 성적표 업로드 (데이터는 자동 저장됩니다)")
        uploaded_grades = st.file_uploader("성적표 이미지", accept_multiple_files=True, key="grade_upl")
        
        if uploaded_grades and st.button("분석 시작"):
            with st.spinner("데이터 추출 중..."):
                data = analyze_grades_structure(uploaded_grades)
                if data:
                    st.session_state.grade_json_data = data
                    # [맥락 저장]
                    if "weakness_analysis" in data:
                        st.session_state.shared_context = data["weakness_analysis"]
                    # [DB 저장]
                    fb_manager.save_user_data('grade_data', 'latest', data)
                    st.rerun()

        if st.session_state.grade_json_data:
            d = st.session_state.grade_json_data
            st.success(f"학번: {d.get('student_info',{}).get('admission_year')} | 전공: {d.get('student_info',{}).get('major')}")
            
            # 맥락 공유 표시
            if st.session_state.shared_context:
                st.info(f"💡 **AI 진단(맥락):** {st.session_state.shared_context}")

            # 강점 키워드
            st.write("🔥 **나의 강점:** " + " ".join([f"`{k}`" for k in d.get("strength_keywords", [])]))
            
            # 성적 그래프
            df = pd.DataFrame(d.get("courses", []))
            if not df.empty:
                df['score'] = pd.to_numeric(df['score'], errors='coerce')
                st.line_chart(df.groupby('year')['score'].mean())
                with st.expander("데이터 원본"): st.json(d)

    # 1-2. 졸업 요건 (도넛 차트 시각화)
    with sub_tabs[1]:
        st.markdown("##### 🎓 졸업 요건 달성률 (시각화)")
        grad_files = st.file_uploader("졸업 요건용 성적표", accept_multiple_files=True, key="grad_upl")
        
        if grad_files and st.button("졸업 요건 진단"):
            with st.spinner("분석 중..."):
                res = analyze_graduation_json(grad_files)
                if res:
                    st.session_state.graduation_json_data = res
                    fb_manager.save_user_data('graduation_data', 'latest', res)
                    st.rerun()
        
        if st.session_state.graduation_json_data:
            data = st.session_state.graduation_json_data.get("chart_data", {})
            report = st.session_state.graduation_json_data.get("report_text", "")
            
            # 도넛 차트 그리기
            if data:
                fig = make_subplots(rows=1, cols=4, specs=[[{'type':'domain'}]*4], 
                                    subplot_titles=['총 학점', '전공 필수', '전공 선택', '교양'])
                
                labels = ["이수", "미이수"]
                colors = ['#4CAF50', '#E0E0E0']
                
                keys = ['total', 'major_req', 'major_sel', 'liberal']
                for i, key in enumerate(keys):
                    curr = data.get(key, {}).get('earned', 0)
                    req = data.get(key, {}).get('required', 100)
                    rem = max(0, req - curr)
                    
                    fig.add_trace(go.Pie(labels=labels, values=[curr, rem], hole=.6, 
                                         marker_colors=colors, textinfo='none'), 1, i+1)
                    
                    # 중앙 텍스트 (달성률)
                    percent = int((curr / req) * 100) if req > 0 else 0
                    fig.add_annotation(text=f"<b>{percent}%</b>", x=[0.11, 0.37, 0.63, 0.89][i], y=0.5, 
                                       showarrow=False, font_size=20)

                fig.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(report)

    # 1-3. 커리어 솔루션
    with sub_tabs[2]:
        st.markdown("##### 🚀 AI 채용 담당자 컨설팅")
        job = st.text_input("희망 직무")
        if st.button("분석"):
            if not st.session_state.grade_json_data: st.error("성적 분석 먼저 진행하세요.")
            else:
                with st.spinner("검색 및 분석 중..."):
                    res = consult_career_path(job, st.session_state.grade_json_data, PRE_LEARNED_DATA)
                    st.markdown(res)
                    # 여기서 나온 조언도 맥락에 추가 가능
                    st.session_state.shared_context += f"\n(진로 조언: {job} 관련 역량 보강 필요)"

# -----------------------------------------------------------------------------
# TAB 2: 스마트 시간표 (전체 학과 + 맥락 반영)
# -----------------------------------------------------------------------------
with tabs[1]:
    st.markdown("### 📅 맥락 기반 AI 시간표")
    
    if st.session_state.shared_context:
        st.info(f"💡 **반영된 맥락:** {st.session_state.shared_context}")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        # [수정 5] 전체 학과 리스트 적용
        major = st.selectbox("학과 선택", ALL_DEPARTMENTS)
        grade = st.selectbox("학년", ["1학년", "2학년", "3학년", "4학년"])
        semester = st.selectbox("학기", ["1학기", "2학기"])
        target = st.number_input("목표 학점", 9, 24, 18)
        req = st.text_area("추가 요구사항")
        
    with col2:
        st.caption("공강 시간 선택 (체크 해제 시 공강)")
        times = ["1교시", "2교시", "3교시", "4교시", "5교시", "6교시", "7교시", "8교시", "9교시"]
        if "sched_df" not in st.session_state:
            st.session_state.sched_df = pd.DataFrame(True, index=times, columns=["월", "화", "수", "목", "금"])
        edited_df = st.data_editor(st.session_state.sched_df, height=300, use_container_width=True)

    if st.button("시간표 생성", type="primary"):
        blocked = [f"{d} {t}" for d in edited_df.columns for t in times if not edited_df.loc[t, d]]
        with st.spinner("AI가 시간표 작성 중..."):
            res = generate_timetable_ai(major, grade, semester, target, ", ".join(blocked), req, st.session_state.shared_context)
            st.session_state.timetable_result = res
            st.rerun()

    if st.session_state.timetable_result:
        st.markdown(st.session_state.timetable_result, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 3: AI 학사 지식인 (보관함 기능)
# -----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("🤖 무엇이든 물어보세요")
    
    # 채팅 히스토리 표시
    for i, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # [수정 4] 질문-답변 세트 저장 버튼
            # assistant 메시지이고, 바로 앞이 user 메시지일 때 저장 버튼 표시
            if msg["role"] == "assistant" and i > 0 and st.session_state.chat_history[i-1]["role"] == "user":
                if st.button("💾 보관함 저장", key=f"save_{i}"):
                    q = st.session_state.chat_history[i-1]["content"]
                    a = msg["content"]
                    if fb_manager.add_bookmark(q, a, "지식인"):
                        st.toast("보관함에 저장되었습니다!", icon="✅")
                    else:
                        st.toast("로그인이 필요합니다.", icon="⚠️")

    if user_input := st.chat_input("질문 입력"):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("생성 중..."):
                # 맥락이 있다면 프롬프트에 살짝 추가 가능
                q_with_ctx = user_input
                if st.session_state.shared_context:
                    q_with_ctx = f"[사용자 상황: {st.session_state.shared_context}] \n질문: {user_input}"
                
                chain = PromptTemplate.from_template("문서: {ctx}\n질문: {q}") | get_llm()
                response = chain.invoke({"ctx": PRE_LEARNED_DATA, "q": q_with_ctx}).content
                st.markdown(response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()


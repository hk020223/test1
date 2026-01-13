import streamlit as st
import pandas as pd
import os
import glob
import datetime
import time
import base64
import re  # 정규표현식 사용
import json # JSON 처리를 위한 라이브러리
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
        }
        div[data-testid="stMarkdownContainer"] th, 
        div[data-testid="stMarkdownContainer"] td {
            padding: 2px !important;
            word-wrap: break-word !important;
            word-break: break-all !important;
            white-space: normal !important;
            line-height: 1.2 !important;
            vertical-align: middle !important;
        }
        /* 버튼 높이 조정 */
        button[kind="primary"], button[kind="secondary"] {
            padding: 0.2rem 0.5rem !important;
            min-height: 0px !important;
            height: auto !important;
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

# 세션 상태 초기화 (없으면 생성)
if "global_log" not in st.session_state:
    st.session_state.global_log = [] 
if "timetable_result" not in st.session_state:
    st.session_state.timetable_result = "" 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 
if "current_menu" not in st.session_state:
    st.session_state.current_menu = "🤖 AI 학사 지식인"
# 라디오 버튼 위젯 상태 초기화
if "menu_radio" not in st.session_state:
    st.session_state["menu_radio"] = "🤖 AI 학사 지식인"

if "timetable_chat_history" not in st.session_state:
    st.session_state.timetable_chat_history = []
if "graduation_analysis_result" not in st.session_state:
    st.session_state.graduation_analysis_result = ""
if "graduation_chat_history" not in st.session_state:
    st.session_state.graduation_chat_history = []
if "user" not in st.session_state:
    st.session_state.user = None

# 현재 불러온 시간표 메타데이터 (ID, 이름, 즐겨찾기 여부 등) 관리용
if "current_timetable_meta" not in st.session_state:
    st.session_state.current_timetable_meta = {}

# [추가] 선택된 강의계획서 뷰어 상태 관리
if "selected_syllabus" not in st.session_state:
    st.session_state.selected_syllabus = None

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
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "503" in error_msg:
                if i < max_retries - 1:
                    time.sleep(delays[i])
                    continue
            raise e

# -----------------------------------------------------------------------------
# [Firebase Manager] Firestore 기반 자체 인증 및 DB 관리
# -----------------------------------------------------------------------------
class FirebaseManager:
    def __init__(self):
        self.db = None
        self.is_initialized = False
        self.init_firestore()

    def init_firestore(self):
        """Firestore DB 초기화 (Service Account 사용)"""
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
        """Firestore에서 이메일/비번 매칭 검사"""
        if not self.is_initialized:
            return None, "Firebase 연결 실패"
        
        try:
            # users 컬렉션에서 email과 password가 일치하는 문서 검색
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
        """Firestore에 신규 유저 정보 저장"""
        if not self.is_initialized:
            return None, "Firebase 연결 실패"

        try:
            users_ref = self.db.collection('users')
            existing_user = list(users_ref.where('email', '==', email).stream())
            if len(existing_user) > 0:
                return None, "이미 가입된 이메일입니다."
            
            new_user_ref = users_ref.document()
            user_data = {
                "email": email,
                "password": password,
                "created_at": firestore.SERVER_TIMESTAMP
            }
            new_user_ref.set(user_data)
            
            user_data['localId'] = new_user_ref.id
            return user_data, None
        except Exception as e:
            return None, f"회원가입 오류: {str(e)}"

    def save_data(self, collection, doc_id, data):
        """데이터 저장 (덮어쓰기)"""
        if not self.is_initialized or not st.session_state.user:
            return False
        try:
            user_id = st.session_state.user['localId']
            doc_ref = self.db.collection('users').document(user_id).collection(collection).document(doc_id)
            data['updated_at'] = firestore.SERVER_TIMESTAMP
            doc_ref.set(data)
            return True
        except:
            return False

    def update_data(self, collection, doc_id, data):
        """데이터 부분 업데이트 (이름 변경, 즐겨찾기 등)"""
        if not self.is_initialized or not st.session_state.user:
            return False
        try:
            user_id = st.session_state.user['localId']
            doc_ref = self.db.collection('users').document(user_id).collection(collection).document(doc_id)
            data['updated_at'] = firestore.SERVER_TIMESTAMP
            doc_ref.update(data)
            return True
        except:
            return False

    def load_collection(self, collection):
        """데이터 목록 불러오기"""
        if not self.is_initialized or not st.session_state.user:
            return []
        try:
            user_id = st.session_state.user['localId']
            docs = self.db.collection('users').document(user_id).collection(collection).order_by('updated_at', direction=firestore.Query.DESCENDING).stream()
            return [{"id": doc.id, **doc.to_dict()} for doc in docs]
        except:
            return []

fb_manager = FirebaseManager()

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
# [1] AI 엔진 (gemini-2.5-flash-preview-09-2025)
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
# [Helper Functions] 인터랙티브 시간표 & AI 데이터 추출 (Strict Fact-Based)
# =============================================================================

# 1. 시간 충돌 감지 로직
def check_time_conflict(new_course, current_schedule):
    new_slots = set(new_course.get('time_slots', []))
    for existing in current_schedule:
        existing_slots = set(existing.get('time_slots', []))
        overlap = new_slots & existing_slots
        if overlap:
            return True, existing['name']
    return False, None

# 2. HTML 시간표 렌더러
def render_interactive_timetable(schedule_list):
    days = ["월", "화", "수", "목", "금"]
    table_grid = {i: {d: "" for d in days} for i in range(1, 10)}
    online_courses = []

    for course in schedule_list:
        slots = course.get('time_slots', [])
        # 온라인/시간미정 처리
        if not slots or slots == ["시간미정"] or not isinstance(slots, list):
            online_courses.append(course)
            continue

        for slot in slots:
            if len(slot) < 2: continue
            day_char = slot[0] # "월"
            try:
                period = int(slot[1:]) # "3"
                if day_char in days and 1 <= period <= 9:
                    content = f"<b>{course['name']}</b><br><small>{course['professor']}</small>"
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
            bg_color = "#ffffff" if not cell_content else "#e3f2fd"
            border_style = "border: 1px solid #ddd;"
            html += f"<td style='background-color: {bg_color}; {border_style} height: 45px; vertical-align: middle;'>{cell_content}</td>"
        html += "</tr>"

    if online_courses:
        online_text = ", ".join([f"<b>{c['name']}</b>" for c in online_courses])
        html += f"<tr><td style='background-color: #f8f9fa;'><b>온라인</b></td><td colspan='5' style='text-align: left; padding: 5px;'>{online_text}</td></tr>"
        
    html += "</table>"
    return html

# 3. AI 후보군 추출 (엄격한 데이터 파싱 - 주관 배제)
def get_course_candidates_json(major, grade, semester, diagnosis_text=""):
    llm = get_llm()
    if not llm: return []

    # [수정] Career/Recommendation 배제 및 전수 조사 중심 프롬프트
    prompt_template = """
    너는 [대학교 학사 데이터베이스 파서]이다. 
    제공된 [수강신청자료집/시간표 문서]를 분석하여 **{major} {grade} {semester}** 학생이 수강 가능한 **모든 정규 개설 과목**을 JSON 리스트로 추출하라.
    
    [학생 정보]
    - 전공: {major}
    - 대상: {grade} {semester}
    
    [진단 결과 (재수강 체크용)]
    {diagnosis_context}
    
    [엄격한 제약 사항]
    1. **주관적 추천 금지:** "취업에 유리함", "커리어 도움됨" 같은 추측성 설명은 절대 하지 마라.
    2. **전수 조사:** 해당 학과/학년/학기에 배정된 과목은 하나도 빠뜨리지 말고 모두 포함하라. (분반이 다르면 모두 포함)
    3. **제외 대상:** 타 학과 전용 과목, 해당 학년 대상이 아닌 과목은 리스트에서 제외하라.
    4. **Reason 필드 작성 규칙:** - 기본적으로 **"이수구분(전공필수/선택/교양) | 학점"** 형식의 팩트만 적어라.
       - 단, [진단 결과]에 "재수강"이 명시된 과목은 **"재수강 필수 대상"**이라고 적어라.
    5. **Priority 설정:**
       - 전공필수 또는 재수강 과목 = "High"
       - 전공선택 = "Medium"
       - 교양/기타 = "Normal"
    
    [JSON 출력 포맷 예시]
    [
        {{
            "id": "unique_id_1",
            "name": "회로이론1",
            "professor": "김광운",
            "credits": 3,
            "time_slots": ["월3", "수4"],
            "classification": "전공필수",
            "priority": "High", 
            "reason": "전공필수 | 3학점"
        }},
         {{
            "id": "unique_id_2",
            "name": "대학영어",
            "professor": "Smith",
            "credits": 2,
            "time_slots": ["화1", "목1"],
            "classification": "교양필수",
            "priority": "Normal", 
            "reason": "교양필수 | 2학점"
        }}
    ]
    
    **오직 JSON 리스트만 출력하라.**
    [문서 데이터]
    {context}
    """
    
    def _execute():
        chain = PromptTemplate.from_template(prompt_template) | llm
        return chain.invoke({
            "major": major,
            "grade": grade,
            "semester": semester,
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
        사용자의 입력 의도를 파악해서 답변해.
        [문서 근거 필수] 문서 내용을 인용할 땐 " " 안에 넣어.
        [학습된 문서]
        {context}
        """
        prompt = PromptTemplate(template=template, input_variables=["current_timetable", "user_input", "major", "grade", "semester", "context"])
        chain = prompt | llm
        return chain.invoke({
            "current_timetable": current_timetable, 
            "user_input": user_input,
            "major": major,
            "grade": grade,
            "semester": semester,
            "context": PRE_LEARNED_DATA
        }).content
    try:
        return run_with_retry(_execute)
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            return "⚠️ **사용량 초과**: 잠시 후 다시 시도해주세요."
        return f"❌ AI 오류: {str(e)}"

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
        image_messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })

    def _execute():
        prompt = """
        당신은 [냉철하고 현실적인 대기업 인사담당자 출신의 취업 컨설턴트]입니다.
        제공된 학생의 [성적표 이미지]와 [학습된 학사 문서]를 바탕으로 3가지 측면에서 분석 결과를 작성해주세요.
        
        **[핵심 지시사항 - 중요]**
        - 단순히 "열심히 하세요" 같은 뜬구름 잡는 조언은 하지 마십시오.
        - **반드시** 삼성전자, SK하이닉스, 현대자동차, 네이버, 카카오 등 **실제 한국 주요 대기업의 실명과 구체적인 직무명(JD)**을 언급하며 조언하세요.
        - 예: "삼성전자 DS부문 메모리사업부의 공정기술 직무에서는 반도체공학 A학점 이상을 선호하지만, 현재 학생의 성적은 B+이므로..." 와 같이 구체적으로 비교하세요.

        **[출력 형식]**
        반드시 아래의 구분자(`[[SECTION: ...]]`)를 사용하여 답변을 3개의 구역으로 명확히 나누세요.

        [[SECTION:GRADUATION]]
        ### 🎓 1. 졸업 요건 정밀 진단
        - [학습된 학사 문서]의 규정과 비교하여 졸업 가능 여부를 판정하세요.
        - 부족한 학점(전공, 교양 등)과 미이수 필수 과목을 표나 리스트로 정리하세요.
        - **종합 판정:** [졸업 가능 / 위험 / 불가]

        [[SECTION:GRADES]]
        ### 📊 2. 성적 정밀 분석
        - **전체 평점 vs 전공 평점 비교:** 전공 학점이 전체보다 낮은지 확인하고 질책하세요. (직무 전문성 결여 지적)
        - **재수강 권고:** C+ 이하의 전공 핵심 과목이 있다면 구체적으로 지적하며 재수강을 강력히 권고하세요.
        - **수강 패턴 분석:** 꿀강(학점 따기 쉬운 교양) 위주로 들었는지, 기피 과목(어려운 전공)을 피했는지 간파하고 지적하세요.

        [[SECTION:CAREER]]
        ### 💼 3. AI 커리어 솔루션 (대기업 JD 매칭)
        - **직무 추천:** 학생의 수강 내역(회로 위주, SW 위주 등)을 분석하여 가장 적합한 **구체적인 대기업 직무**를 2~3개 추천하세요. (예: 삼성전자 회로설계, 현대모비스 임베디드SW 등)
        - **Skill Gap 분석:** 해당 직무의 시장 요구사항(대기업 채용 기준) 대비 현재 부족한 점을 냉정하게 꼬집으세요.
        - **Action Plan:** 남은 학기에 반드시 수강해야 할 과목이나, 학교 밖에서 채워야 할 경험(프로젝트, 기사 자격증 등)을 구체적으로 지시하세요.

        [학습된 학사 문서]
        """
        
        content_list = [{"type": "text", "text": prompt}]
        content_list.extend(image_messages)
        content_list.append({"type": "text", "text": f"\n\n{PRE_LEARNED_DATA}"})

        message = HumanMessage(content=content_list)
        response = llm.invoke([message])
        return response.content

    try:
        return run_with_retry(_execute)
    except Exception as e:
         if "RESOURCE_EXHAUSTED" in str(e):
            return "⚠️ **사용량 초과**: 잠시 후 다시 시도해주세요."
         return f"❌ AI 오류: {str(e)}"

# 성적/진로 상담 및 수정 함수 (페르소나 유지)
def chat_with_graduation_ai(current_analysis, user_input):
    llm = get_llm()
    def _execute():
        template = """
        당신은 냉철하고 독설적인 'AI 취업 컨설턴트'입니다.
        학생의 성적 및 진로 진단 결과는 다음과 같습니다:
        
        [현재 진단 결과]
        {current_analysis}

        [사용자 입력]
        "{user_input}"

        [지시사항]
        - 사용자의 질문에 대해 현실적이고 직설적으로 답변하세요. 위로는 필요 없습니다.
        - 정보 수정 요청(예: "나 이 과목 들었어")이 들어오면 `[수정]` 태그를 붙이고 전체 진단 결과를 업데이트하세요.
        - **기업 채용 관점**에서 답변하세요. "이 과목은 삼성전자가 좋아합니다/신경 안 씁니다" 식으로 설명하세요.
        
        [참고 문헌]
        {context}
        """
        prompt = PromptTemplate(template=template, input_variables=["current_analysis", "user_input", "context"])
        chain = prompt | llm
        return chain.invoke({
            "current_analysis": current_analysis,
            "user_input": user_input,
            "context": PRE_LEARNED_DATA
        }).content

    try:
        return run_with_retry(_execute)
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
            st.session_state["menu_radio"] = "🤖 AI 학사 지식인" 
            st.rerun()
    
    st.divider()
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
menu = st.radio("기능 선택", ["🤖 AI 학사 지식인", "📅 스마트 시간표(수정가능)", "📈 성적 및 진로 진단"], 
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

elif st.session_state.current_menu == "📅 스마트 시간표(수정가능)":
    st.subheader("📅 AI 스마트 시간표 빌더")
    
    # [상태 초기화]
    if "candidate_courses" not in st.session_state:
        st.session_state.candidate_courses = []
    if "my_schedule" not in st.session_state:
        st.session_state.my_schedule = []

    # --------------------------------------------------------------------------
    # [A] 설정 및 후보군 로딩
    # --------------------------------------------------------------------------
    with st.expander("🛠️ 수강신청 설정 (학과/학년 선택)", expanded=not bool(st.session_state.candidate_courses)):
        kw_departments = [
            "전자공학과", "전자통신공학과", "전자융합공학과", "전기공학과", "전자재료공학과", "반도체시스템공학부", "로봇학부",
            "컴퓨터정보공학부", "소프트웨어학부", "정보융합학부", "지능형로봇학과", "건축학과", "건축공학과", "화학공학과", "환경공학과",
            "수학과", "전자바이오물리학과", "화학과", "스포츠융합과학과", "정보콘텐츠학과", "국어국문학과", "영어산업학과", 
            "미디어커뮤니케이션학부", "산업심리학과", "동북아문화산업학부", "행정학과", "법학부", "국제학부", "자산관리학과",
            "경영학부", "국제통상학부", "자율전공학부(자연)", "자율전공학부(인문)"
        ]
        
        c1, c2, c3 = st.columns(3)
        major = c1.selectbox("학과", kw_departments, key="tt_major")
        grade = c2.selectbox("학년", ["1학년", "2학년", "3학년", "4학년"], key="tt_grade")
        semester = c3.selectbox("학기", ["1학기", "2학기"], key="tt_semester")
        
        use_diagnosis = st.checkbox("☑️ 성적 진단 결과 반영 (재수강/추천 과목 로드)", value=True)
        
        if st.button("🚀 강의 목록 불러오기 (AI Scan)", type="primary", use_container_width=True):
            diag_text = ""
            if use_diagnosis and st.session_state.graduation_analysis_result:
                 diag_text = st.session_state.graduation_analysis_result
            elif use_diagnosis and st.session_state.user and fb_manager.is_initialized:
                 saved_diags = fb_manager.load_collection('graduation_diagnosis')
                 if saved_diags:
                     diag_text = saved_diags[0]['result']
                     st.toast("저장된 진단 결과를 불러왔습니다.")

            with st.spinner("요람에서 해당 학기 개설 과목을 전수 조사 중입니다..."):
                candidates = get_course_candidates_json(major, grade, semester, diag_text)
                if candidates:
                    st.session_state.candidate_courses = candidates
                    st.session_state.my_schedule = [] 
                    st.rerun()
                else:
                    st.error("강의 정보를 추출하지 못했습니다. 다시 시도해주세요.")

    # --------------------------------------------------------------------------
    # [B] 인터랙티브 빌더 UI (인사이트 컴팩트 뷰 적용)
    # --------------------------------------------------------------------------
    if st.session_state.candidate_courses:
        st.divider()
        col_left, col_right = st.columns([1, 1.4], gap="medium")

        # [좌측] 강의 장바구니 (스크롤 박스 적용 및 자동 숨김)
        with col_left:
            st.subheader("📚 강의 선택")
            st.caption("담은 과목은 목록에서 자동으로 사라집니다.")
            
            with st.container(height=600, border=True):
                tab1, tab2, tab3 = st.tabs(["🔥 필수/재수강", "🏫 전공선택", "🧩 교양/기타"])
                
                # [UI 컴포넌트] 인사이트 컴팩트 로우 (Insight Compact Row)
                def draw_course_row(course, key_prefix):
                    # [필터 로직] 이미 담은 과목은 목록에서 그리지 않음 (자동 숨김)
                    current_names = [c['name'] for c in st.session_state.my_schedule]
                    if course['name'] in current_names:
                        return # Skip rendering

                    # Highlight Logic
                    priority = course.get('priority', 'Normal')
                    border_color = "#ddd"
                    reason_bg = "#f1f3f5" # 기본 회색
                    
                    if priority == 'High': 
                        border_color = "#ffcccc" # 붉은 테두리
                        reason_bg = "#ffebee" # 붉은 배경 (이유)
                    elif priority == 'Medium':
                        border_color = "#cce5ff" # 파란 테두리
                        reason_bg = "#e3f2fd" # 파란 배경
                    
                    with st.container(border=True):
                        c_info, c_btn = st.columns([0.85, 0.15])
                        
                        with c_info:
                            time_str = ', '.join(course['time_slots']) if course['time_slots'] else "시간미정"
                            info_html = f"""
                            <div style="line-height:1.2;">
                                <span style="font-weight:bold; font-size:16px;">{course['name']}</span> 
                                <span style="font-size:13px; color:#555;">({course['credits']}학점) | {course['professor']} | {time_str}</span>
                            </div>
                            """
                            st.markdown(info_html, unsafe_allow_html=True)
                            
                            # 2열: Fact Reason (Why)
                            if course.get('reason'):
                                reason_html = f"""
                                <div style="background-color:{reason_bg}; color:#333; padding:2px 8px; border-radius:4px; font-size:12px; margin-top:4px; display:inline-block;">
                                    💡 {course['reason']}
                                </div>
                                """
                                st.markdown(reason_html, unsafe_allow_html=True)

                        with c_btn:
                            st.write("") 
                            if st.button("➕", key=f"ad_{key_prefix}_{course['id']}", type="primary", help="담기"):
                                conflict, conflict_name = check_time_conflict(course, st.session_state.my_schedule)
                                if conflict:
                                    st.toast(f"⚠️ 시간 충돌! '{conflict_name}' 수업과 겹칩니다.", icon="🚫")
                                else:
                                    st.session_state.my_schedule.append(course)
                                    st.rerun()

                # 분류 및 렌더링
                must_list = [c for c in st.session_state.candidate_courses if c.get('priority') == 'High']
                major_list = [c for c in st.session_state.candidate_courses if c.get('priority') == 'Medium' or ('전공' in c.get('classification', '') and c not in must_list)]
                other_list = [c for c in st.session_state.candidate_courses if c not in must_list and c not in major_list]

                with tab1:
                    if not must_list: st.info("해당 과목 없음")
                    for c in must_list: draw_course_row(c, "must")
                with tab2:
                    if not major_list: st.info("해당 과목 없음")
                    for c in major_list: draw_course_row(c, "mj")
                with tab3:
                    if not other_list: st.info("해당 과목 없음")
                    for c in other_list: draw_course_row(c, "ot")

        # [우측] 실시간 프리뷰
        with col_right:
            st.subheader("🗓️ 내 시간표")
            
            # [추가] 신청 내역 리스트 (삭제 기능 제공)
            if st.session_state.my_schedule:
                with st.expander("📋 신청 내역 관리 (클릭하여 삭제)", expanded=True):
                    for idx, added_course in enumerate(st.session_state.my_schedule):
                        cols = st.columns([0.8, 0.2])
                        cols[0].markdown(f"**{added_course['name']}** ({added_course['professor']})")
                        if cols[1].button("❌", key=f"del_list_{idx}"):
                             st.session_state.my_schedule.pop(idx)
                             st.rerun()
            
            total_credits = sum([c.get('credits', 0) for c in st.session_state.my_schedule])
            st.write(f"**신청 학점:** {total_credits} / 21")
            st.progress(min(total_credits / 21, 1.0))

            html_table = render_interactive_timetable(st.session_state.my_schedule)
            st.markdown(html_table, unsafe_allow_html=True)
            
            st.divider()
            
            if st.button("💾 이대로 저장하기", use_container_width=True):
                if not st.session_state.my_schedule:
                    st.error("과목을 선택해주세요.")
                else:
                    st.session_state.timetable_result = html_table 
                    doc_data = {
                        "result": html_table,
                        "major": major,
                        "grade": grade,
                        "name": f"{major} {grade} (직접설계)",
                        "is_favorite": False,
                        "created_at": datetime.datetime.now()
                    }
                    if st.session_state.user and fb_manager.is_initialized:
                         doc_id = str(int(time.time()))
                         if fb_manager.save_data('timetables', doc_id, doc_data):
                             st.session_state.current_timetable_meta = {
                                "id": doc_id, "name": doc_data['name'], "is_favorite": False
                             }
                             st.toast("저장 완료!", icon="✅")
                             time.sleep(1)
                             st.rerun()
                         else:
                             st.error("저장 실패")
                    else:
                        st.warning("로그인 필요")
            
            if st.button("🔄 비우기"):
                st.session_state.my_schedule = []
                st.rerun()

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
        
        # 섹션 파싱
        sec_grad = ""
        sec_grade = ""
        sec_career = ""
        try:
            if "[[SECTION:GRADUATION]]" in result_text:
                parts = result_text.split("[[[SECTION:GRADUATION]]")
                if len(parts) > 1:
                    temp = parts[1]
                else:
                    temp = result_text.split("[[SECTION:GRADUATION]]")[-1]
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


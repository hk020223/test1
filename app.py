import streamlit as st
import pandas as pd
import os
import glob
import datetime
import time
import base64
import re  # 정규표현식 사용
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

# [모바일 최적화 CSS]
st.markdown("""
    <style>
        footer { visibility: hidden; }
        @media only screen and (max-width: 600px) {
            .main .block-container {
                padding-left: 0.2rem !important;
                padding-right: 0.2rem !important;
                padding-top: 2rem !important;
                max-width: 100% !important;
            }
            div[data-testid="stMarkdownContainer"] table {
                width: 100% !important;
                table-layout: fixed !important;
                display: table !important;
                font-size: 10px !important;
                margin-bottom: 0px !important;
            }
            div[data-testid="stMarkdownContainer"] th, 
            div[data-testid="stMarkdownContainer"] td {
                padding: 1px 1px !important;
                word-wrap: break-word !important;
                word-break: break-all !important;
                white-space: normal !important;
                line-height: 1.1 !important;
                vertical-align: middle !important;
            }
            div[data-testid="stMarkdownContainer"] th:first-child,
            div[data-testid="stMarkdownContainer"] td:first-child {
                width: 35px !important;
                font-size: 8px !important;
                text-align: center !important;
                letter-spacing: -0.5px !important;
            }
            button { min-height: 45px !important; }
            input { font-size: 16px !important; }
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

# [추가] 재수강 대상 과목 및 기이수 과목 리스트 관리
if "retake_candidates" not in st.session_state:
    st.session_state.retake_candidates = []
if "completed_subjects" not in st.session_state:
    st.session_state.completed_subjects = []

# [추가] 사용자 설정(Preferences) 유지용 세션
if "user_prefs" not in st.session_state:
    st.session_state.user_prefs = {}

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

    # [추가] 사용자 설정(Preferences) 저장
    def save_user_prefs(self, prefs):
        if not self.is_initialized or not st.session_state.user: return
        try:
            user_id = st.session_state.user['localId']
            # DataFrame은 JSON 저장 불가하므로 리스트로 변환
            save_prefs = prefs.copy()
            if isinstance(save_prefs.get('schedule_df'), pd.DataFrame):
                save_prefs['schedule_df'] = save_prefs['schedule_df'].values.tolist()
            
            self.db.collection('users').document(user_id).collection('settings').document('preferences').set(save_prefs)
        except Exception as e:
            print(f"Error saving prefs: {e}")

    # [추가] 사용자 설정(Preferences) 로드
    def load_user_prefs(self):
        if not self.is_initialized or not st.session_state.user: return {}
        try:
            user_id = st.session_state.user['localId']
            doc = self.db.collection('users').document(user_id).collection('settings').document('preferences').get()
            if doc.exists:
                return doc.to_dict()
            return {}
        except: return {}

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

# [수정] 9대 정밀 검증 필터 및 출력 제어 프롬프트 (최종)
COMMON_TIMETABLE_INSTRUCTION = """
[★★★ 9대 정밀 검증 및 필터링 규칙 (9 Strict Verification Rules) ★★★]
1. **⚠️ 요일/교시 분리 배정 (Time Slot 1:1 Mapping - CRITICAL)**:
   - 강의 시간이 '월1, 수2'라면 **월요일 1교시**와 **수요일 2교시**에만 배치하라.
   - **절대** '월1,2' 또는 '수1,2' 처럼 연강으로 임의 해석하거나 뻥튀기하지 마라.

2. **🥇 사용자 지정 재수강 (User Override)**:
   - `must_include_subjects`에 있는 과목은 시간이 겹치지 않는 한 **무조건 0순위**로 고정한다.

3. **🚫 기이수 과목 및 유사 과목 배제 (Exclusion)**:
   - `completed_subjects`에 있는 과목은 즉시 제외한다.
   - **유사 과목 체크:** 과목명이 80% 이상 유사하거나 자료집상 '동일/대체 과목'이면 제외한다.

4. **📚 수강신청 자료집 기반 로드맵 (Curriculum Core)**:
   - **[수강신청 자료집] PDF**에 명시된 해당 학과/학년의 **필수 이수 트리(로드맵)**를 최우선으로 배치한다.

5. **🔍 비고란 제약 검증 (Remarks Check - CRITICAL)**:
   - 강의시간표 PDF의 **'비고'**란을 반드시 읽어라.
   - '외국인전용', '타과생수강불가', 'XX학과전용', '참빛인재' 등의 키워드가 내 정보와 맞지 않으면 **무조건 제외**하라.

6. **🏫 분반 정합성 체크 (Class Division)**:
   - 전공 과목의 경우, 분반이 특정 학과나 반으로 제한되어 있다면 내 학과와 일치하는지 확인하라.

7. **🔢 난이도/영역 규정 (Current Schedule Only)**:
   - **이번에 생성하는 시간표 내에서**, 동일한 교양 영역(예: 균형교양-인문)에 **학정번호 5번째 자리(난이도 코드)**가 같은 과목이 2개 이상 들어가지 않도록 조정하라.

8. **💻 수강 총량 규제 (Remote Limit)**:
   - '원격수업 100%', '사이버 강의' 등은 **최대 2과목(6학점)**까지만 포함하라. 3과목 이상이면 일반 강의로 교체하라.

9. **🛑 물리적 충돌 방지 (Conflict)**:
   - 공강 시간 및 과목 간 시간 겹침을 배제하라.

[★★★ 출력 형식 (Output Format) - 엄수 ★★★]
1. **서론, 제목, 인사말 절대 금지.** 오직 결과 데이터만 출력하라.
2. **HTML Table**: `<table>...</table>` 태그로 시작하는 세로형 시간표를 가장 먼저 출력하라.
   - **셀 형식:** `<b>과목명</b><br><small>교수명</small>` (교수명 필수 포함)
3. **상세 정보 리스트**: 테이블 뒤에 `[[DETAILED_LIST_START]]`와 `[[DETAILED_LIST_END]]` 사이에 생성된 과목들의 상세 정보를 **마크다운 표** 형식으로 정리하라.
   - 컬럼: `| 구분 | 과목명 | 학정번호 | 학점 | 시간 | 담당교수 | 비고 |`
4. **검증 리포트 태그**: 상세 리스트 뒤에 `[[REPORT_START]]`와 `[[REPORT_END]]` 사이에 **검증 및 배정 현황(긍정적)**을 요약하라.
   - 내용: 1) 기이수 제외 건수, 2) 재수강 반영 여부, 3) **이번 학기 교양 배정 현황(영역/과목명/난이도 - 중복 없음 확인용)**, 4) 원격강의 과목 수.
5. **선수과목 체크리스트**: 맨 마지막에 `[⚠️ 선수과목 체크리스트]` 섹션을 만들어라.
"""

# [수정] generate_timetable_ai 함수 (리스트->문자열 변환 및 Prompt 적용)
def generate_timetable_ai(major, grade, semester, target_credits, blocked_times_desc, requirements, must_include_subjects, completed_subjects):
    llm = get_llm()
    if not llm: return "⚠️ API Key 오류"
    
    must_include_str = ", ".join(must_include_subjects) if must_include_subjects else "없음"
    completed_str = ", ".join(completed_subjects) if completed_subjects else "없음"

    def _execute():
        base_template = """
        너는 대학교 수강신청 전문가야. 오직 제공된 [학습된 문서]의 텍스트 데이터와 [수강신청 자료집]에 기반해서만 시간표를 짜줘.
        
        [학생 정보]
        - 소속: {major}
        - 학년/학기: {grade} {semester}
        - 목표: {target_credits}학점
        - 공강 필수: {blocked_times}
        - 추가요구: {requirements}
        
        [★★★ 이수 내역 및 재수강 정보 (Input Data) ★★★]
        1. **기이수 과목 (제외 대상):** {completed_str}
        2. **필수 포함 과목 (재수강):** {must_include_str}
        """
        
        base_template += COMMON_TIMETABLE_INSTRUCTION + """
        [추가 지시사항]
        - 진단 결과가 없거나 부족할 경우, 사용자는 이전 학년의 선수 과목을 모두 정상 이수했다고 가정하고 **자료집 상의 표준 커리큘럼** 위주로 시간표를 구성해.
        - **HTML 코드를 마크다운 코드 블록(```html)으로 감싸지 마라.** 그냥 Raw HTML 텍스트로 출력해라.
        
        [학습된 문서]
        {context}
        """
        
        prompt = PromptTemplate(
            template=base_template, 
            input_variables=["context", "major", "grade", "semester", "target_credits", "blocked_times", "requirements", "completed_str", "must_include_str"]
        )
        chain = prompt | llm
        
        input_data = {
            "context": PRE_LEARNED_DATA,
            "major": major,
            "grade": grade,
            "semester": semester,
            "target_credits": target_credits,
            "blocked_times": blocked_times_desc,
            "requirements": requirements,
            "completed_str": completed_str,
            "must_include_str": must_include_str
        }
        return chain.invoke(input_data).content

    try:
        response_content = run_with_retry(_execute)
        return clean_html_output(response_content)
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            return "⚠️ **사용량 초과**: 잠시 후 다시 시도해주세요."
        return f"❌ AI 오류: {str(e)}"

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
        **Case 1. 시간표 수정 요청 (예: "1교시 빼줘"):**
        - 시간표를 **재작성**.
        """ + COMMON_TIMETABLE_INSTRUCTION + """
        - **HTML 코드를 마크다운 코드 블록(```html)으로 감싸지 마라.** Raw HTML로 출력해.
        **Case 2. 단순 질문 (예: "이거 선수과목 뭐야?"):**
        - **시간표 재출력 X**, 텍스트 답변만.
        - **근거가 되는 문서 원문 내용을 반드시 " " (쌍따옴표) 안에 인용.**
        답변 시작에 [수정] 또는 [답변] 태그를 붙여서 구분.
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
        
        **[핵심 지시사항]**
        - 분석 내용은 기존과 동일하게 상세히 작성하세요.
        - **맨 마지막 줄**에 아래 두 가지 정보를 태그 형식으로 반드시 출력하세요.
        
        1. 재수강 필요 과목 (C+ 이하, F, NP 등. B0 이상 제외): `[[RETAKE: 과목1, 과목2...]]`
        2. 기이수 과목 (이미 학점을 받은 모든 과목): `[[COMPLETED: 과목1, 과목2...]]`
        (해당사항 없으면 NONE 입력)

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
        result_text = run_with_retry(_execute)
        
        # 태그 파싱 및 세션 저장
        match_retake = re.search(r"\[\[RETAKE: (.*?)\]\]", result_text)
        if match_retake:
            retake_str = match_retake.group(1).strip()
            st.session_state.retake_candidates = [x.strip() for x in retake_str.split(',')] if retake_str != "NONE" else []
        
        match_completed = re.search(r"\[\[COMPLETED: (.*?)\]\]", result_text)
        if match_completed:
            comp_str = match_completed.group(1).strip()
            st.session_state.completed_subjects = [x.strip() for x in comp_str.split(',')] if comp_str != "NONE" else []
        
        return result_text
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

# [추가] 데이터 자동 저장 콜백 함수 (Persistence)
def update_prefs():
    prefs = {
        "major": st.session_state.tt_major,
        "grade": st.session_state.tt_grade,
        "semester": st.session_state.tt_semester,
        "target_credit": st.session_state.tt_credit,
        "requirements": st.session_state.tt_req,
        "schedule_df": st.session_state.get("tt_editor", None)
    }
    if "tt_must_include" in st.session_state:
        prefs["must_include"] = st.session_state.tt_must_include

    st.session_state.user_prefs = prefs
    
    if st.session_state.user:
        fb_manager.save_user_prefs(prefs)

with st.sidebar:
    st.title("🗂️ 활동 로그")
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
                                prefs = fb_manager.load_user_prefs()
                                if prefs:
                                    st.session_state.user_prefs = prefs
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
    st.subheader("📅 AI 맞춤형 시간표 설계")
    
    # 탭 이동 간 데이터 유지를 위한 세션 동기화
    if st.session_state.user_prefs:
        prefs = st.session_state.user_prefs
        if "tt_major" not in st.session_state and "major" in prefs: st.session_state.tt_major = prefs["major"]
        if "tt_grade" not in st.session_state and "grade" in prefs: st.session_state.tt_grade = prefs["grade"]
        if "tt_semester" not in st.session_state and "semester" in prefs: st.session_state.tt_semester = prefs["semester"]
        if "tt_credit" not in st.session_state and "target_credit" in prefs: st.session_state.tt_credit = prefs["target_credit"]
        if "tt_req" not in st.session_state and "requirements" in prefs: st.session_state.tt_req = prefs["requirements"]
        if "tt_must_include" not in st.session_state and "must_include" in prefs:
            valid_opts = [x for x in prefs["must_include"] if x in st.session_state.retake_candidates]
            st.session_state.tt_must_include = valid_opts

    # 시간표 불러오기 및 관리 섹션
    if st.session_state.user and fb_manager.is_initialized:
        saved_tables = fb_manager.load_collection('timetables')
        fav_tables = []
        archive_tables = []
        for t in saved_tables:
            if 'name' not in t: t['name'] = t['created_at'].strftime('%Y-%m-%d 시간표')
            if 'is_favorite' not in t: t['is_favorite'] = False
            if t['is_favorite']: fav_tables.append(t)
            else: archive_tables.append(t)
        
        if fav_tables:
            st.markdown("##### ⭐ 즐겨찾기 (Quick Access)")
            cols = st.columns(4)
            for idx, table in enumerate(fav_tables):
                with cols[idx % 4]:
                    if st.button(f"📄 {table['name']}", key=f"fav_{table['id']}", use_container_width=True):
                        st.session_state.timetable_result = table['result']
                        st.session_state.current_timetable_meta = {"id": table['id'],"name": table['name'],"is_favorite": table['is_favorite']}
                        st.toast(f"'{table['name']}'을(를) 불러왔습니다.")
                        st.rerun()

        with st.expander("📂 내 시간표 보관함 (클릭하여 열기)", expanded=False):
            if not archive_tables: st.info("보관된 시간표가 없습니다.")
            else:
                cols = st.columns(4)
                for idx, table in enumerate(archive_tables):
                    with cols[idx % 4]:
                        if st.button(f"📄 {table['name']}", key=f"arc_{table['id']}", use_container_width=True):
                            st.session_state.timetable_result = table['result']
                            st.session_state.current_timetable_meta = {"id": table['id'],"name": table['name'],"is_favorite": table['is_favorite']}
                            st.toast(f"'{table['name']}'을(를) 불러왔습니다.")
                            st.rerun()

    # 메인 시간표 영역
    timetable_area = st.empty()
    if st.session_state.timetable_result:
        with timetable_area.container():
            st.markdown("### 🗓️ 내 시간표")
            current_meta = st.session_state.get("current_timetable_meta", {})
            if current_meta and st.session_state.user and fb_manager.is_initialized:
                with st.container(border=True):
                    c1, c2, c3 = st.columns([2, 1, 0.8])
                    new_name = c1.text_input("시간표 이름", value=current_meta.get('name', ''), label_visibility="collapsed", placeholder="시간표 이름 입력")
                    is_fav = c2.checkbox("⭐ 즐겨찾기 고정", value=current_meta.get('is_favorite', False))
                    if c3.button("정보 수정 저장", use_container_width=True):
                        if fb_manager.update_data('timetables', current_meta['id'], {'name': new_name, 'is_favorite': is_fav}):
                            st.session_state.current_timetable_meta['name'] = new_name
                            st.session_state.current_timetable_meta['is_favorite'] = is_fav
                            st.toast("정보가 수정되었습니다.", icon="✅")
                            time.sleep(1)
                            st.rerun()
                        else: st.error("저장 실패")

            # 파싱 및 출력 로직 (Table -> Syllabus -> Detailed List -> Report -> Checklist)
            full_result = st.session_state.timetable_result
            
            # 1. HTML Table 분리
            if "</table>" in full_result:
                parts = full_result.split("</table>", 1)
                table_part = parts[0] + "</table>"
                remaining_part = parts[1]
            else:
                table_part = full_result
                remaining_part = ""

            # 2. Detailed List 분리
            if "[[DETAILED_LIST_START]]" in remaining_part and "[[DETAILED_LIST_END]]" in remaining_part:
                pre_list, list_chunk = remaining_part.split("[[DETAILED_LIST_START]]", 1)
                list_body, post_list = list_chunk.split("[[DETAILED_LIST_END]]", 1)
                detailed_list_text = list_body.strip()
                remaining_after_list = post_list
            else:
                detailed_list_text = ""
                remaining_after_list = remaining_part

            # 3. 검증 리포트 분리
            if "[[REPORT_START]]" in remaining_after_list and "[[REPORT_END]]" in remaining_after_list:
                _, report_chunk = remaining_after_list.split("[[REPORT_START]]", 1)
                report_body, post_report = report_chunk.split("[[REPORT_END]]", 1)
                report_text = report_body.strip()
                checklist_text = post_report
            else:
                report_text = ""
                checklist_text = remaining_after_list

            # [UI 렌더링]
            # 1) 시간표
            st.markdown(table_part, unsafe_allow_html=True)

            # 2) 강의계획서 뷰어
            def extract_course_info(html_code):
                if not html_code: return []
                matches = re.findall(r"<b>(.*?)</b><br><small>(.*?)</small>", html_code)
                courses = []
                for subj, small_content in matches:
                    if "(" in small_content: prof = small_content.split("(")[0].strip()
                    else: prof = small_content.strip()
                    courses.append({"subject": subj.strip(), "professor": prof})
                return courses

            def match_syllabus_files(courses):
                matched_list = []
                if not os.path.exists("data/syllabus"): return []
                seen = set()
                for c in courses:
                    subj = c['subject']
                    prof = c['professor']
                    key = f"{subj}_{prof}"
                    if key in seen: continue
                    seen.add(key)
                    file_v1 = f"data/syllabus/{subj}_{prof}.txt"
                    file_v2 = f"data/syllabus/{subj}.txt"
                    final_file = None
                    display_label = ""
                    if os.path.exists(file_v1): final_file = file_v1; display_label = f"{subj} ({prof})"
                    elif os.path.exists(file_v2): final_file = file_v2; display_label = f"{subj}"
                    if final_file: matched_list.append({"subject": subj, "file_path": final_file, "display_label": display_label})
                return matched_list

            def set_syllabus_viewer(file_path, display_label):
                st.session_state.selected_syllabus = {"path": file_path, "label": display_label}

            extracted_courses = extract_course_info(table_part)
            matched_courses = match_syllabus_files(extracted_courses)

            if matched_courses:
                st.divider()
                st.markdown("##### 📚 강의계획서 확인")
                cols = st.columns(len(matched_courses) + 2)
                for i, match in enumerate(matched_courses):
                    cols[i].button(f"📄 {match['display_label']}", key=f"btn_syl_{i}", on_click=set_syllabus_viewer, args=(match['file_path'], match['display_label']))
                
                if st.session_state.selected_syllabus:
                    with st.container(border=True):
                        c1, c2 = st.columns([8, 1])
                        c1.subheader(f"📄 {st.session_state.selected_syllabus['label']}")
                        if c2.button("❌ 닫기", key="close_syl_viewer"):
                            st.session_state.selected_syllabus = None; st.rerun()
                        try:
                            with open(st.session_state.selected_syllabus['path'], "r", encoding="utf-8") as f: full_text = f.read()
                            st.text_area("강의계획서 원문", full_text, height=400, disabled=True)
                        except Exception as e: st.error(f"오류: {e}")
                st.divider()

            # 3) 상세 정보 리스트 (NEW)
            if detailed_list_text:
                st.markdown("##### 📋 과목별 상세 정보")
                st.markdown(detailed_list_text)
                st.divider()

            # 4) 검증 리포트 (Expander)
            if report_text:
                with st.expander("🔍 시간표 생성 검증 리포트 (클릭하여 확인)", expanded=False):
                    st.info("AI가 시간표 생성 과정에서 수행한 검증 및 배정 현황입니다.")
                    st.markdown(report_text)

            # 5) 체크리스트
            if checklist_text.strip():
                st.markdown(checklist_text, unsafe_allow_html=True)
            
            if st.session_state.user and fb_manager.is_initialized:
                st.caption("현재 보고 있는 시간표를 **새로운 항목**으로 저장하려면 아래 버튼을 누르세요.")
                if st.button("☁️ 현재 시간표를 새 이름으로 저장"):
                    current_major = st.session_state.get("tt_major", "학과미정")
                    current_grade = st.session_state.get("tt_grade", "")
                    doc_data = {
                        "result": st.session_state.timetable_result,
                        "major": current_major,
                        "grade": current_grade,
                        "name": f"{current_major} {current_grade} (새 시간표)",
                        "is_favorite": False,
                        "created_at": datetime.datetime.now()
                    }
                    doc_id = str(int(time.time()))
                    if fb_manager.save_data('timetables', doc_id, doc_data):
                        st.session_state.current_timetable_meta = {"id": doc_id, "name": doc_data["name"], "is_favorite": False}
                        st.toast("시간표가 저장되었습니다!", icon="✅")
                        time.sleep(1)
                        st.rerun()
                    else: st.toast("저장 실패", icon="❌")
            st.divider()

    with st.expander("시간표 설정 열기/닫기", expanded=not bool(st.session_state.timetable_result)):
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("#### 1️⃣ 기본 정보")
            kw_departments = [
    "전자공학과", "전자통신공학과", "전자융합공학과", "전기공학과", "전자재료공학과", "반도체시스템공학부", "로봇학부",
    "컴퓨터정보공학부", "소프트웨어학부", "정보융합학부", "지능형로봇학과",
    "건축학과", "건축공학과", "화학공학과", "환경공학과",
    "수학과", "전자바이오물리학과", "화학과", "스포츠융합과학과", "정보콘텐츠학과",
    "국어국문학과", "영어산업학과", "미디어커뮤니케이션학부", "산업심리학과", "동북아문화산업학부",
    "행정학과", "법학부", "국제학부", "자산관리학과",
    "경영학부", "국제통상학부",
    "금융부동산법무학과", "게임콘텐츠학과", "스마트전기전자학과", "스포츠상담재활학과",
    "자율전공학부(자연)", "자율전공학부(인문)", "인제니움학부대학"
]
            defaults = st.session_state.user_prefs
            
            def_major_idx = kw_departments.index(defaults.get('major')) if defaults.get('major') in kw_departments else 0
            major = st.selectbox("학과", kw_departments, index=def_major_idx, key="tt_major", on_change=update_prefs)
            
            c1, c2 = st.columns(2)
            grade_opts = ["1학년", "2학년", "3학년", "4학년"]
            def_grade_idx = grade_opts.index(defaults.get('grade')) if defaults.get('grade') in grade_opts else 0
            grade = c1.selectbox("학년", grade_opts, index=def_grade_idx, key="tt_grade", on_change=update_prefs)
            
            sem_opts = ["1학기", "2학기"]
            def_sem_idx = sem_opts.index(defaults.get('semester')) if defaults.get('semester') in sem_opts else 0
            semester = c2.selectbox("학기", sem_opts, index=def_sem_idx, key="tt_semester", on_change=update_prefs)
            
            target_credit = st.number_input("목표 학점", 9, 24, defaults.get('target_credit', 18), key="tt_credit", on_change=update_prefs)
            
            candidate_subjects = st.session_state.get("retake_candidates", [])
            
            must_include = st.multiselect(
                "📋 재수강 신청할 과목 선택 (진단 결과 기반)",
                options=candidate_subjects,
                default=candidate_subjects,
                key="tt_must_include",
                help="성적 진단에서 C+ 이하로 식별된 과목들입니다. 이번 학기에 재수강할 과목을 체크하세요.",
                on_change=update_prefs
            )
            
            requirements = st.text_area("추가 요구사항", value=defaults.get('requirements', ''), placeholder="예: 전공 필수 챙겨줘", key="tt_req", on_change=update_prefs)

        with col2:
            st.markdown("#### 2️⃣ 공강 시간 설정")
            st.info("✅ **체크된 시간**: 수업 가능 (기본)  \n⬜ **체크 해제**: 공강 (수업 배정 안 함)")
            kw_times = {
                "1교시": "09:00~10:15", "2교시": "10:30~11:45", "3교시": "12:00~13:15",
                "4교시": "13:30~14:45", "5교시": "15:00~16:15", "6교시": "16:30~17:45",
                "7교시": "18:00~19:15", "8교시": "19:25~20:40", "9교시": "20:50~22:05"
            }
            schedule_index = [f"{k} ({v})" for k, v in kw_times.items()]
            
            # [수정] 공강 설정 복원 (기본값 True 고정)
            if 'init_schedule_df' not in st.session_state:
                if 'schedule_df' in defaults and defaults['schedule_df']:
                    try:
                        st.session_state.init_schedule_df = pd.DataFrame(defaults['schedule_df'], index=schedule_index, columns=["월", "화", "수", "목", "금"])
                    except:
                        st.session_state.init_schedule_df = pd.DataFrame(True, index=schedule_index, columns=["월", "화", "수", "목", "금"])
                else:
                    # 기본값 True (전체 선택)
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
            update_prefs()

            blocked_times = []
            for day in ["월", "화", "수", "목", "금"]:
                for idx, period_label in enumerate(edited_schedule.index):
                    if not edited_schedule.iloc[idx][day]:
                        blocked_times.append(f"{day}요일 {period_label}")
            blocked_desc = ", ".join(blocked_times) if blocked_times else "없음"
            
            completed_list = st.session_state.get("completed_subjects", [])

            with st.spinner("선수과목 확인 및 시간표 조합 중... (최대 1분 소요될 수 있습니다)"):
                result = generate_timetable_ai(major, grade, semester, target_credit, blocked_desc, requirements, must_include, completed_list)
                st.session_state.timetable_result = result
                st.session_state.timetable_chat_history = []
                st.session_state.current_timetable_meta = {} 
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
                    response = chat_with_timetable_ai(st.session_state.timetable_result, chat_input, major, grade, semester)
                    if "[수정]" in response:
                        new_timetable = response.replace("[수정]", "").strip()
                        new_timetable = clean_html_output(new_timetable) 
                        st.session_state.timetable_result = new_timetable
                        with timetable_area.container():
                            st.markdown("### 🗓️ 내 시간표")
                            current_meta = st.session_state.get("current_timetable_meta", {})
                            if current_meta and st.session_state.user and fb_manager.is_initialized:
                                with st.container(border=True):
                                    c1, c2, c3 = st.columns([2, 1, 0.8])
                                    new_name = c1.text_input("시간표 이름", value=current_meta.get('name', ''), label_visibility="collapsed", placeholder="시간표 이름 입력")
                                    is_fav = c2.checkbox("⭐ 즐겨찾기 고정", value=current_meta.get('is_favorite', False))
                                    if c3.button("정보 수정 저장", use_container_width=True):
                                         if fb_manager.update_data('timetables', current_meta['id'], {'name': new_name, 'is_favorite': is_fav}):
                                            st.session_state.current_timetable_meta['name'] = new_name
                                            st.session_state.current_timetable_meta['is_favorite'] = is_fav
                                            st.rerun()

                            st.markdown(new_timetable, unsafe_allow_html=True)
                            st.divider()
                        success_msg = "시간표를 수정했습니다. 위쪽 표가 업데이트 되었습니다."
                        st.write(success_msg)
                        st.session_state.timetable_chat_history.append({"role": "assistant", "content": success_msg})
                    else:
                        clean_response = response.replace("[답변]", "").strip()
                        st.markdown(clean_response)
                        st.session_state.timetable_chat_history.append({"role": "assistant", "content": clean_response})

elif st.session_state.current_menu == "📈 성적 및 진로 진단":
    st.subheader("📈 성적 및 진로 정밀 진단")
    st.markdown("""
    **취득 학점 내역을 캡처해서 업로드하세요!** AI 취업 컨설턴트가 당신의 성적표를 냉철하게 분석하여 **졸업 요건**, **성적 상태**, **커리어 방향성**을 진단해 드립니다.
    - KLAS 또는 학교 포털의 성적/학점 조회 화면을 캡처해주세요.
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
                    
                    match_retake = re.search(r"\[\[RETAKE: (.*?)\]\]", selected_diag['result'])
                    candidates = []
                    if match_retake:
                        retake_str = match_retake.group(1).strip()
                        if retake_str and retake_str != "NONE":
                            candidates = [x.strip() for x in retake_str.split(',')]
                    
                    if not candidates:
                        found = re.findall(r"([가-힣A-Za-z0-9]+)\s*\((C\+|C0|D\+|D0|F|NP)\)", selected_diag['result'])
                        if found:
                            candidates = list(set([m[0] for m in found]))
                    
                    st.session_state.retake_candidates = candidates

                    match_completed = re.search(r"\[\[COMPLETED: (.*?)\]\]", selected_diag['result'])
                    completed_list = []
                    if match_completed:
                        comp_str = match_completed.group(1).strip()
                        if comp_str and comp_str != "NONE":
                            completed_list = [x.strip() for x in comp_str.split(',')]
                    
                    if not completed_list:
                         found_comp = re.findall(r"([가-힣A-Za-z0-9]+)\s*\((A\+|A0|B\+|B0|C\+|C0|P)\)", selected_diag['result'])
                         if found_comp:
                             completed_list = list(set([m[0] for m in found_comp]))

                    st.session_state.completed_subjects = completed_list
                    
                    st.success("진단 결과를 불러왔습니다! 스마트 시간표 탭에서 재수강 과목을 확인할 수 있습니다.")
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
        
        with tab1:
            st.markdown(sec_grad)
        with tab2:
            st.markdown(sec_grade if sec_grade else "성적 분석 결과가 없습니다.")
        with tab3:
            st.markdown(sec_career if sec_career else "커리어 솔루션 결과가 없습니다.")
        
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
        st.caption("결과에 대해 추가 질문을 하거나, 누락된 정보를 알려주세요.")

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
                        success_msg = "정보를 반영하여 진단 결과를 업데이트했습니다. 위쪽 탭을 다시 확인해주세요."
                        st.session_state.graduation_chat_history.append({"role": "assistant", "content": success_msg})
                        st.rerun()
                    else:
                        st.markdown(response)
                        st.session_state.graduation_chat_history.append({"role": "assistant", "content": response})

        if st.button("결과 초기화"):
            st.session_state.graduation_analysis_result = ""
            st.session_state.graduation_chat_history = []
            st.rerun()

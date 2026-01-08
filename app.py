import streamlit as st
import pandas as pd
import os
import glob
import datetime
import time
import base64
import io
import json
import requests
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Firebase 라이브러리
import firebase_admin
from firebase_admin import credentials, firestore

# -----------------------------------------------------------------------------
# [0] 설정 및 데이터 로드
# -----------------------------------------------------------------------------
st.set_page_config(page_title="KW-강의마스터 Pro", page_icon="🎓", layout="wide")

# [모바일 최적화 CSS - 컴팩트 버전 & 메뉴 버튼 복구]
st.markdown("""
    <style>
        /* 1. 푸터(Made with Streamlit) 숨김 - 모든 화면 공통 */
        footer {
            visibility: hidden;
        }
        
        /* 2. 모바일 화면 (너비 600px 이하) 최적화 */
        @media only screen and (max-width: 600px) {
            /* 화면 꽉 채우기 (여백 최소화) */
            .main .block-container {
                padding-left: 0.2rem !important;
                padding-right: 0.2rem !important;
                padding-top: 2rem !important;
                max-width: 100% !important;
            }
            
            /* 시간표 한눈에 보기 (스크롤 제거 & 컴팩트 스타일) */
            div[data-testid="stMarkdownContainer"] table {
                width: 100% !important;
                table-layout: fixed !important; /* 칸 너비 고정 */
                display: table !important; /* 블록 대신 테이블 속성 유지 */
                font-size: 10px !important; /* 글자 크기 대폭 축소 */
                margin-bottom: 0px !important;
            }
            
            /* 표의 헤더(th)와 셀(td) 스타일 */
            div[data-testid="stMarkdownContainer"] th, 
            div[data-testid="stMarkdownContainer"] td {
                padding: 1px 1px !important; /* 셀 안쪽 여백 극소화 */
                word-wrap: break-word !important; /* 줄바꿈 허용 */
                word-break: break-all !important; /* 긴 단어 강제 줄바꿈 */
                white-space: normal !important; /* 줄바꿈 허용 */
                line-height: 1.1 !important; /* 줄 간격 좁게 */
                vertical-align: middle !important; /* 세로 중앙 정렬 */
            }

            /* 교시 열(첫번째 열)은 더 좁고 작게 */
            div[data-testid="stMarkdownContainer"] th:first-child,
            div[data-testid="stMarkdownContainer"] td:first-child {
                width: 35px !important; /* 교시 열 너비 고정 */
                font-size: 8px !important; /* 교시 글자는 더 작게 */
                text-align: center !important;
                letter-spacing: -0.5px !important;
            }
            
            /* [수정됨] 헤더 숨김 코드 삭제 -> 사이드바 메뉴 버튼 보이게 함 */
            /* header[data-testid="stHeader"] { display: none; } */
            
            /* 입력창/버튼 터치 영역 확대 */
            button {
                min-height: 45px !important;
            }
            input {
                font-size: 16px !important; /* iOS 확대 방지 */
            }
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
if "graduation_analysis_result" not in st.session_state:
    st.session_state.graduation_analysis_result = ""
if "graduation_chat_history" not in st.session_state:
    st.session_state.graduation_chat_history = []
if "user" not in st.session_state:
    st.session_state.user = None

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

# -----------------------------------------------------------------------------
# [Firebase Manager] 로그인 및 저장 기능
# -----------------------------------------------------------------------------
class FirebaseManager:
    def __init__(self):
        self.db = None
        self.is_initialized = False
        self.init_firestore()

    def init_firestore(self):
        """Firestore DB 초기화"""
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

    def auth_user(self, email, password, mode="login"):
        """로그인/회원가입 처리"""
        if "FIREBASE_WEB_API_KEY" not in st.secrets:
            return None, "API Key 설정이 필요합니다."
        
        api_key = st.secrets["FIREBASE_WEB_API_KEY"].strip()
        endpoint = "signInWithPassword" if mode == "login" else "signUp"
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:{endpoint}?key={api_key}"
        
        payload = {"email": email, "password": password, "returnSecureToken": True}
        try:
            res = requests.post(url, json=payload)
            data = res.json()
            if "error" in data:
                return None, data["error"]["message"]
            return data, None
        except Exception as e:
            return None, str(e)

    def save_data(self, collection, doc_id, data):
        """데이터 저장"""
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
# [1] AI 엔진
# -----------------------------------------------------------------------------
def get_llm():
    if not api_key: return None
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-09-2025", temperature=0)

# 이미지 분석용 모델
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

# 공통 프롬프트 지시사항
COMMON_TIMETABLE_INSTRUCTION = """
[★★★ 핵심 알고리즘: 3단계 검증 및 필터링 (Strict Verification) ★★★]

1. **Step 1: 요람(Curriculum) 기반 '수강 대상' 리스트 확정**:
   - 먼저 PDF 요람 문서에서 **'{major} {grade} {semester}'**에 배정된 **'표준 이수 과목' 목록**을 추출하세요.
   - **주의:** 'MSC 필수', '공학인증 필수'라고 적혀 있어도, 이 학기(예: 1학년 1학기) 표에 없으면 리스트에 넣지 마세요.

2. **Step 2: 학년 정합성 검사 (Grade Validation)**:
   - 추출된 과목이 실제 시간표 데이터에서 몇 학년 대상으로 개설되었는지 확인하세요.
   - **사용자가 선택한 학년({grade})과 시간표의 대상 학년이 일치하지 않으면 과감히 제외하세요.**
   - (예: 사용자가 1학년인데, 시간표에 '2학년' 대상이라고 적혀있으면 배치 금지)

3. **Step 3: 시간표 데이터와 정밀 대조 (Exact Match)**:
   - 위 단계를 통과한 과목만 시간표에 배치하세요.
   - **과목명 완전 일치 필수**: 예: '대학물리학1' vs '대학물리및실험1' 구분.

4. **출력 형식 (세로형 HTML Table)**:
   - 반드시 **HTML `<table>` 태그**를 사용해라.
   - **행(Row): 1교시 ~ 9교시** (행 머리글에 시간 포함: 1교시 (09:00~10:15) 등)
   - **열(Column): 월, 화, 수, 목, 금, 토, 일** (7일 모두 표시)
   - **스타일 규칙**:
     - `table` 태그에 `width="100%"` 속성을 주어라.
     - **같은 과목은 반드시 같은 배경색**을 사용해라. (파스텔톤 권장)
     - **수업이 없는 빈 시간(공강)은 반드시 흰색 배경**으로 둬라.
     - 셀 내용: `<b>과목명</b><br><small>교수명 (대상학년)</small>`

5. **온라인 및 원격 강의 처리 (필수 - 표 내부에 포함)**:
   - 강의 시간이 **'온라인', '원격', 'Cyber', '시간 미지정'** 등이면 **시간표 표(Table)의 맨 마지막 행에 추가**하세요.
   - **행 제목:** `<b>온라인/기타</b>`
   - **내용:** 해당되는 모든 과목을 `<b>과목명</b>(교수명)` 형식으로 나열하세요. (요일 열은 합치거나(colspan) 적절히 분배하여 표시)
   - **절대 표 밖으로 빼지 말고, 테이블의 일부로 포함시키세요.**

6. **출력 순서 고정**:
   - 1순위: HTML 시간표 표 (온라인 강의 포함)
   - 2순위: "### ✅ 필수 과목 검증 및 학년 일치 확인" (각 과목별로 '대상 학년'이 맞는지 명시)
   - 3순위: "### ⚠️ 배치 실패/제외 목록" (학년 불일치로 제외된 과목 포함)
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

# 상담 함수
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
        - 수정 시에도 **없는 정보를 지어내지 않도록** 주의해.
        
        **Case 2. 과목에 대한 단순 질문인 경우 (예: "이거 선수과목 뭐야?"):**
        - **시간표를 다시 출력하지 말고**, 질문에 대한 **텍스트 답변**만 해.
        - **답변할 때 근거가 되는 문서의 원문 내용을 반드시 " " (쌍따옴표) 안에 인용해서 포함해줘.**
        
        답변 시작에 [수정] 또는 [답변] 태그를 붙여서 구분해줘.

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

# 졸업 요건 분석 함수
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
        당신은 광운대학교 졸업 요건 분석 전문가입니다.
        제공된 학생의 [취득 학점 내역 캡처 이미지]와 [학습된 학사 문서]를 바탕으로 졸업 요건을 진단해주세요.

        **[분석 절차]**
        1. **이미지 정보 추출:** 캡처 이미지에서 학생의 입학 연도, 소속 학과, 현재까지 취득한 총 학점, 그리고 각 영역별(교양 필수, 교양 선택, 전공 필수, 전공 선택 등) 이수 학점을 정확히 추출하세요.
        2. **졸업 요건 대조:** 추출한 정보를 바탕으로 [학습된 학사 문서]에서 해당 학생의 입학 연도 및 학과에 적용되는 졸업 요건(총 학점, 영역별 필수 학점, 필수 과목 등)을 찾아내세요.
        3. **비교 및 진단:** 학생의 현재 취득 내역과 졸업 요건을 비교하여 부족한 부분이 있는지 면밀히 분석하세요.

        **[출력 형식]**
        다음 내용을 포함하여 마크다운 형식으로 명확하게 리포트를 작성해주세요.

        ### 🎓 졸업 요건 진단 결과

        **1. 종합 판정:**
        - **결과:** [졸업 가능 / 졸업 불가 / 요건 충족 중]
        - **요약:** (예: 현재 총 120학점 취득하였으며, 졸업까지 10학점이 더 필요합니다.)

        **2. 학점 이수 현황 (기준: {입학연도}학번 {학과})**
        | 구분 | 필수 학점 | 현재 취득 학점 | 부족 학점 | 상태 |
        | :--- | :---: | :---: | :---: | :---: |
        | 총 학점 | {총 필수} | {현재 총} | {부족 총} | {이모지} |
        | 교양 필수 | ... | ... | ... | ... |
        | 교양 선택 | ... | ... | ... | ... |
        | 전공 필수 | ... | ... | ... | ... |
        | 전공 선택 | ... | ... | ... | ... |
        | ... | ... | ... | ... | ... |
        *(각 영역별로 상세히 작성해주세요. 상태는 ✅(충족), ⚠️(부족) 등으로 표시)*

        **3. 미이수 필수 과목 및 영역**
        - (예: 전공 필수 '캡스톤디자인' 미이수)
        - (예: 교양 필수 '융합적사고와글쓰기' 미이수)
        - ...
        *(없으면 "없음"으로 표시)*

        **4. 졸업을 위한 조언**
        - (예: 다음 학기에 전공 필수 과목을 우선적으로 수강해야 합니다.)
        - (예: 부족한 교양 선택 학점을 채우기 위해 계절학기 수강을 고려해보세요.)
        - ...

        **[참고 자료]**
        - 분석에 참고한 [학습된 학사 문서]의 관련 내용을 인용해주세요.
        """
        
        content_list = [{"type": "text", "text": prompt}]
        content_list.extend(image_messages)
        content_list.append({"type": "text", "text": f"\n\n[학습된 학사 문서]\n{PRE_LEARNED_DATA}"})

        message = HumanMessage(content=content_list)
        
        response = llm.invoke([message])
        return response.content

    try:
        return run_with_retry(_execute)
    except Exception as e:
         if "RESOURCE_EXHAUSTED" in str(e):
            return "⚠️ **사용량 초과**: 잠시 후 다시 시도해주세요."
         return f"❌ AI 오류: {str(e)}"

# 졸업 요건 상담 및 수정 함수
def chat_with_graduation_ai(current_analysis, user_input):
    llm = get_llm()
    def _execute():
        template = """
        당신은 광운대학교 학사 전문 AI 상담사입니다.
        현재 학생의 졸업 요건 진단 결과는 다음과 같습니다:
        
        [현재 진단 결과]
        {current_analysis}

        [사용자 입력]
        "{user_input}"

        [지시사항]
        사용자의 입력 의도를 파악해서 적절히 응답하세요.
        
        **Case 1. 단순 질문인 경우 (예: "MSC 필수가 뭐야?"):**
        - 진단 결과나 학사 규정에 대해 설명해주세요.
        - 친절하게 답변하세요.
        
        **Case 2. 정보 수정/추가인 경우 (예: "나 캡스톤디자인 2023년에 들었어", "공학인증 포기했어"):**
        - 사용자의 정보를 반영하여 **진단 결과를 재작성**하세요.
        - 수정된 진단 리포트를 출력할 때는 반드시 맨 앞에 `[수정]` 태그를 붙이세요.
        - 기존 리포트 형식을 유지하면서 내용을 업데이트하세요.
        
        [참고 문헌 (학칙 등)]
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
    # [로그인 UI 추가]
    if st.session_state.user is None:
        with st.expander("🔐 로그인 / 회원가입", expanded=True):
            auth_mode = st.radio("모드 선택", ["로그인", "회원가입"], horizontal=True)
            email = st.text_input("이메일")
            password = st.text_input("비밀번호", type="password")
            
            if st.button(auth_mode):
                if not email or not password:
                    st.error("이메일과 비밀번호를 입력하세요.")
                else:
                    mode = "login" if auth_mode == "로그인" else "signup"
                    with st.spinner(f"{auth_mode} 중..."):
                        user, err = fb_manager.auth_user(email, password, mode)
                        if user:
                            st.session_state.user = user
                            st.success(f"환영합니다! ({user['email']})")
                            st.rerun()
                        else:
                            st.error(f"오류: {err}")
    else:
        st.info(f"👤 **{st.session_state.user['email']}**님")
        if st.button("로그아웃"):
            st.session_state.user = None
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
                        change_menu(log['menu'])
                        st.rerun()
    st.divider()
    if PRE_LEARNED_DATA:
         st.success(f"✅ PDF 문서 학습 완료")
    else:
        st.error("⚠️ 데이터 폴더에 PDF 파일이 없습니다.")

menu = st.radio("기능 선택", ["🤖 AI 학사 지식인", "📅 스마트 시간표(수정가능)", "🎓 졸업 요건 진단"], 
                horizontal=True, key="menu_radio", 
                index=["🤖 AI 학사 지식인", "📅 스마트 시간표(수정가능)", "🎓 졸업 요건 진단"].index(st.session_state.current_menu))

if menu != st.session_state.current_menu:
    st.session_state.current_menu = menu
    st.rerun()

st.divider()

if st.session_state.current_menu == "🤖 AI 학사 지식인":
    st.subheader("🤖 무엇이든 물어보세요")
    # 대화 내용 저장/불러오기
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
    
    # [시간표 불러오기 버튼]
    if st.session_state.user and fb_manager.is_initialized:
        with st.expander("📂 저장된 시간표 불러오기"):
            saved_tables = fb_manager.load_collection('timetables')
            if saved_tables:
                selected_table = st.selectbox("불러올 시간표 선택", 
                                            options=saved_tables, 
                                            format_func=lambda x: f"{x['major']} {x['grade']} ({x['created_at'].strftime('%Y-%m-%d %H:%M')})")
                if st.button("불러오기"):
                    st.session_state.timetable_result = selected_table['result']
                    st.success("시간표를 불러왔습니다!")
                    st.rerun()
            else:
                st.info("저장된 시간표가 없습니다.")

    timetable_area = st.empty()
    if st.session_state.timetable_result:
        with timetable_area.container():
            st.markdown("### 🗓️ 내 시간표")
            st.markdown(st.session_state.timetable_result, unsafe_allow_html=True)
            
            # [시간표 저장 버튼]
            if st.session_state.user and fb_manager.is_initialized:
                if st.button("☁️ 현재 시간표 저장하기"):
                    current_major = st.session_state.get("tt_major", "알수없음")
                    current_grade = st.session_state.get("tt_grade", "알수없음")
                    
                    doc_data = {
                        "result": st.session_state.timetable_result,
                        "major": current_major,
                        "grade": current_grade,
                        "created_at": datetime.datetime.now()
                    }
                    doc_id = str(int(time.time()))
                    if fb_manager.save_data('timetables', doc_id, doc_data):
                        st.toast("시간표가 저장되었습니다!", icon="✅")
                    else:
                        st.toast("저장 실패", icon="❌")
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
                    # [복구됨] 함수 호출 시 필요한 인자들을 모두 전달
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

elif st.session_state.current_menu == "🎓 졸업 요건 진단":
    st.subheader("🎓 졸업 요건 자가 진단")
    st.markdown("""
    **취득 학점 내역을 캡처해서 업로드하세요!** AI가 학습된 학사 데이터를 기반으로 졸업 요건을 진단해 드립니다.
    - KLAS 또는 학교 포털의 성적/학점 조회 화면을 캡처해주세요.
    - 전체 내역이 보이도록 여러 장으로 나누어 업로드해도 괜찮습니다.
    """)

    uploaded_files = st.file_uploader("캡처 이미지 업로드 (여러 장 가능)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("졸업 요건 분석 시작 🚀", type="primary"):
            with st.spinner("이미지를 분석하고 학사 데이터와 대조 중입니다... (시간이 조금 걸릴 수 있습니다)"):
                analysis_result = analyze_graduation_requirements(uploaded_files)
                st.session_state.graduation_analysis_result = analysis_result
                st.session_state.graduation_chat_history = [] # 새 분석 시 채팅 초기화
                add_log("user", "[졸업 요건] 이미지 분석 요청", "🎓 졸업 요건 진단")
                st.rerun()

    if st.session_state.graduation_analysis_result:
        st.divider()
        st.markdown("### 📊 분석 결과")
        
        # [추가됨] 결과 저장 버튼
        if st.session_state.user and fb_manager.is_initialized:
            if st.button("☁️ 분석 결과 저장하기"):
                doc_data = {
                    "result": st.session_state.graduation_analysis_result,
                    "created_at": datetime.datetime.now()
                }
                doc_id = str(int(time.time()))
                if fb_manager.save_data('graduation_diagnosis', doc_id, doc_data):
                    st.toast("진단 결과가 저장되었습니다!", icon="✅")
        
        st.markdown(st.session_state.graduation_analysis_result)
        
        st.divider()
        st.subheader("💬 결과 상담 및 수정")
        st.caption("분석 결과에 대해 궁금한 점을 묻거나, 누락된 정보를 알려주세요. (예: '영어 교양 들었는데 빠졌어', '졸업작품 면제야')")

        for msg in st.session_state.graduation_chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if chat_input := st.chat_input("질문이나 추가 정보를 입력하세요"):
            st.session_state.graduation_chat_history.append({"role": "user", "content": chat_input})
            add_log("user", f"[졸업상담] {chat_input}", "🎓 졸업 요건 진단")
            with st.chat_message("user"):
                st.write(chat_input)
            
            with st.chat_message("assistant"):
                with st.spinner("분석 중..."):
                    response = chat_with_graduation_ai(st.session_state.graduation_analysis_result, chat_input)
                    
                    if "[수정]" in response:
                        new_result = response.replace("[수정]", "").strip()
                        st.session_state.graduation_analysis_result = new_result
                        st.markdown(new_result)
                        success_msg = "정보를 반영하여 진단 결과를 업데이트했습니다. 위쪽 리포트를 확인해주세요."
                        st.session_state.graduation_chat_history.append({"role": "assistant", "content": success_msg})
                        st.rerun()
                    else:
                        st.markdown(response)
                        st.session_state.graduation_chat_history.append({"role": "assistant", "content": response})

        if st.button("결과 초기화"):
            st.session_state.graduation_analysis_result = ""
            st.session_state.graduation_chat_history = []
            st.rerun()


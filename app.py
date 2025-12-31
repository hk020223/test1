import streamlit as st
import pandas as pd
import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# -----------------------------------------------------------------------------
# [1] 서버 설정 및 데이터 로드
# -----------------------------------------------------------------------------
st.set_page_config(page_title="KW-강의마스터", page_icon="🎓", layout="wide")
api_key = os.environ.get("GOOGLE_API_KEY", "")

# 지식 베이스 로딩 함수 (data 폴더의 모든 PDF 읽기)
@st.cache_resource(show_spinner="학교 정보를 학습하는 중입니다... (약 1분 소요)")
def load_knowledge_base():
    all_content = ""
    
    # 'data' 폴더가 없으면 생성 (에러 방지용)
    if not os.path.exists("data"):
        os.makedirs("data")
        return ""

    # data 폴더 안의 모든 .pdf 파일 찾기
    pdf_files = glob.glob("data/*.pdf")
    
    if not pdf_files:
        return ""

    # 각 PDF 파일을 순서대로 읽어서 텍스트 합치기
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            pages = loader.load_and_split()
            
            # 파일명을 헤더로 추가해서 AI가 출처를 알게 함
            filename = os.path.basename(pdf_file)
            all_content += f"\n\n--- [문서 시작: {filename}] ---\n"
            
            for page in pages:
                all_content += page.page_content
                
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
            continue
            
    return all_content

# 앱 시작 시 한 번만 실행되어 모든 PDF를 메모리에 올림
PRE_LEARNED_DATA = load_knowledge_base()

# -----------------------------------------------------------------------------
# [2] AI 엔진 (질의응답 & 시간표 생성)
# -----------------------------------------------------------------------------
def get_llm():
    """모델 인스턴스 반환 (공통 사용)"""
    if not api_key:
        return None
    # 404 오류 방지를 위한 현재 환경 지원 모델
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-09-2025", temperature=0)

def ask_ai(question):
    llm = get_llm()
    if not llm:
        return "⚠️ 서버에 API Key가 설정되지 않았습니다."
    
    if not PRE_LEARNED_DATA: 
        return "⚠️ 학습된 데이터가 없습니다. VS Code의 'data' 폴더에 PDF 파일을 넣어주세요."

    try:
        template = """
        너는 광운대학교 학사 전문 상담 비서 'KW-강의마스터'야.
        너는 아래 제공된 [학습된 PDF 문서들]의 내용을 완벽하게 숙지하고 있어.
        
        [지시사항]
        1. 질문에 대한 답변은 오직 제공된 문서 내용에 기반해서 작성해.
        2. 답변할 때 "참고한 문서의 이름(예: 장학금규정.pdf)"을 언급해주면 더 좋아.
        3. 문서에 없는 내용은 솔직하게 모른다고 답해.

        [학습된 PDF 문서들]
        {context}

        [학생의 질문]
        {question}
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        chain = prompt | llm
        response = chain.invoke({"context": PRE_LEARNED_DATA, "question": question})
        return response.content
    except Exception as e:
        return f"❌ AI 오류: {str(e)}"

def generate_timetable_ai(grade, target_credits, free_days, requirements):
    llm = get_llm()
    if not llm:
        return "⚠️ 서버에 API Key가 설정되지 않았습니다."
    
    if not PRE_LEARNED_DATA: 
        return "⚠️ 학습된 데이터가 없습니다. 데이터가 없으면 시간표를 짤 수 없습니다."

    try:
        # 시간표 생성 전용 프롬프트
        template = """
        너는 대학교 수강신청 전문가야. 
        제공된 [학습된 PDF 문서들]에 포함된 '강의 시간표'와 '커리큘럼' 정보를 바탕으로 학생에게 최적화된 시간표를 짜줘.

        [학생 요구사항]
        - 학년: {grade}
        - 목표 학점: {target_credits}학점 내외
        - 공강 희망 요일(수업 없음): {free_days} (이 요일에는 절대 수업을 넣지 마)
        - 기타 요구사항: {requirements}

        [지시사항]
        1. PDF 문서 내에 있는 **실제 개설 과목**과 **수업 시간** 정보를 찾아서 배치해.
        2. 수업 시간이 겹치지 않게 배치해야 해.
        3. 학년과 전공 필수/선택 구분을 고려해서 추천해줘.
        4. 만약 PDF에 구체적인 '요일/교시' 정보가 없다면, 대략적인 커리큘럼 위주로 추천하고 "시간 정보가 문서에 없어 임의 배정했습니다"라고 명시해.
        5. 결과는 **가독성 좋은 마크다운 표**로 출력해줘. (요일별, 교시별 정리)
        6. 마지막에 왜 이 시간표를 추천했는지, 수강신청 유의사항(선수과목 등)이 있다면 같이 설명해줘.

        [학습된 PDF 문서들]
        {context}
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "grade", "target_credits", "free_days", "requirements"])
        chain = prompt | llm
        
        input_data = {
            "context": PRE_LEARNED_DATA,
            "grade": grade,
            "target_credits": target_credits,
            "free_days": ", ".join(free_days) if free_days else "없음",
            "requirements": requirements if requirements else "없음"
        }
        
        response = chain.invoke(input_data)
        return response.content
    except Exception as e:
        return f"❌ AI 오류: {str(e)}"

# -----------------------------------------------------------------------------
# [3] UI 구성
# -----------------------------------------------------------------------------
st.sidebar.title("🎓 KW-강의마스터")
# glob 모듈이 없는 경우 대비
try:
    pdf_count = len(glob.glob("data/*.pdf"))
except:
    pdf_count = 0
st.sidebar.info(f"📚 현재 {pdf_count}개의 문서를 학습했습니다.")

menu = st.sidebar.radio("메뉴", ["AI 학사 지식인", "이수학점 진단", "스마트 시간표"])

if menu == "AI 학사 지식인":
    st.header("🤖 AI 학사 지식인")
    st.caption("업로드된 PDF 문서들을 기반으로 답변합니다.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("질문하세요 (예: 이번 학기 장학금 기준이 뭐야?)"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("문서를 검색 중입니다..."):
                answer = ask_ai(user_input)
                st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

elif menu == "이수학점 진단":
    st.header("📊 졸업 이수 현황")
    col1, col2 = st.columns(2)
    with col1:
        major = st.number_input("전공 이수 학점", 0, 130, 45)
        ge = st.number_input("교양 이수 학점", 0, 130, 20)
    with col2:
        total = major + ge
        st.metric("현재 총 이수", f"{total} / 130")
        st.progress(total/130)

elif menu == "스마트 시간표":
    st.header("📅 AI 맞춤형 시간표 생성")
    st.info("업로드된 강의 시간표 PDF 파일을 기반으로 공강을 고려한 최적의 시간표를 생성합니다.")

    col1, col2 = st.columns(2)
    with col1:
        grade_input = st.selectbox("학년 선택", ["1학년", "2학년", "3학년", "4학년"])
        target_credit = st.number_input("목표 학점", 9, 24, 18)
    with col2:
        # 공강 요일 다중 선택
        free_days = st.multiselect("희망 공강 요일 (수업 제외)", ["월", "화", "수", "목", "금"])
        requirements = st.text_input("추가 요구사항 (예: 전공 필수 위주로, 오전 수업 제외 등)")

    if st.button("시간표 생성하기 ✨"):
        with st.spinner("강의 시간표 PDF를 분석하여 최적의 조합을 찾는 중입니다..."):
            result = generate_timetable_ai(grade_input, target_credit, free_days, requirements)
            st.markdown("### 🗓️ 추천 시간표")
            st.markdown(result)

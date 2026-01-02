import streamlit as st
import pandas as pd
import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# -----------------------------------------------------------------------------
# [0] 가상의 강의평 데이터 생성 (크롤링 대용)
# -----------------------------------------------------------------------------
# 실제로는 이 데이터를 csv 파일로 관리하거나 DB에 저장해야 합니다.
def load_mock_reviews():
    data = {
        "과목명": ["C프로그래밍", "C프로그래밍", "대학수학1", "공학설계입문", "대학영어"],
        "교수명": ["김코딩", "김코딩", "이수학", "박설계", "Brown"],
        "강의평": [
            "교수님 설명은 좋은데 과제가 진짜 너무 많아요. 매주 밤샘.",
            "학점은 잘 주시는 편입니다. 시험은 족보에서 많이 나옴.",
            "수포자라면 비추. 진도 엄청 빠름. 대신 질문은 잘 받아주심.",
            "조별과제 지옥... 팀원 잘못 만나면 한 학기 망함.",
            "출석만 잘 하면 B+은 깔고 감. 꿀강임."
        ],
        "시험정보": [
            "중간고사 코딩 테스트 손코딩 나옴",
            "기말은 프로젝트로 대체",
            "교재 연습문제 숫자만 바꿔서 나옴",
            "발표 비중이 큼",
            "오픈북 시험임"
        ]
    }
    return pd.DataFrame(data)

REVIEW_DB = load_mock_reviews()

# -----------------------------------------------------------------------------
# [1] 서버 설정 및 데이터 로드
# -----------------------------------------------------------------------------
st.set_page_config(page_title="KW-강의마스터", page_icon="🎓", layout="wide")
api_key = os.environ.get("GOOGLE_API_KEY", "")

# ... (기존 load_knowledge_base 함수 동일) ...
@st.cache_resource(show_spinner="학교 정보를 학습하는 중입니다...")
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
            all_content += f"\n\n--- [문서 시작: {filename}] ---\n"
            for page in pages:
                all_content += page.page_content
        except Exception as e:
            continue
    return all_content

PRE_LEARNED_DATA = load_knowledge_base()

# -----------------------------------------------------------------------------
# [2] AI 엔진
# -----------------------------------------------------------------------------
def get_llm():
    if not api_key: return None
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-09-2025", temperature=0)

def ask_ai(question):
    # ... (기존 ask_ai 함수 동일) ...
    llm = get_llm()
    if not llm: return "⚠️ API Key 오류"
    try:
        template = """
        [학습된 PDF 문서들] {context}
        [질문] {question}
        위 내용을 바탕으로 답변해줘.
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        chain = prompt | llm
        return chain.invoke({"context": PRE_LEARNED_DATA, "question": question}).content
    except Exception as e: return str(e)

# ★★★ 강의평 분석 AI 함수 추가 ★★★
def analyze_reviews_ai(course_name, professor_name):
    llm = get_llm()
    if not llm: return "⚠️ API Key 오류"

    # 해당 과목/교수의 리뷰 데이터 필터링
    relevant_reviews = REVIEW_DB[
        (REVIEW_DB['과목명'] == course_name) & 
        (REVIEW_DB['교수명'] == professor_name)
    ]

    if relevant_reviews.empty:
        return None # 데이터 없음

    # 리뷰 텍스트 합치기
    reviews_text = "\n".join(relevant_reviews['강의평'].tolist())
    exams_text = "\n".join(relevant_reviews['시험정보'].tolist())

    try:
        template = """
        너는 수강신청 도우미 AI야. 학생들의 강의평 데이터를 요약해서 알려줘.

        [강의평 데이터]
        {reviews}

        [시험 정보 데이터]
        {exams}

        [지시사항]
        1. **한 줄 요약**: 이 강의의 전반적인 분위기를 한 문장으로 요약해.
        2. **장점/단점**: 핵심 키워드(과제 양, 학점, 강의력 등) 위주로 정리해.
        3. **시험 꿀팁**: 시험 스타일이나 대비 방법을 알려줘.
        4. 어조는 대학생 선배가 조언해주듯이 친근하게 해.
        """
        prompt = PromptTemplate(template=template, input_variables=["reviews", "exams"])
        chain = prompt | llm
        
        response = chain.invoke({"reviews": reviews_text, "exams": exams_text})
        return response.content
    except Exception as e:
        return f"분석 오류: {str(e)}"

def generate_timetable_ai(major, grade, semester, target_credits, free_days, requirements):
    # ... (기존 generate_timetable_ai 함수 로직 유지) ...
    # 다만 프롬프트에 "강의평 데이터를 참고하여 꿀강/헬강을 구분해달라"는 내용을 추가할 수 있음
    llm = get_llm()
    if not llm: return "⚠️ API Key 오류"
    
    # 강의평 데이터를 프롬프트에 주입하기 위해 텍스트로 변환
    review_summary_str = REVIEW_DB.to_string()

    try:
        template = """
        너는 대학교 수강신청 전문가야. 
        [학습된 PDF 문서들](시간표, 요람)과 [학생들의 강의평 데이터]를 모두 고려해서 최적의 시간표를 짜줘.

        [학생 정보]
        - {major} / {grade} {semester} / 목표 {target_credits}학점
        - 공강 희망: {free_days}
        - 요구사항: {requirements}

        [학생들의 리얼 강의평 데이터 (참고용)]
        {review_data}

        [지시사항]
        1. PDF에서 필수 과목과 시간을 찾아서 시간표를 구성해.
        2. **중요**: 강의평 데이터를 참고해서, 만약 "과제가 너무 많다"거나 "팀플 지옥"인 과목이 있다면, 시간표 추천 이유에 **경고 메시지**를 함께 적어줘. (예: "⚠️ 이 수업은 조별과제가 빡세다는 평이 있습니다.")
        3. 결과는 **마크다운 표**로 작성하고, 그 아래에 상세 분석을 적어줘.

        [학습된 PDF 문서들]
        {context}
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "major", "grade", "semester", "target_credits", "free_days", "requirements", "review_data"])
        chain = prompt | llm
        
        input_data = {
            "context": PRE_LEARNED_DATA,
            "major": major,
            "grade": grade,
            "semester": semester,
            "target_credits": target_credits,
            "free_days": ", ".join(free_days) if free_days else "없음",
            "requirements": requirements if requirements else "없음",
            "review_data": review_summary_str
        }
        
        return chain.invoke(input_data).content
    except Exception as e: return str(e)


# -----------------------------------------------------------------------------
# [3] UI 구성
# -----------------------------------------------------------------------------
st.sidebar.title("🎓 KW-강의마스터")
menu = st.sidebar.radio("메뉴", ["AI 학사 지식인", "스마트 시간표", "강의평 분석(Beta)"])

if menu == "AI 학사 지식인":
    st.header("🤖 AI 학사 지식인")
    # ... (기존 코드 동일) ...
    if user_input := st.chat_input("질문하세요"):
        # ... (기존 코드 동일) ...
        st.write(ask_ai(user_input))

elif menu == "스마트 시간표":
    st.header("📅 AI 맞춤형 시간표")
    # ... (기존 폼 코드 동일) ...
    with st.form("timetable_form"):
        col1, col2 = st.columns(2)
        with col1:
            major_input = st.text_input("학과", "전자융합공학과")
            grade_input = st.selectbox("학년", ["1학년", "2학년", "3학년", "4학년"])
            semester_input = st.selectbox("학기", ["1학기", "2학기"])
        with col2:
            target_credit = st.number_input("목표 학점", 9, 24, 19)
            free_days = st.multiselect("공강 희망", ["월", "화", "수", "목", "금"])
            requirements = st.text_input("요구사항")
        submitted = st.form_submit_button("생성하기")

    if submitted:
        with st.spinner("분석 중..."):
            result = generate_timetable_ai(major_input, grade_input, semester_input, target_credit, free_days, requirements)
            st.markdown(result, unsafe_allow_html=True)

elif menu == "강의평 분석(Beta)":
    st.header("🔍 강의평 AI 분석")
    st.info("학생들의 강의평 데이터를 AI가 분석하여 핵심만 요약해 드립니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        # DB에 있는 과목만 선택하게 함
        c_name = st.selectbox("과목명", REVIEW_DB['과목명'].unique())
    with col2:
        p_name = st.selectbox("교수명", REVIEW_DB[REVIEW_DB['과목명'] == c_name]['교수명'].unique())

    if st.button("분석 결과 보기"):
        with st.spinner("리뷰 데이터를 분석하는 중..."):
            analysis = analyze_reviews_ai(c_name, p_name)
            if analysis:
                st.success(f"✅ {c_name}({p_name}) 분석 결과")
                st.markdown(analysis)
            else:
                st.error("해당 과목의 리뷰 데이터가 없습니다.")

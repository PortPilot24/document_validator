# app.py (최종 안정화 버전)

import streamlit as st
import pandas as pd
import json
import re
import os
import base64
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage

# --- 1. 기본 설정 및 모델/DB 로드 ---
st.set_page_config(page_title="AI 세관 서류 자동 검토 시스템", page_icon="🤖")

@st.cache_resource
def load_resources():
    try:
        embeddings_model = OpenAIEmbeddings()
        vectordb = Chroma(persist_directory='./customs_rules_db', embedding_function=embeddings_model)
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
        return vectordb, llm
    except Exception as e:
        st.error(f"AI 리소스 로딩 중 오류 발생: {e}. OPENAI_API_key가 올바르게 설정되었는지 확인하세요.")
        return None, None

@st.cache_data
def load_rule_sheet(path):
    try:
        df = pd.read_excel(path, sheet_name='Sheet1')
        return df
    except FileNotFoundError:
        st.error(f"규칙 파일({path})을 찾을 수 없습니다. customs_app 폴더에 파일이 있는지 확인하세요.")
        return None

# --- 2. 핵심 기능 함수들 ---
def encode_image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

def parse_image_to_json_text(image_bytes, llm):
    """GPT-4o Vision으로 이미지를 분석하여 원본 텍스트 응답을 반환"""
    base64_image = encode_image_to_base64(image_bytes)
    message = HumanMessage(
        content=[
            {"type": "text", "text": "이 이미지는 대한민국의 수입신고서입니다. 모든 텍스트를 분석하여, 각 항목에 맞는 데이터를 추출한 후, 요청하는 JSON 형식으로만 응답해주세요. 없는 정보는 null, 날짜는 YYYYMMDD, 숫자는 콤마 없이 변환해주세요. 부가 설명 없이 JSON 객체만 응답하세요. [요청하는 JSON 형식] {\"서류종류\": \"수입신고서\", \"필드\": {\"①신고번호\": \"string\", \"②신고일\": \"string\", \"⑩수입자\": {\"상호\": \"string\", \"수입자구분\": \"string\"}, \"⑪납세의무자\": {\"상호\": \"string\"}, \"㊼결제금액\": {\"인도조건\": \"string\"}}} "},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]
    )
    response = llm.invoke([message])
    return response.content

def validate_document(data, rules_df):
    errors = []
    fields = data.get('필드', {})
    importer_name = fields.get('⑩수입자', {}).get('상호')
    if not importer_name:
        errors.append({"field_name": "수입자 상호", "user_value": "없음(null)", "error_message": "필수 항목인 수입자 상호가 누락되었습니다."})
    
    importer_category = fields.get('⑩수입자', {}).get('수입자구분')
    taxpayer_name = fields.get('⑪납세의무자', {}).get('상호')
    if importer_name and taxpayer_name and importer_name == taxpayer_name and importer_category == 'B':
        errors.append({"field_name": "수입자 구분", "user_value": importer_category, "error_message": "수입자와 납세의무자 정보가 동일하지만 'B'(상이)로 신고되었습니다."})
        
    return errors

def get_intelligent_error_report(error_info, vectordb, llm):
    context_from_db = "관련 규정을 찾을 수 없음"
    if vectordb:
        retrieved_docs = vectordb.similarity_search(error_info["field_name"], k=1)
        if retrieved_docs: context_from_db = retrieved_docs[0].page_content
    template = "당신은 베테랑 관세사입니다. 오류 항목: {field}, 사용자의 입력값: {value}, 시스템이 발견한 문제: {error}, 관련된 작성 규정 (참고): {context} 정보를 바탕으로 오류 리포트를 상세히 작성해주세요."
    prompt = PromptTemplate.from_template(template)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    report = llm_chain.run({"field": error_info["field_name"], "value": error_info["user_value"], "error": error_info["error_message"], "context": context_from_db})
    return report

# --- 3. Streamlit UI 구성 ---
st.title("🤖 AI 세관 서류 자동 검토 시스템")
st.markdown("---")

vectordb, llm = load_resources()
uploaded_file = st.file_uploader("검토할 수입신고서 이미지 파일을 업로드하세요 (PNG, JPG)", type=['png', 'jpg'])

if uploaded_file is not None and llm is not None:
    st.image(uploaded_file, caption='업로드된 신고서 이미지')
    
    if st.button("검증 시작"):
        with st.spinner("AI가 이미지를 분석하고 있습니다..."):
            try:
                response_text = parse_image_to_json_text(uploaded_file.getvalue(), llm)
                
                # 🚨 --- 수정된 부분: AI의 응답에서 순수 JSON만 추출하는 필터 로직 ---
                # 응답에서 첫 '{'와 마지막 '}'를 찾아 그 사이의 내용만 잘라냅니다.
                match = re.search(r"\{.*\}", response_text, re.DOTALL)
                if match:
                    json_string = match.group(0)
                    parsed_json = json.loads(json_string)
                    st.success("AI 분석 및 JSON 변환 성공!")
                    st.session_state['parsed_json'] = parsed_json
                    
                    with st.expander("AI가 분석한 JSON 데이터 보기"):
                        st.json(parsed_json)
                else:
                    st.error("AI의 응답에서 유효한 JSON 형식을 찾을 수 없습니다.")
                    st.text_area("LLM 원본 응답 내용", response_text, height=200)
                    st.stop()
            except Exception as e:
                st.error(f"OCR/파싱 단계에서 오류가 발생했습니다: {e}")
                st.stop()

        if 'parsed_json' in st.session_state:
            with st.spinner("규칙 기반으로 데이터를 검증하고 있습니다..."):
                rules_df = load_rule_sheet('수입신고서_검증규칙.xlsx')
                if rules_df is not None:
                    errors = validate_document(st.session_state['parsed_json'], rules_df)
                    st.session_state['errors'] = errors
                    st.success("규칙 검증 완료!")

        if 'errors' in st.session_state:
            errors = st.session_state['errors']
            st.markdown("---")
            st.subheader("✅ 최종 검증 결과")

            if not errors:
                st.balloons()
                st.success("축하합니다! 발견된 오류가 없습니다.")
            else:
                st.error(f"총 {len(errors)}개의 오류가 발견되었습니다.")
                with st.spinner("AI가 오류에 대한 상세 리포트를 생성 중입니다..."):
                    for error in errors:
                        report = get_intelligent_error_report(error, vectordb, llm)
                        st.markdown(report)
                        st.markdown("---")

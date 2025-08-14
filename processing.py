# processing.py

import pandas as pd
import json
import re
import base64
import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage

# api키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

print(f"✅ API Key 로드 완료: {OPENAI_API_KEY[:8]}...")  # 앞 몇 글자만 출력

# --- 1. 리소스 로드 함수 ---
# (앱 시작 시 한번만 로드되도록 main.py에서 호출할 예정)
def load_ai_resources():
    embeddings_model = OpenAIEmbeddings()

    persist_dir = os.getenv("CHROMA_DIR", "/home/site/chromadb")
    os.makedirs(persist_dir, exist_ok=True)  # 폴더 없으면 생성

    vectordb = Chroma(persist_directory=persist_dir, 
                      embedding_function=embeddings_model)
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, max_tokens=2048)
    return vectordb, llm

def load_rule_sheet(path='수입신고서_검증규칙.xlsx'):
    df = pd.read_excel(path, sheet_name='Sheet1')
    return df

# --- 2. 핵심 처리 함수들 ---
def parse_image_to_json(image_bytes, llm):
    """GPT-4o Vision으로 이미지를 분석하여 JSON으로 변환"""
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    message = HumanMessage(
        content=[
            {"type": "text", "text": "이 이미지는 대한민국의 수입신고서입니다. 모든 텍스트를 분석하여, 각 항목에 맞는 데이터를 추출한 후, 요청하는 JSON 형식으로만 응답해주세요. 없는 정보는 null, 날짜는 YYYYMMDD, 숫자는 콤마 없이 변환해주세요. 부가 설명 없이 JSON 객체만 응답하세요. [요청하는 JSON 형식] {\"서류종류\": \"수입신고서\", \"필드\": {\"①신고번호\": \"string\", \"②신고일\": \"string\", \"⑩수입자\": {\"상호\": \"string\", \"수입자구분\": \"string\"}, \"⑪납세의무자\": {\"상호\": \"string\"}, \"㊼결제금액\": {\"인도조건\": \"string\"}}} "},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]
    )
    response = llm.invoke([message])
    
    # AI 응답에서 순수 JSON만 추출하는 필터 로직
    match = re.search(r"\{.*\}", response.content, re.DOTALL)
    if match:
        json_string = match.group(0)
        return json.loads(json_string)
    else:
        raise ValueError("AI 응답에서 유효한 JSON 형식을 찾을 수 없습니다.")

# processing.py 파일의 validate_document 함수를 아래 코드로 교체하세요.

def validate_document(data, rules_df):
    """JSON 데이터를 규칙 시트와 비교하여 오류 목록을 반환 (완성본)"""
    errors = []
    fields = data.get('필드', {})

    # --- 검증 1: '수입자 상호' 필수값 누락 ---
    importer_data = fields.get('⑩수입자', {})
    importer_name = importer_data.get('상호')
    if not importer_name:
        errors.append({
            "field_name": "수입자 상호",
            "user_value": "없음(null)",
            "error_message": "필수 항목인 수입자 상호가 누락되었습니다."
        })

    # --- 검증 2: '수입자 구분' 교차 검증 ---
    taxpayer_data = fields.get('⑪납세의무자', {})
    importer_category = importer_data.get('수입자구분')
    taxpayer_name = taxpayer_data.get('상호')
    
    # OCR로 읽은 두 이름이 모두 존재하고, 서로 같은데 'B'로 신고한 경우
    if importer_name and taxpayer_name and importer_name == taxpayer_name and importer_category == 'B':
        errors.append({
            "field_name": "수입자 구분",
            "user_value": importer_category,
            "error_message": f"수입자({importer_name})와 납세의무자({taxpayer_name}) 정보가 동일하지만 'B'(상이)로 신고되었습니다."
        })

    # --- 검증 3: '결제금액 인도조건' 코드 유효성 검증 ---
    incoterm = fields.get('㊼결제금액', {}).get('인도조건')
    valid_incoterms = ["FOB", "CIF", "CFR", "EXW", "FAS", "FCA", "CPT", "CIP", "DAP", "DPU", "DDP"]
    if incoterm and incoterm not in valid_incoterms:
        errors.append({
            "field_name": "결제금액 인도조건",
            "user_value": incoterm,
            "error_message": f"'{incoterm}'은(는) 유효한 INCOTERMS 코드가 아닙니다."
        })
        
    # --- 여기에 엑셀 마스터 시트의 다른 규칙들을 계속 추가할 수 있습니다 ---

    return errors

def get_intelligent_error_report(error_info, vectordb, llm):
    """오류 정보를 바탕으로 RAG와 LLM을 통해 상세 리포트를 생성"""
    # ... (이전의 리포트 생성 로직 전체) ...
    context_from_db = "관련 규정을 찾을 수 없음"
    if vectordb:
        retrieved_docs = vectordb.similarity_search(error_info["field_name"], k=1)
        if retrieved_docs: context_from_db = retrieved_docs[0].page_content
    template = "당신은 베테랑 관세사입니다. 오류 항목: {field}, 사용자의 입력값: {value}, 시스템이 발견한 문제: {error}, 관련된 작성 규정 (참고): {context} 정보를 바탕으로 오류 리포트를 상세히 작성해주세요."
    prompt = PromptTemplate.from_template(template)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    report = llm_chain.run({"field": error_info["field_name"], "value": error_info["user_value"], "error": error_info["error_message"], "context": context_from_db})
    return report
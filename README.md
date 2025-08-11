# 🤖 AI 세관 서류 자동 검토 API (Document Validator)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

지능형 항만 플랫폼 `PortPilot24`의 핵심 기능으로, 이미지 형태의 세관 서류를 AI가 자동으로 검토하고 오류를 리포트하는 FastAPI 기반의 API 서버입니다.

## ✨ 핵심 기능

-   **이미지 기반 서류 인식:** GPT-4o Vision API를 활용하여 수입/수출 신고서 이미지의 텍스트와 구조를 인식합니다.
-   **구조화된 데이터 자동 추출:** 인식된 텍스트를 분석하여 시스템이 이해할 수 있는 표준 JSON 형식으로 자동 변환합니다.
-   **규칙 기반 유효성 검증:** 사전에 정의된 엑셀 '마스터 규칙 시트'를 기반으로 각 항목의 형식, 필수값, 논리적 관계(교차 검증)를 검사합니다.
-   **RAG & LLM 기반 지능형 리포팅:** 오류 감지 시, Vector DB에 저장된 법규 지식 베이스를 RAG로 검색하여 관련 근거를 찾고, LLM이 이를 바탕으로 전문가 수준의 상세 분석 리포트를 생성합니다.

## ⚙️ 시스템 아키텍처

본 시스템은 3단계의 파이프라인으로 작동합니다.

**이미지 입력 → [1. OCR & JSON 파서] → 구조화된 데이터 → [2. 규칙/교차 검증 엔진] → 오류 목록 → [3. RAG + LLM 리포터] → 최종 분석 리포트**

## 🛠️ 기술 스택

-   **언어:** Python 3.11
-   **API 프레임워크:** FastAPI, Uvicorn
-   **AI/LLM 프레임워크:** LangChain
-   **AI 모델:** OpenAI GPT-4o (Vision & Language)
-   **벡터 데이터베이스:** ChromaDB
-   **데이터 처리:** Pandas, OpenPyXL

## 🚀 실행 방법

### 1. 프로젝트 복제
```bash
git clone [https://github.com/PortPilot24/document_validator.git](https://github.com/PortPilot24/document_validator.git)
cd document_validator
```

### 2. 가상환경 생성 및 활성화
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows Git Bash)
source venv/Scripts/activate
```

### 3. 필요 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 4. API 키 설정
프로젝트를 실행하려면 OpenAI API 키가 필요합니다. 터미널에서 아래 명령어를 사용하여 환경 변수를 설정해주세요.
```bash
# "sk-..." 부분에 실제 API 키를 입력하세요.
export OPENAI_API_KEY="sk-..."
```

### 5. API 서버 실행
```bash
uvicorn main:app --reload
```
서버가 성공적으로 실행되면 터미널에 `Uvicorn running on http://127.0.0.1:8000` 메시지가 나타납니다.

### 6. API 테스트

#### 로컬 개발 환경
서버 실행 후 웹 브라우저에서 다음 주소로 접속합니다.

- API 문서: **`http://127.0.0.1:8000/docs`**
- 파일 업로드: **POST** `http://127.0.0.1:8000/customs_review`
- 헬스 체크: **GET** `http://127.0.0.1:8000/health`

#### 통합 환경(Nginx 뒤)
통합 플랫폼(Nginx 라우팅)에서는 서비스별 프리픽스를 사용합니다.

- API 문서: **`https://<게이트웨이 주소>/document-validator/docs`**
- 파일 업로드: **POST** `https://<게이트웨이 주소>/document-validator/customs_review`
- 헬스 체크: **GET** `https://<게이트웨이 주소>/document-validator/health`

## 📁 프로젝트 구조

```
.
├── customs_rules_db/     # RAG가 사용하는 Vector DB (자동 생성)
├── venv/                   # 파이썬 가상환경 (Git 무시됨)
├── .gitignore              # Git이 무시할 파일/폴더 목록
├── main.py                 # FastAPI 서버 실행 파일
├── processing.py           # 핵심 AI 로직 및 데이터 처리 함수
├── requirements.txt        # 프로젝트 필요 라이브러리 목록
├── 대한민국송품장_검증규칙.xlsx  # 상업송장 검증 규칙 시트
├── 수입신고서_검증규칙.xlsx    # 수입신고서 검증 규칙 시트
├── 수출신고서_검증규칙.xlsx    # 수출신고서 검증 규칙 시트
└── 코드목록.xlsx             # 규칙 시트에서 사용하는 코드 목록
```
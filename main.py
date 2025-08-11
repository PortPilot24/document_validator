# main.py (상단)
from fastapi import FastAPI, File, UploadFile, HTTPException
import processing
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

# 프리픽스 환경에서 문서/경로 생성이 예쁘게 되게 하려면 root_path 지정(선택 권장)
app = FastAPI(
    title="세관 서류 자동 검토 API",
    root_path="/document-validator"   # Nginx 뒤에서 동작할 때 문서/링크가 올바르게 생성됨
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_methods=["*"],
  allow_headers=["*"],
)

# 2. 앱 시작 시 AI 모델, DB, 규칙 시트 로드 (한번만 실행됨)
try:
    vectordb, llm = processing.load_ai_resources()
    rules_df = processing.load_rule_sheet('수입신고서_검증규칙.xlsx')
    print("✅ AI 리소스 및 규칙 시트 로딩 완료")
except Exception as e:
    print(f"❌ 서버 시작 중 리소스 로딩 실패: {e}")
    vectordb, llm, rules_df = None, None, None

# 3. API 엔드포인트(Endpoint) 정의
#    프론트엔드 개발자가 접속할 '주소'를 만들어주는 부분입니다.
@app.post("/customs_review", summary="문서 이미지 검증 및 리포트 생성")
async def validate_document_api(file: UploadFile = File(...)):
    """
    수입신고서 이미지 파일을 받아 OCR, 유효성 검증, 지능형 리포팅을 수행합니다.
    """
    if not llm:
        raise HTTPException(status_code=500, detail="서버의 AI 리소스가 준비되지 않았습니다.")
        
    # 파일 확장자 확인
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="이미지 파일(PNG, JPG)만 업로드할 수 있습니다.")

    try:
        # 파일 내용을 바이트로 읽기
        image_bytes = await file.read()

        # 1단계: OCR & JSON 파싱
        parsed_json = processing.parse_image_to_json(image_bytes, llm)

        # 2단계: 규칙 기반 검증
        errors = processing.validate_document(parsed_json, rules_df)

        # 3단계: 지능형 리포팅
        reports = []
        if errors:
            for error in errors:
                report = processing.get_intelligent_error_report(error, vectordb, llm)
                reports.append(report)
        
        # 4. 최종 결과 응답
        return {
            "file_name": file.filename,
            "parsed_data": parsed_json,
            "validation_result": {
                "error_count": len(errors),
                "errors": errors,
                "reports": reports
            }
        }

    except ValueError as e: # JSON 파싱 실패 등
        raise HTTPException(status_code=400, detail=f"데이터 처리 오류: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {e}")

@app.get("/health")
def health():
    return {"ok": True}

# 5. (선택) 서버 직접 실행을 위한 코드
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

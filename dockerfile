# 베이스: 가벼운 Python 3.11
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# (선택) 시스템 패키지 — pandas 등 빌드/런타임에 필요한 것들
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 비루트 사용자 생성(보안)
RUN useradd -m appuser

# ChromaDB 디렉토리 생성 (쓰기 가능하도록)
# (방법 B) mkdir 후 chown (A와 B 중 하나만 쓰면 됨)
RUN mkdir -p /home/site/chromadb && chown -R appuser:appuser /home/site/chromadb

# 1) 의존성 먼저 복사/설치 (레이어 캐시 최적화)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) 애플리케이션 복사
COPY . .

# 규칙 시트 복사 (명시적)
COPY 대한민국송품장_검증규칙.xlsx /app/
COPY 수입신고서_검증규칙.xlsx /app/
COPY 수출신고서_검증규칙.xlsx /app/
COPY 코드목록.xlsx /app/

# 5) 비루트로 전환
USER appuser

# FastAPI 기본 포트
EXPOSE 8000

# 헬스체크 (main.py에 /health 있으면 이대로 사용)
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# 실행 커맨드
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

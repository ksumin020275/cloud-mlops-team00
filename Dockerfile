# 서울시 자전거 대여량 예측 서비스 (Streamlit + XGBoost)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# xgboost 실행에 필요한 라이브러리 (slim 이미지에는 없음)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) 패키지 먼저 설치 (코드만 바뀌면 이 단계는 캐시 재사용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) 앱 코드와 데이터 복사
COPY app.py train_model.py SeoulBikeData.csv ./

# 3) 이미지 안에서 모델 학습 → 컨테이너의 xgboost 버전과 모델 파일 버전이 항상 일치
RUN python train_model.py

EXPOSE 8501

# 서버가 살아 있는지 30초마다 확인
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

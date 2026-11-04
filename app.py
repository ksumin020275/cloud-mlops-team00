import datetime
import json
import logging
import os
import uuid

import joblib
import pandas as pd
import streamlit as st

from train_model import make_features

# ---- CloudWatch 로그 설정 (재학습용 로그 수집) ----
MODEL_VERSION = "xgb-20261014"          # 모델을 다시 학습할 때마다 바꾸기
LOG_GROUP = "/mlops/bike-rental"
LOG_STREAM = "streamlit-app"
AWS_REGION = "ap-northeast-2"


@st.cache_resource
def get_logger():
    logger = logging.getLogger("bike-predict")
    if logger.handlers:                                  # 이미 설정됐으면 중복 추가하지 않음
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(logging.StreamHandler())          # 터미널에도 같은 로그 출력
    try:
        import boto3
        import watchtower
        logger.addHandler(watchtower.CloudWatchLogHandler(
            log_group_name=LOG_GROUP,
            log_stream_name=LOG_STREAM,
            boto3_client=boto3.client("logs", region_name=AWS_REGION),
            create_log_group=False,
            send_interval=5))                            # 5초마다 CloudWatch로 전송
    except Exception as e:                               # AWS 연결이 안 돼도 서비스는 계속 동작
        logger.warning(f"CloudWatch 연결 실패, 터미널에만 기록합니다: {e}")
    return logger


def log_event(event, **data):
    get_logger().info(json.dumps({"event": event, "model_version": MODEL_VERSION,
                                  "ts": datetime.datetime.now().isoformat(timespec="seconds"), **data},
                                 ensure_ascii=False, default=str))

SEASON_BY_MONTH = {12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
                   6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
SEASON_KR = {'Spring': '봄', 'Summer': '여름', 'Autumn': '가을', 'Winter': '겨울'}
WEEKDAY_KR = ['월', '화', '수', '목', '금', '토', '일']


@st.cache_resource
def load_model():
    return joblib.load('xgboost_bike_model.pkl')


@st.cache_data
def load_actual():
    """SeoulBikeData.csv에서 날짜·시간별 실제 대여량을 불러옴 (운영한 날만)"""
    if not os.path.exists('SeoulBikeData.csv'):
        return None
    try:
        df = pd.read_csv('SeoulBikeData.csv', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv('SeoulBikeData.csv', encoding='cp949')
    df = df.iloc[:, [0, 1, 2, 13]]
    df.columns = ['Date', 'Count', 'Hour', 'Functioning']
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y').dt.date
    return df


def load_metrics():
    if os.path.exists('metrics.json'):
        with open('metrics.json', encoding='utf-8') as f:
            return json.load(f)
    return None


def predict_day(model, date, temp, humidity, wind, rain, is_holiday):
    """다음 날 0~23시, 24시간 각각의 대여량을 예측"""
    hours = list(range(24))
    season = SEASON_BY_MONTH[date.month]
    weekend = date.weekday() >= 5
    X = make_features(hours, [temp] * 24, [humidity] * 24, [wind] * 24, [rain] * 24,
                      [season] * 24, [is_holiday] * 24, [weekend] * 24)
    pred = model.predict(X).clip(min=0).round().astype(int)
    return pd.DataFrame({'시간': hours, '예상 대여량(대)': pred})


model = load_model()
metrics = load_metrics()


@st.cache_resource
def log_model_info():
    # [로그] 앱 시작 시 한 번: 지금 서비스 중인 모델 정보
    log_event("model_loaded", metrics=metrics)
    return True


log_model_info()
actual_data = load_actual()

st.title("🚲 서울시 자전거 대여량 예측 시스템")
st.write("날짜와 날씨를 입력하면 **0시~23시, 시간대별 예상 자전거 대여량**을 알려드립니다.")
st.caption("2017-12-01 ~ 2018-11-30 사이 날짜를 고르면 실제 대여량과 비교해 볼 수 있어요.")

st.sidebar.header("예측 조건 입력")
date = st.sidebar.date_input("예측할 날짜", datetime.date.today() + datetime.timedelta(days=1),
                             min_value=datetime.date(2017, 1, 1), max_value=datetime.date(2035, 12, 31))
season = SEASON_BY_MONTH[date.month]
st.sidebar.caption(f"{WEEKDAY_KR[date.weekday()]}요일 · {SEASON_KR[season]} (날짜로 자동 판단)")
temp = st.sidebar.number_input("기온 (℃)", value=20.0, step=0.5)
humidity = st.sidebar.slider("습도 (%)", 0, 100, 50)
wind = st.sidebar.number_input("풍속 (m/s)", value=1.5, min_value=0.0, step=0.1)
rain = st.sidebar.number_input("강수량 (mm)", value=0.0, min_value=0.0, step=0.5)
holiday_input = st.sidebar.radio("공휴일 여부", ['평일', '공휴일'])

if st.button("예상 대여량 확인하기"):
    result = predict_day(model, date, temp, humidity, wind, rain, holiday_input == '공휴일')

    # [로그] 예측 요청: 입력값과 24시간 예측값 (재학습 시 입력 X)
    request_id = str(uuid.uuid4())
    log_event("prediction", request_id=request_id,
              input={"date": date, "temp": temp, "humidity": humidity, "wind": wind, "rain": rain,
                     "holiday": int(holiday_input == '공휴일'), "weekend": int(date.weekday() >= 5),
                     "season": season},
              pred_total=int(result['예상 대여량(대)'].sum()),
              pred_hourly=result['예상 대여량(대)'].tolist())
    peak = result.loc[result['예상 대여량(대)'].idxmax()]

    # 데이터에 있는 날짜면 실제 대여량을 함께 보여줌
    day = None
    if actual_data is not None:
        day = actual_data[actual_data['Date'] == date].sort_values('Hour')
        if len(day) != 24:
            day = None

    c1, c2 = st.columns(2)
    c1.metric("하루 총 예상 대여량", f"{result['예상 대여량(대)'].sum():,}대")
    c2.metric("가장 붐비는 시간", f"{int(peak['시간'])}시", f"{int(peak['예상 대여량(대)']):,}대")

    st.subheader(f"{date} ({WEEKDAY_KR[date.weekday()]}) 시간대별 예상 대여량")

    if day is not None and (day['Functioning'] == 'No').all():
        st.warning("이 날은 따릉이를 운영하지 않은 날이라 실제 대여량이 0대입니다. 비교에서 제외하세요.")
        day = None

    if day is not None:
        result['실제 대여량(대)'] = day['Count'].values
        result['오차(대)'] = (result['예상 대여량(대)'] - result['실제 대여량(대)']).abs()
        result['판정'] = result['오차(대)'].apply(lambda e: '✅ 통과' if e <= 240 else '❌ 미통과')

        actual_total = int(result['실제 대여량(대)'].sum())
        pred_total = int(result['예상 대여량(대)'].sum())
        total_err = abs(pred_total - actual_total) / actual_total * 100

        # [로그] 실제값 피드백: 정답과 오차 (재학습 시 정답 y, 성능 저하 감지)
        log_event("actual", request_id=request_id, date=date,
                  actual_total=actual_total, pred_total=pred_total,
                  total_error_rate=round(total_err, 1),
                  mean_abs_error=round(float(result['오차(대)'].mean()), 1),
                  actual_hourly=result['실제 대여량(대)'].tolist(),
                  abs_error_hourly=result['오차(대)'].tolist())
        d1, d2, d3 = st.columns(3)
        d1.metric("실제 하루 총 대여량", f"{actual_total:,}대")
        d2.metric("하루 총량 오차율", f"{total_err:.1f}%", "통과 (±20% 이내)" if total_err <= 20 else "미통과",
                  delta_color="normal" if total_err <= 20 else "inverse")
        d3.metric("시간당 평균 오차", f"{result['오차(대)'].mean():.0f}대",
                  f"통과 {int((result['오차(대)'] <= 240).sum())}/24시간", delta_color="off")

        st.caption("📊 데이터에 기록된 날짜라 실제 대여량과 비교합니다. (판정 기준: 시간당 오차 240대 이내, 하루 총량 ±20% 이내)")
        st.line_chart(result.set_index('시간')[['예상 대여량(대)', '실제 대여량(대)']])
    else:
        st.bar_chart(result.set_index('시간')[['예상 대여량(대)']])

    with st.expander("시간대별 숫자로 보기"):
        st.dataframe(result, hide_index=True, use_container_width=True)

    if metrics:
        st.info(f"💡 최근 기간 데이터로 검증한 평균 오차(MAE)는 시간당 약 ±{metrics['mae']:.0f}대입니다. "
                f"(단순 평균 방식은 ±{metrics['baseline_mae']:.0f}대) 여유 수량을 준비하세요.")
    st.caption("※ 하루 동안 날씨가 같다고 가정한 예측입니다.")

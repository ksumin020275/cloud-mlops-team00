import datetime
import json
import os

import joblib
import pandas as pd
import streamlit as st

from train_model import make_features

SEASON_BY_MONTH = {12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
                   6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
SEASON_KR = {'Spring': '봄', 'Summer': '여름', 'Autumn': '가을', 'Winter': '겨울'}
WEEKDAY_KR = ['월', '화', '수', '목', '금', '토', '일']


@st.cache_resource
def load_model():
    return joblib.load('xgboost_bike_model.pkl')


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

st.title("🚲 서울시 자전거 대여량 예측 시스템")
st.write("내일의 날씨 예보를 입력하면 **0시~23시, 시간대별 예상 자전거 대여량**을 알려드립니다.")

st.sidebar.header("예측 조건 입력")
date = st.sidebar.date_input("예측할 날짜", datetime.date.today() + datetime.timedelta(days=1))
season = SEASON_BY_MONTH[date.month]
st.sidebar.caption(f"{WEEKDAY_KR[date.weekday()]}요일 · {SEASON_KR[season]} (날짜로 자동 판단)")
temp = st.sidebar.number_input("기온 (℃)", value=20.0, step=0.5)
humidity = st.sidebar.slider("습도 (%)", 0, 100, 50)
wind = st.sidebar.number_input("풍속 (m/s)", value=1.5, min_value=0.0, step=0.1)
rain = st.sidebar.number_input("강수량 (mm)", value=0.0, min_value=0.0, step=0.5)
holiday_input = st.sidebar.radio("공휴일 여부", ['평일', '공휴일'])

if st.button("예상 대여량 확인하기"):
    result = predict_day(model, date, temp, humidity, wind, rain, holiday_input == '공휴일')
    peak = result.loc[result['예상 대여량(대)'].idxmax()]

    c1, c2 = st.columns(2)
    c1.metric("하루 총 예상 대여량", f"{result['예상 대여량(대)'].sum():,}대")
    c2.metric("가장 붐비는 시간", f"{int(peak['시간'])}시", f"{int(peak['예상 대여량(대)']):,}대")

    st.subheader(f"{date} ({WEEKDAY_KR[date.weekday()]}) 시간대별 예상 대여량")
    st.bar_chart(result.set_index('시간'))
    with st.expander("시간대별 숫자로 보기"):
        st.dataframe(result, hide_index=True, use_container_width=True)

    if metrics:
        st.info(f"💡 최근 기간 데이터로 검증한 평균 오차(MAE)는 시간당 약 ±{metrics['mae']:.0f}대입니다. "
                f"(단순 평균 방식은 ±{metrics['baseline_mae']:.0f}대) 여유 수량을 준비하세요.")
    st.caption("※ 하루 동안 날씨가 같다고 가정한 예측입니다.")

import streamlit as st
import pandas as pd
import joblib

model = joblib.load('xgboost_bike_model.pkl')

st.title("🚲 서울시 자전거 대여량 예측 시스템")
st.write("날씨와 시간 정보를 입력하면 예상 자전거 수요를 알려드립니다.")

st.sidebar.header("예측 조건 입력")
hour = st.sidebar.slider("시간 (0~23시)", 0, 23, 18)
temp = st.sidebar.number_input("기온 (℃)", value=20.0)
humidity = st.sidebar.slider("습도 (%)", 0, 100, 50)
wind = st.sidebar.number_input("풍속 (m/s)", value=1.5)
rain = st.sidebar.number_input("강수량 (mm)", value=0.0)
season = st.sidebar.selectbox("계절", ['Spring', 'Summer', 'Autumn', 'Winter'])
holiday_input = st.sidebar.radio("휴일 여부", ['평일', '휴일'])

if st.button("예상 대여량 확인하기"):
    peak_hour = 1 if hour in [8, 9, 18, 19, 20, 21] else 0
    discomfort = 1.8 * temp - 0.55 * (1 - humidity/100) * (1.8 * temp - 26) + 32
    precip_yn = 1 if rain > 0 else 0
    holiday = 1 if holiday_input == '휴일' else 0

    input_df = pd.DataFrame([{
        'Hour': hour, 'Temperature': temp, 'Humidity': humidity, 'WindSpeed': wind, 'Rainfall': rain,
        'Peak_Hour': peak_hour, 'Discomfort_Index': discomfort, 'Precipitation_YN': precip_yn,
        'Holiday': holiday,
        'Seasons_Spring': 1 if season == 'Spring' else 0,
        'Seasons_Summer': 1 if season == 'Summer' else 0,
        'Seasons_Autumn': 1 if season == 'Autumn' else 0,
        'Seasons_Winter': 1 if season == 'Winter' else 0
    }])

    prediction = int(model.predict(input_df)[0])
    st.success(f"해당 조건에서의 예상 자전거 대여량은 **{prediction}대** 입니다!")
    st.info("💡 오차 범위(MAE)를 고려하여 ±160대 정도의 여유 수량을 준비하는 것을 권장합니다.")
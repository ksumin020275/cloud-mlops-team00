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
EXTRA_HOLIDAYS = {datetime.date(2018, 9, 26)}


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
    df = df.iloc[:, [0, 1, 2, 3, 9, 12, 13]]
    df.columns = ['Date', 'Count', 'Hour', 'Temp', 'Rain', 'Holiday', 'Functioning']
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y').dt.date
    # 원본 데이터에 평일로 잘못 표시된 공휴일 보정 (2018-09-26 추석 대체공휴일)
    df.loc[df['Date'].isin(EXTRA_HOLIDAYS), 'Holiday'] = 'Holiday'
    return df


@st.cache_data
def load_test_predictions():
    """train_model.py가 저장한 평가 기간(학습에 쓰지 않은 기간)의 예측값과 실제값"""
    if not os.path.exists('test_predictions.csv'):
        return None
    df = pd.read_csv('test_predictions.csv')
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df


def find_similar_day(date, temp, rain, is_holiday):
    """평가 기간(학습에 안 쓴 기간) 중 입력 조건과 가장 비슷한 날을 찾음
    - 쉬는 날(주말·공휴일)인지 평일인지가 같은 날 중에서
    - 하루 평균 기온과 시간당 강수량이 가장 가까운 날"""
    if test_pred is None or actual_data is None:
        return None
    test_dates = set(test_pred['Date'])
    days = (actual_data[actual_data['Date'].isin(test_dates)]
            .groupby('Date')
            .agg(temp=('Temp', 'mean'), rain=('Rain', 'mean'),
                 holiday=('Holiday', lambda h: (h == 'Holiday').any()))
            .reset_index())
    days['dayoff'] = days['Date'].map(lambda d: d.weekday() >= 5) | days['holiday']
    want_dayoff = date.weekday() >= 5 or is_holiday
    cand = days[days['dayoff'] == want_dayoff]
    if cand.empty:
        cand = days
    score = (cand['temp'] - temp).abs() / 5 + (cand['rain'] - rain).abs() * 2
    return cand.loc[score.idxmin()]


def show_compare_strip(caption, day_label, day_sub, actual_total, pred_total, pred_label, note):
    """예측 vs 실제를 가로 띠 한 줄로 보여줌: 구분 | 날짜 | 실제 | 예측 | 오차 판정"""
    err = abs(pred_total - actual_total) / actual_total * 100 if actual_total else 0
    ok = err <= TOTAL_TOL
    color = "#177233" if ok else "#c0392b"
    cell = "padding:12px 20px;border-left:1px solid #eee"
    small = "font-size:12px;color:#808495"
    st.markdown(f"""
<div style="display:flex;flex-wrap:wrap;align-items:center;border:1px solid #d6d6d9;border-radius:10px;overflow:hidden;margin:8px 0 4px">
  <div style="background:#f6f7f9;padding:16px 18px;white-space:nowrap">
    <div style="font-weight:700;font-size:15px">📊 예측 vs 실제</div><div style="{small}">{caption}</div></div>
  <div style="{cell}"><div style="{small}">{day_label}</div>
    <div style="font-size:17px;font-weight:600">{day_sub[0]}</div><div style="{small}">{day_sub[1]}</div></div>
  <div style="{cell}"><div style="{small}">실제</div><div style="font-size:26px">{actual_total:,}대</div></div>
  <div style="{cell}"><div style="{small}">{pred_label}</div><div style="font-size:26px">{pred_total:,}대</div></div>
  <div style="margin-left:auto;padding:0 22px;text-align:center;white-space:nowrap">
    <div style="font-size:30px;font-weight:700;color:{color}">{'✅' if ok else '❌'} {err:.0f}%</div>
    <div style="{small}">오차 · {'적중' if ok else '빗나감'} (±{TOTAL_TOL}%)</div></div>
</div>""", unsafe_allow_html=True)
    st.caption(note)


def weather_text(temp, rain):
    return f"{temp:.1f}℃ · " + ("비 없음" if rain < 0.05 else f"비 {rain:.1f}mm/시간")


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
test_pred = load_test_predictions()

HOUR_TOL = metrics['mae'] if metrics else 240   # 시간당 허용 오차 = 모델의 실제 MAE
TOTAL_TOL = 20                                   # 하루 총량 허용 오차율(%)

st.title("🚲 서울시 따릉이 대여량 예측 시스템")
st.write("날짜와 날씨를 입력하면 **0시~23시, 시간대별 예상 자전거 대여량**을 알려드립니다.")

st.sidebar.header("예측 조건 입력")
st.sidebar.caption("'🔮 예측하기' 탭에서 사용하는 입력값이에요.")
date = st.sidebar.date_input("예측할 날짜", datetime.date.today() + datetime.timedelta(days=1),
                             min_value=datetime.date(2017, 1, 1), max_value=datetime.date(2035, 12, 31))
season = SEASON_BY_MONTH[date.month]
st.sidebar.caption(f"{WEEKDAY_KR[date.weekday()]}요일 · {SEASON_KR[season]} (날짜로 자동 판단)")
temp = st.sidebar.number_input("기온 (℃)", value=20.0, step=0.5)
humidity = st.sidebar.slider("습도 (%)", 0, 100, 50)
wind = st.sidebar.number_input("풍속 (m/s)", value=1.5, min_value=0.0, step=0.1)
rain = st.sidebar.number_input("강수량 (mm)", value=0.0, min_value=0.0, step=0.5)
holiday_input = st.sidebar.radio("공휴일 여부", ['평일', '공휴일'])

tab_predict, tab_compare = st.tabs(["🔮 예측하기", "📊 예측 vs 실제 비교"])

# =========================== 탭 1: 예측하기 ===========================
with tab_predict:
    st.caption("2017-12-01 ~ 2018-11-30 사이 날짜를 고르면 실제 대여량과 비교해 볼 수 있어요.")
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

        if day is not None and (day['Functioning'] == 'No').all():
            st.warning("이 날은 따릉이를 운영하지 않은 날이라 실제 대여량이 0대입니다. 비교에서 제외하세요.")
            day = None

        # ---- 예측 vs 실제 비교 띠 ----
        in_test = test_pred is not None and date in set(test_pred['Date'])
        if day is not None:
            actual_total = int(day['Count'].sum())
            result['실제 대여량(대)'] = day['Count'].values
            result['오차(대)'] = (result['예상 대여량(대)'] - result['실제 대여량(대)']).abs()
            result['판정'] = result['오차(대)'].apply(lambda e: '✅' if e <= HOUR_TOL else '❌')
            day_sub = (f"{date} ({WEEKDAY_KR[date.weekday()]})",
                       weather_text(day['Temp'].mean(), day['Rain'].mean()))
            if in_test:
                # 학습에 안 쓴 기간 → 그날 실제 날씨로 예측했던 값과 비교 (정직한 비교)
                pred_total = int(test_pred[test_pred['Date'] == date]['Predicted'].sum())
                show_compare_strip("선택한 날짜의 실제 기록", "선택한 날짜", day_sub,
                                   actual_total, pred_total, "당시 예측",
                                   "※ 학습에 쓰지 않은 기간이라, 그날 실제 날씨로 예측했던 값과 비교했어요.")
            else:
                pred_total = int(result['예상 대여량(대)'].sum())
                show_compare_strip("선택한 날짜의 실제 기록", "선택한 날짜", day_sub,
                                   actual_total, pred_total, "입력값으로 예측",
                                   "※ 이 날짜는 모델 학습에 사용돼서 실제보다 잘 맞게 나올 수 있어요. "
                                   "정직한 비교는 2018-09-11 ~ 11-30 날짜로 해 보세요.")

            # [로그] 실제값 피드백: 정답과 오차 (재학습 시 정답 y, 성능 저하 감지)
            log_event("actual", request_id=request_id, date=date,
                      actual_total=actual_total, pred_total=pred_total,
                      total_error_rate=round(abs(pred_total - actual_total) / actual_total * 100, 1),
                      mean_abs_error=round(float(result['오차(대)'].mean()), 1),
                      actual_hourly=result['실제 대여량(대)'].tolist(),
                      abs_error_hourly=result['오차(대)'].tolist())
        else:
            # 미래 날짜 → 조건이 비슷했던 과거 날짜에서 모델이 얼마나 맞혔는지
            sim = find_similar_day(date, temp, rain, holiday_input == '공휴일')
            if sim is None:
                st.info("비교 데이터(test_predictions.csv)가 없어 비교할 수 없어요. train_model.py를 먼저 실행하세요.")
            else:
                sd = sim['Date']
                one = test_pred[test_pred['Date'] == sd]
                show_compare_strip("비슷했던 날로 검증", "비슷했던 날",
                                   (f"{sd} ({WEEKDAY_KR[sd.weekday()]})", weather_text(sim['temp'], sim['rain'])),
                                   int(one['Actual'].sum()), int(one['Predicted'].sum()), "당시 예측",
                                   "※ 실제값이 아직 없는 날짜라, 쉬는 날 여부가 같고 기온·강수량이 가장 비슷했던 날"
                                   "(학습에 쓰지 않은 기간)에서 모델이 얼마나 맞혔는지 보여줘요.")

        # ---- 예측 그래프 ----
        st.subheader(f"{date} ({WEEKDAY_KR[date.weekday()]}) 시간대별 예상 대여량")
        st.bar_chart(result.set_index('시간')[['예상 대여량(대)']])
        with st.expander("시간대별 숫자로 보기"):
            st.dataframe(result, hide_index=True, use_container_width=True)

        if metrics:
            st.info(f"💡 최근 기간 데이터로 검증한 평균 오차(MAE)는 시간당 약 ±{metrics['mae']:.0f}대입니다. "
                    f"(단순 평균 방식은 ±{metrics['baseline_mae']:.0f}대) 여유 수량을 준비하세요.")
        st.caption("※ 하루 동안 날씨가 같다고 가정한 예측입니다.")

# ======================= 탭 2: 예측 vs 실제 비교 =======================
with tab_compare:
    if test_pred is None:
        st.warning("비교 데이터(test_predictions.csv)가 없습니다. train_model.py를 먼저 실행하세요.")
    else:
        st.write("모델이 **학습할 때 보지 않은 기간**(가장 최근 기간)의 실제 대여량과 예측값을 비교합니다.")
        start, end = test_pred['Date'].min(), test_pred['Date'].max()
        st.caption(f"평가 기간: {start} ~ {end} · 이 기간의 데이터는 모델 학습에 사용하지 않았어요.")

        # ---- 1) 성적표 ----
        daily = test_pred.groupby('Date')[['Actual', 'Predicted', 'Baseline']].sum().reset_index()
        daily['오차율(%)'] = (daily['Predicted'] - daily['Actual']).abs() / daily['Actual'] * 100
        hit_rate = (daily['오차율(%)'] <= TOTAL_TOL).mean() * 100
        mae = (test_pred['Predicted'] - test_pred['Actual']).abs().mean()
        base_mae = (test_pred['Baseline'] - test_pred['Actual']).abs().mean()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("시간당 평균 오차 (MAE)", f"{mae:.0f}대")
        m2.metric("단순 평균 대비", f"{(1 - mae / base_mae) * 100:.0f}% 개선",
                  f"단순 평균 오차 {base_mae:.0f}대", delta_color="off")
        m3.metric(f"하루 총량 ±{TOTAL_TOL}% 적중", f"{hit_rate:.0f}%",
                  f"{int((daily['오차율(%)'] <= TOTAL_TOL).sum())}/{len(daily)}일", delta_color="off")
        m4.metric("하루 총량 평균 오차율", f"{daily['오차율(%)'].mean():.1f}%")

        # ---- 2) 기간 전체: 하루 총량 비교 ----
        st.subheader("날짜별 하루 총 대여량: 예측 vs 실제")
        chart = daily.rename(columns={'Actual': '실제', 'Predicted': '예측'}).set_index('Date')[['실제', '예측']]
        st.line_chart(chart)

        # ---- 3) 날짜 하나 골라서 시간대별 비교 ----
        st.subheader("날짜를 골라 시간대별로 비교하기")
        dates = list(daily['Date'])
        pick = st.selectbox("비교할 날짜", dates,
                            format_func=lambda d: f"{d} ({WEEKDAY_KR[d.weekday()]})")
        one = test_pred[test_pred['Date'] == pick].sort_values('Hour')
        one_view = pd.DataFrame({
            '시간': one['Hour'].values,
            '실제 대여량(대)': one['Actual'].values,
            '예측 대여량(대)': one['Predicted'].values,
        })
        one_view['오차(대)'] = (one_view['예측 대여량(대)'] - one_view['실제 대여량(대)']).abs()
        one_view['판정'] = one_view['오차(대)'].apply(lambda e: '✅' if e <= HOUR_TOL else '❌')

        row = daily[daily['Date'] == pick].iloc[0]
        s1, s2, s3 = st.columns(3)
        s1.metric("실제 하루 총량", f"{int(row['Actual']):,}대")
        s2.metric("예측 하루 총량", f"{int(row['Predicted']):,}대",
                  f"{int(row['Predicted'] - row['Actual']):+,}대", delta_color="off")
        s3.metric("오차율", f"{row['오차율(%)']:.1f}%",
                  "적중" if row['오차율(%)'] <= TOTAL_TOL else "빗나감",
                  delta_color="normal" if row['오차율(%)'] <= TOTAL_TOL else "inverse")
        st.line_chart(one_view.set_index('시간')[['실제 대여량(대)', '예측 대여량(대)']])
        with st.expander("시간대별 숫자로 보기"):
            st.dataframe(one_view, hide_index=True, use_container_width=True)

        # ---- 4) 가장 잘 맞힌 날 / 가장 많이 틀린 날 ----
        st.subheader("가장 잘 맞힌 날 · 가장 많이 틀린 날")
        rank = daily.assign(요일=daily['Date'].map(lambda d: WEEKDAY_KR[d.weekday()]))
        rank = rank.rename(columns={'Date': '날짜', 'Actual': '실제(대)', 'Predicted': '예측(대)'})
        cols = ['날짜', '요일', '실제(대)', '예측(대)', '오차율(%)']
        b1, b2 = st.columns(2)
        b1.caption("✅ 가장 잘 맞힌 날 TOP 3")
        b1.dataframe(rank.nsmallest(3, '오차율(%)')[cols].round(1), hide_index=True, use_container_width=True)
        b2.caption("❌ 가장 많이 틀린 날 TOP 3")
        b2.dataframe(rank.nlargest(3, '오차율(%)')[cols].round(1), hide_index=True, use_container_width=True)
        st.caption(f"※ 판정 기준: 하루 총량 ±{TOTAL_TOL}% 이내면 적중, 시간당 오차 {HOUR_TOL:.0f}대(모델 MAE) 이내면 ✅. "
                   "많이 틀린 날을 골라 위 '날짜를 골라 시간대별로 비교하기'에서 확인해 보세요.")

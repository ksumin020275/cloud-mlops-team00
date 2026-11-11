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
EXTRA_HOLIDAYS = {datetime.date(2018, 9, 26)}      # 원본 데이터에 평일로 잘못 표시된 공휴일 (추석 대체공휴일)
RECORD_START, RECORD_END = datetime.date(2017, 12, 1), datetime.date(2018, 11, 30)
TOTAL_TOL = 20                                     # 하루 총량 차이가 ±20% 이내면 '잘 맞힘'


# =============================== 데이터·모델 ===============================
@st.cache_resource
def load_model():
    return joblib.load('xgboost_bike_model.pkl')


@st.cache_data
def load_backtest():
    """train_model.py가 만든 과거 기록 확인용 예측
    (각 날짜가 속한 달을 빼고 학습한 모델의 예측 → 모델이 정답을 모르는 상태의 예측)"""
    if not os.path.exists('backtest_predictions.csv'):
        return None, None
    hourly = pd.read_csv('backtest_predictions.csv')
    hourly['Date'] = pd.to_datetime(hourly['Date']).dt.date
    hourly.loc[hourly['Date'].isin(EXTRA_HOLIDAYS), 'Holiday'] = 1
    daily = (hourly.groupby('Date')
             .agg(Actual=('Actual', 'sum'), Predicted=('Predicted', 'sum'), temp=('Temp', 'mean'),
                  humidity=('Humidity', 'mean'), rain=('Rain', 'sum'), holiday=('Holiday', 'max'),
                  weekend=('Weekend', 'max'))
             .reset_index())
    daily['dayoff'] = (daily['holiday'] == 1) | (daily['weekend'] == 1)
    daily['err'] = (daily['Predicted'] - daily['Actual']).abs() / daily['Actual'] * 100
    return hourly, daily


def predict_hours(model, date, temps, hums, winds, rains, holidays):
    """0~23시 예측. 날씨는 24개 값(시간대별) 리스트"""
    X = make_features(list(range(24)), temps, hums, winds, rains,
                      [SEASON_BY_MONTH[date.month]] * 24, holidays, [date.weekday() >= 5] * 24)
    return model.predict(X).clip(min=0).round().astype(int)


def date_text(d):
    return f"{d} ({WEEKDAY_KR[d.weekday()]})"


def weather_text(temp, rain_day):
    return f"{temp:.1f}℃ · " + ("비 없음" if rain_day < 0.1 else f"비 {rain_day:.1f}mm")


def diff_badge(actual, predicted):
    err = abs(predicted - actual) / actual * 100 if actual else 0
    ok = err <= TOTAL_TOL
    return err, ok, ("#177233" if ok else "#c0392b"), ("잘 맞힘" if ok else "많이 빗나감")


def show_strip(title, subtitle, cells, actual, predicted):
    """가로 띠 한 줄: [제목] | 칸들 | 차이 %"""
    err, ok, color, word = diff_badge(actual, predicted)
    small = "font-size:12px;opacity:0.65"
    cell_style = "padding:12px 20px;border-left:1px solid rgba(128,128,128,0.25)"
    cells_html = "".join(
        f'<div style="{cell_style}"><div style="{small}">{label}</div>'
        f'<div style="font-size:{size}px;font-weight:{600 if size < 20 else 400}">{value}</div>'
        f'<div style="{small}">{sub}</div></div>'
        for label, value, sub, size in cells)
    st.markdown(f"""
<div style="display:flex;flex-wrap:wrap;align-items:center;border:1px solid rgba(128,128,128,0.4);
            border-radius:10px;overflow:hidden;margin:8px 0 4px;color:inherit">
  <div style="background:rgba(128,128,128,0.10);padding:16px 18px;white-space:nowrap">
    <div style="font-weight:700;font-size:15px">{title}</div><div style="{small}">{subtitle}</div></div>
  {cells_html}
  <div style="margin-left:auto;padding:8px 22px;text-align:center;white-space:nowrap">
    <div style="font-size:30px;font-weight:700;color:{color}">{'✅' if ok else '❌'} {err:.0f}%</div>
    <div style="{small}">차이 · {word}</div></div>
</div>""", unsafe_allow_html=True)


model = load_model()
bt_hourly, bt_daily = load_backtest()
HOUR_MAE = float((bt_hourly['Predicted'] - bt_hourly['Actual']).abs().mean()) if bt_hourly is not None else None


@st.cache_resource
def log_model_info():
    # [로그] 앱 시작 시 한 번: 지금 서비스 중인 모델 정보
    log_event("model_loaded", backtest_hour_mae=round(HOUR_MAE, 1) if HOUR_MAE else None)
    return True


log_model_info()

# ================================= 화면 =================================
st.title("🚲 서울시 따릉이 대여량 예측 시스템")
st.write("날씨 예보를 넣으면 **하루 동안(0~23시) 따릉이를 몇 대 빌릴지** 예측해요. "
         "과거 실제 기록과 비교해 **예측이 얼마나 정확한지**도 확인할 수 있어요.")

tab_predict, tab_check, tab_score = st.tabs(["🔮 예측하기", "✅ 과거 기록으로 확인하기", "📊 전체 성적표"])

# ------------------------------ 탭 1: 예측하기 ------------------------------
with tab_predict:
    st.caption("예측할 날짜와 그날의 **날씨 예보**를 넣고 [예측하기]를 누르세요.")
    with st.form("predict_form"):
        r1 = st.columns([2, 2, 3])
        date = r1[0].date_input("📅 날짜", datetime.date.today() + datetime.timedelta(days=1),
                                min_value=RECORD_END + datetime.timedelta(days=1),
                                max_value=datetime.date(2035, 12, 31))
        is_holiday = r1[1].radio("공휴일인가요?", ["아니요", "네"], horizontal=True) == "네"
        r2 = st.columns(4)
        temp = r2[0].number_input("🌡️ 평균 기온 (℃)", value=15.0, step=0.5, min_value=-20.0, max_value=40.0)
        humidity = r2[1].number_input("💧 습도 (%)", value=50, step=5, min_value=0, max_value=100)
        wind = r2[2].number_input("🍃 풍속 (m/s)", value=1.5, step=0.1, min_value=0.0, max_value=10.0)
        rain_day = r2[3].number_input("☔ 하루 강수량 (mm)", value=0.0, step=1.0, min_value=0.0, max_value=200.0)
        submitted = st.form_submit_button("🔮 예측하기", type="primary", use_container_width=True)

    if submitted:
        pred = predict_hours(model, date, [temp] * 24, [humidity] * 24, [wind] * 24,
                             [rain_day / 24] * 24, [is_holiday] * 24)
        total, peak = int(pred.sum()), int(pred.argmax())

        # [로그] 예측 요청: 입력값과 24시간 예측값
        log_event("prediction", request_id=str(uuid.uuid4()), source="forecast",
                  input={"date": date, "temp": temp, "humidity": humidity, "wind": wind,
                         "rain_day": rain_day, "holiday": int(is_holiday),
                         "weekend": int(date.weekday() >= 5), "season": SEASON_BY_MONTH[date.month]},
                  pred_total=total, pred_hourly=pred.tolist())

        # 과거 기록에 없던 날씨인지 확인 (같은 달 기준)
        unusual = []
        if bt_daily is not None:
            same_month = bt_daily[bt_daily['Date'].map(lambda d: d.month) == date.month]
            t_lo, t_hi = same_month['temp'].min(), same_month['temp'].max()
            h_lo, h_hi = same_month['humidity'].min(), same_month['humidity'].max()
            r_hi = bt_daily['rain'].max()
            if not (t_lo - 5 <= temp <= t_hi + 5):
                unusual.append(f"{date.month}월 평균 기온은 과거 기록에서 {t_lo:.0f}~{t_hi:.0f}℃였는데, 입력값은 {temp:.0f}℃예요")
            if not (h_lo - 15 <= humidity <= h_hi + 15):
                unusual.append(f"{date.month}월 평균 습도는 과거 기록에서 {h_lo:.0f}~{h_hi:.0f}%였는데, 입력값은 {humidity}%예요")
            if rain_day > r_hi:
                unusual.append(f"하루 강수량은 과거 기록에서 최대 {r_hi:.0f}mm였는데, 입력값은 {rain_day:.0f}mm예요")

        st.markdown(f"#### {date_text(date)} 예측 결과")
        if unusual:
            st.warning("⚠️ **과거 기록(2017-12 ~ 2018-11)에 없던 날씨라 예측이 정확하지 않을 수 있어요.**\n\n"
                       + "\n".join(f"- {u}" for u in unusual)
                       + "\n\n모델은 과거 기록에서 배운 범위 안에서만 정확해요. 실제 날씨 예보 값을 넣어 보세요.")
        m = st.columns(3)
        m[0].metric("하루 동안 빌릴 자전거", f"{total:,}대")
        m[1].metric("가장 붐비는 시간", f"{peak}시", f"{pred[peak]:,}대", delta_color="off")
        m[2].metric("가장 한가한 시간", f"{int(pred.argmin())}시", f"{pred.min():,}대", delta_color="off")
        st.bar_chart(pd.DataFrame({'예측 대여량(대)': pred}, index=pd.Index(range(24), name='시간')))

        # 이 예측을 얼마나 믿을 수 있는지: 날씨가 비슷했던 과거 날의 실제 vs 예측
        if bt_daily is not None:
            st.markdown("#### 🎯 이 예측, 믿어도 될까요?")
            want_off = is_holiday or date.weekday() >= 5
            cand = bt_daily[bt_daily['dayoff'] == want_off].copy()
            cand['t_gap'] = (cand['temp'] - temp).abs()
            cand['h_gap'] = (cand['humidity'] - humidity).abs()
            cand['r_gap'] = (cand['rain'] - rain_day).abs()
            score = cand['t_gap'] / 3 + cand['h_gap'] / 15 + cand['r_gap'] / 10
            sim = cand.loc[score.idxmin()]
            really_similar = sim['t_gap'] <= 3 and sim['h_gap'] <= 15 and sim['r_gap'] <= 10
            if unusual or not really_similar:
                st.info(f"과거 기록에 이런 날씨({temp:.0f}℃ · 습도 {humidity}% · "
                        f"{'비 없음' if rain_day < 0.1 else f'비 {rain_day:.0f}mm'})와 비슷한 날이 없어서, "
                        "모델이 얼마나 맞힐지 비교해 드릴 수 없어요.")
            else:
                st.caption(f"{date}의 실제 대여량은 아직 알 수 없어요. 그래서 **비슷한 날씨였던 과거 하루**를 골라, "
                           "모델이 그날을 얼마나 맞혔는지 보여드려요.")
                show_strip("날씨가 비슷했던 날", "그날 모델이 얼마나 맞혔는지",
                           [("날짜", date_text(sim['Date']),
                             weather_text(sim['temp'], sim['rain']) + f" · 습도 {sim['humidity']:.0f}%", 17),
                            ("실제로 빌린 대수", f"{int(sim['Actual']):,}대", "", 26),
                            ("모델의 예측", f"{int(sim['Predicted']):,}대", "", 26)],
                           int(sim['Actual']), int(sim['Predicted']))
                st.caption(f"※ {'쉬는 날' if want_off else '평일'} 중 기온·습도·강수량이 가장 비슷했던 날이에요. "
                           f"차이가 ±{TOTAL_TOL}% 안이면 '잘 맞힘'이에요.")

        with st.expander("시간대별 숫자로 보기"):
            st.dataframe(pd.DataFrame({'시간': range(24), '예측 대여량(대)': pred}),
                         hide_index=True, use_container_width=True)
        st.caption("※ 하루 동안 날씨가 같다고 보고 예측해요. 날씨 예보가 정확할수록 예측도 정확해져요.")

# ------------------------ 탭 2: 과거 기록으로 확인하기 ------------------------
with tab_check:
    st.caption(f"{RECORD_START} ~ {RECORD_END} 중 날짜를 고르면, **그날 실제로 빌린 대수**와 "
               "**모델의 예측**을 비교해요. 날씨는 그날의 실제 날씨가 자동으로 들어가요.")
    if bt_hourly is None:
        st.warning("비교 데이터(backtest_predictions.csv)가 없어요. train_model.py를 먼저 실행하세요.")
    else:
        pick = st.date_input("📅 확인할 날짜", datetime.date(2018, 10, 17),
                             min_value=RECORD_START, max_value=RECORD_END, key="check_date")
        one = bt_hourly[bt_hourly['Date'] == pick].sort_values('Hour')
        if one.empty:
            st.warning("이 날은 따릉이를 운영하지 않아 기록이 없어요. 다른 날짜를 골라 주세요.")
        else:
            row = bt_daily[bt_daily['Date'] == pick].iloc[0]
            kind = "공휴일" if row['holiday'] else ("주말" if row['weekend'] else "평일")
            show_strip("예측 vs 실제", f"{kind} · {weather_text(row['temp'], row['rain'])}",
                       [("날짜", date_text(pick), f"습도 {row['humidity']:.0f}%", 17),
                        ("실제로 빌린 대수", f"{int(row['Actual']):,}대", "", 26),
                        ("모델의 예측", f"{int(row['Predicted']):,}대", "", 26)],
                       int(row['Actual']), int(row['Predicted']))
            st.line_chart(pd.DataFrame({'실제': one['Actual'].values, '예측': one['Predicted'].values},
                                       index=pd.Index(one['Hour'].values, name='시간')))
            with st.expander("시간대별 숫자로 보기"):
                st.dataframe(pd.DataFrame({'시간': one['Hour'].values, '실제(대)': one['Actual'].values,
                                           '예측(대)': one['Predicted'].values,
                                           '차이(대)': (one['Predicted'] - one['Actual']).values}),
                             hide_index=True, use_container_width=True)
            st.caption("※ 공정한 비교를 위해, 이 날짜가 속한 **달을 빼고 학습한 모델**이 예측한 값이에요. "
                       "(모델이 정답을 모르는 상태에서 한 예측)")

            # [로그] 예측 + 실제값 피드백 (같은 날짜는 한 번만 기록) → 재학습 데이터
            if st.session_state.get("logged_check") != pick:
                rid = str(uuid.uuid4())
                log_event("prediction", request_id=rid, source="backtest",
                          input={"date": pick, "temp": round(float(row['temp']), 1),
                                 "humidity": round(float(row['humidity']), 1),
                                 "rain_day": round(float(row['rain']), 1), "holiday": int(row['holiday']),
                                 "weekend": int(row['weekend']), "season": SEASON_BY_MONTH[pick.month]},
                          pred_total=int(row['Predicted']), pred_hourly=one['Predicted'].tolist())
                log_event("actual", request_id=rid, date=pick, actual_total=int(row['Actual']),
                          pred_total=int(row['Predicted']), total_error_rate=round(float(row['err']), 1),
                          actual_hourly=one['Actual'].tolist(),
                          abs_error_hourly=(one['Predicted'] - one['Actual']).abs().tolist())
                st.session_state["logged_check"] = pick

# ------------------------------ 탭 3: 전체 성적표 ------------------------------
with tab_score:
    if bt_daily is None:
        st.warning("비교 데이터(backtest_predictions.csv)가 없어요. train_model.py를 먼저 실행하세요.")
    else:
        n_ok = int((bt_daily['err'] <= TOTAL_TOL).sum())
        st.caption(f"{RECORD_START} ~ {RECORD_END}, 기록이 있는 {len(bt_daily)}일 전체를 모델이 얼마나 맞혔는지 보여줘요.")
        m = st.columns(3)
        m[0].metric("잘 맞힌 날", f"{n_ok / len(bt_daily) * 100:.0f}%", f"{n_ok}/{len(bt_daily)}일 (차이 ±{TOTAL_TOL}% 이내)",
                    delta_color="off")
        m[1].metric("하루 총량 차이 (중간값)", f"{bt_daily['err'].median():.0f}%")
        m[2].metric("시간당 평균 차이", f"±{HOUR_MAE:.0f}대")

        st.markdown("#### 날짜별 하루 대여량: 실제 vs 예측")
        st.line_chart(bt_daily.set_index('Date')[['Actual', 'Predicted']].rename(
            columns={'Actual': '실제', 'Predicted': '예측'}))

        st.markdown("#### 월별로 얼마나 맞혔나 (하루 총량 차이의 중간값, %)")
        monthly = bt_daily.assign(월=bt_daily['Date'].map(lambda d: f"{d.year}-{d.month:02d}"))
        st.bar_chart(monthly.groupby('월')['err'].median().rename('차이(%)'))

        st.markdown("#### 가장 잘 맞힌 날 · 가장 많이 빗나간 날")
        show = bt_daily.assign(날짜=bt_daily['Date'].map(date_text), 날씨=[weather_text(t, r) for t, r in
                                                                       zip(bt_daily['temp'], bt_daily['rain'])])
        show = show.rename(columns={'Actual': '실제(대)', 'Predicted': '예측(대)', 'err': '차이(%)'})
        cols = ['날짜', '날씨', '실제(대)', '예측(대)', '차이(%)']
        b = st.columns(2)
        b[0].caption("✅ 가장 잘 맞힌 날")
        b[0].dataframe(show.nsmallest(5, '차이(%)')[cols].round(1), hide_index=True, use_container_width=True)
        b[1].caption("❌ 가장 많이 빗나간 날")
        b[1].dataframe(show.nlargest(5, '차이(%)')[cols].round(1), hide_index=True, use_container_width=True)
        st.caption("※ 각 날짜는 그 날짜가 속한 달을 빼고 학습한 모델로 예측했어요. "
                   "비가 많이 와서 실제 대여량이 아주 적은 날은 몇천 대만 차이 나도 %가 크게 보여요.")

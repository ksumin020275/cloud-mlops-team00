import datetime
import json
import logging
import os
import uuid

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from train_daily_model import make_daily_features

# ---- CloudWatch 로그 설정 (재학습용 로그 수집) ----
MODEL_VERSION = "xgb-daily-20261111"   # 서울시 공식 2018~2026 데이터로 재학습          # 모델을 다시 학습할 때마다 바꾸기
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
WEEKDAY_KR = ['월', '화', '수', '목', '금', '토', '일']
TOTAL_TOL = 20                                     # 하루 총량 차이가 ±20% 이내면 '잘 맞힘'


# =============================== 데이터·모델 ===============================
@st.cache_resource
def load_model():
    return joblib.load('daily_model.pkl')


@st.cache_data
def load_data():
    bt = pd.read_csv('daily_backtest.csv')
    bt['Date'] = pd.to_datetime(bt['Date']).dt.date
    bt['DayOff'] = ((bt['Weekend'] == 1) | (bt['Holiday'] == 1)).astype(int)
    bt['err'] = (bt['Predicted'] - bt['Count']).abs() / bt['Count'] * 100
    prof = pd.read_csv('hour_profile.csv')
    metrics = json.load(open('daily_metrics.json', encoding='utf-8')) if os.path.exists('daily_metrics.json') else {}
    return bt, prof, metrics


model = load_model()
bt, profile, metrics = load_data()
RECORD_START, RECORD_END = bt['Date'].min(), bt['Date'].max()


def predict_total(date, temp, humidity, wind, rain_day, is_holiday):
    X = make_daily_features([date], [temp], [humidity], [wind], [rain_day], [int(is_holiday)],
                            [int(date.weekday() >= 5)])
    return int(round(float(np.expm1(model.predict(X)[0]))))


def split_hours(total, date, is_dayoff):
    """하루 총량을 0~23시로 나눔 (2018년 시간대별 기록에서 배운 계절·요일별 비율)"""
    share = profile[(profile['Season'] == SEASON_BY_MONTH[date.month]) & (profile['DayOff'] == int(is_dayoff))]
    return (share.sort_values('Hour')['Share'].values * total).round().astype(int)


def date_text(d):
    return f"{d} ({WEEKDAY_KR[d.weekday()]})"


def weather_text(temp, rain_day, humidity=None):
    t = f"{temp:.1f}℃ · " + ("비 없음" if rain_day < 0.1 else f"비 {rain_day:.1f}mm")
    return t + (f" · 습도 {humidity:.0f}%" if humidity is not None else "")


def show_strip(title, subtitle, cells, actual, predicted):
    """가로 띠 한 줄: [제목] | 칸들 | 차이 %"""
    err = abs(predicted - actual) / actual * 100 if actual else 0
    ok = err <= TOTAL_TOL
    color, word = ("#177233", "잘 맞힘") if ok else ("#c0392b", "많이 빗나감")
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


@st.cache_resource
def log_model_info():
    log_event("model_loaded", metrics=metrics)       # [로그] 앱 시작 시 한 번: 서비스 중인 모델 정보
    return True


log_model_info()

# ================================= 화면 =================================
st.title("🚲 서울시 따릉이 대여량 예측 시스템")
st.write("날씨 예보를 넣으면 **하루 동안(0~23시) 따릉이를 몇 대 빌릴지** 예측해요. "
         f"{RECORD_START.year}년부터의 실제 기록과 비교해 **예측이 얼마나 정확한지**도 확인할 수 있어요.")

tab_predict, tab_check, tab_score = st.tabs(["🔮 예측하기", "✅ 과거 기록으로 확인하기", "📊 전체 성적표"])

# ------------------------------ 탭 1: 예측하기 ------------------------------
with tab_predict:
    st.caption(f"예측할 날짜와 그날의 **날씨 예보**를 넣고 [예측하기]를 누르세요. "
               f"{RECORD_END} 이전 날짜는 '✅ 과거 기록으로 확인하기'에서 실제 기록과 비교할 수 있어요.")
    with st.form("predict_form"):
        r1 = st.columns([2, 2, 3])
        date = r1[0].date_input("📅 날짜", max(datetime.date.today() + datetime.timedelta(days=1),
                                               RECORD_END + datetime.timedelta(days=1)),
                                min_value=RECORD_END + datetime.timedelta(days=1),
                                max_value=datetime.date(2035, 12, 31))
        is_holiday = r1[1].radio("공휴일인가요?", ["아니요", "네"], horizontal=True) == "네"
        r2 = st.columns(4)
        temp = r2[0].number_input("🌡️ 평균 기온 (℃)", value=15.0, step=0.5, min_value=-20.0, max_value=40.0)
        humidity = r2[1].number_input("💧 습도 (%)", value=60, step=5, min_value=0, max_value=100)
        wind = r2[2].number_input("🍃 풍속 (m/s)", value=2.0, step=0.1, min_value=0.0, max_value=15.0)
        rain_day = r2[3].number_input("☔ 하루 강수량 (mm)", value=0.0, step=1.0, min_value=0.0, max_value=300.0)
        submitted = st.form_submit_button("🔮 예측하기", type="primary", use_container_width=True)

    if submitted:
        is_dayoff = is_holiday or date.weekday() >= 5
        total = predict_total(date, temp, humidity, wind, rain_day, is_holiday)
        hours = split_hours(total, date, is_dayoff)
        peak = int(hours.argmax())

        # [로그] 예측 요청: 입력값과 예측값
        log_event("prediction", request_id=str(uuid.uuid4()), source="forecast",
                  input={"date": date, "temp": temp, "humidity": humidity, "wind": wind, "rain_day": rain_day,
                         "holiday": int(is_holiday), "weekend": int(date.weekday() >= 5)},
                  pred_total=total, pred_hourly=hours.tolist())

        # 과거 기록에 없던 날씨인지 확인 (같은 달 기준)
        same_month = bt[bt['Date'].map(lambda d: d.month) == date.month]
        t_lo, t_hi = same_month['Temp'].min(), same_month['Temp'].max()
        h_lo, h_hi = same_month['Humidity'].min(), same_month['Humidity'].max()
        unusual = []
        if not (t_lo - 5 <= temp <= t_hi + 5):
            unusual.append(f"{date.month}월 평균 기온은 과거 기록에서 {t_lo:.0f}~{t_hi:.0f}℃였는데, 입력값은 {temp:.0f}℃예요")
        if not (h_lo - 15 <= humidity <= h_hi + 15):
            unusual.append(f"{date.month}월 평균 습도는 과거 기록에서 {h_lo:.0f}~{h_hi:.0f}%였는데, 입력값은 {humidity}%예요")
        if rain_day > bt['Rain'].max():
            unusual.append(f"하루 강수량은 과거 기록에서 최대 {bt['Rain'].max():.0f}mm였는데, 입력값은 {rain_day:.0f}mm예요")

        st.markdown(f"#### {date_text(date)} 예측 결과")
        if unusual:
            st.warning(f"⚠️ **과거 기록({RECORD_START.year}~{RECORD_END.year})에 없던 날씨라 예측이 정확하지 않을 수 있어요.**\n\n"
                       + "\n".join(f"- {u}" for u in unusual))
        m = st.columns(3)
        m[0].metric("하루 동안 빌릴 자전거", f"{total:,}대")
        m[1].metric("가장 붐비는 시간", f"{peak}시", f"{hours[peak]:,}대", delta_color="off")
        m[2].metric("가장 한가한 시간", f"{int(hours.argmin())}시", f"{hours.min():,}대", delta_color="off")
        st.bar_chart(pd.DataFrame({'예측 대여량(대)': hours}, index=pd.Index(range(24), name='시간')))
        st.caption("※ 하루 총량을 예측한 뒤, 계절·요일별로 시간대마다 빌리는 비율에 맞춰 0~23시로 나눴어요.")

        # 이 예측을 믿어도 될지: 날씨가 비슷했던 과거 날의 실제 vs 예측 (최근 날짜 우선)
        st.markdown("#### 🎯 이 예측, 믿어도 될까요?")
        cand = bt[bt['DayOff'] == int(is_dayoff)].copy()
        cand = cand[((cand['Temp'] - temp).abs() <= 3) & ((cand['Humidity'] - humidity).abs() <= 15)
                    & ((cand['Rain'] - rain_day).abs() <= max(5, rain_day * 0.5))]
        if unusual or cand.empty:
            st.info(f"과거 기록에 이런 날씨({weather_text(temp, rain_day, humidity)})와 비슷한 날이 없어서, "
                    "모델이 얼마나 맞힐지 비교해 드릴 수 없어요.")
        else:
            sim = cand.sort_values('Date').iloc[-1]          # 조건이 비슷한 날 중 가장 최근
            st.caption(f"{date}의 실제 대여량은 아직 알 수 없어요. 그래서 **날씨가 비슷했던 가장 최근 하루**를 골라, "
                       "모델이 그날을 얼마나 맞혔는지 보여드려요.")
            show_strip("날씨가 비슷했던 날", "그날 모델이 얼마나 맞혔는지",
                       [("날짜", date_text(sim['Date']), weather_text(sim['Temp'], sim['Rain'], sim['Humidity']), 17),
                        ("실제로 빌린 대수", f"{int(sim['Count']):,}대", "", 26),
                        ("모델의 예측", f"{int(sim['Predicted']):,}대", "", 26)],
                       int(sim['Count']), int(sim['Predicted']))
            st.caption(f"※ {'쉬는 날' if is_dayoff else '평일'} 중 기온 ±3℃, 습도 ±15%, 강수량이 비슷했던 날이에요. "
                       f"비슷한 날 {len(cand)}일 중 {(cand['err'] <= TOTAL_TOL).mean() * 100:.0f}%를 ±{TOTAL_TOL}% 안으로 맞혔어요.")

        with st.expander("시간대별 숫자로 보기"):
            st.dataframe(pd.DataFrame({'시간': range(24), '예측 대여량(대)': hours}),
                         hide_index=True, use_container_width=True)

# ------------------------ 탭 2: 과거 기록으로 확인하기 ------------------------
with tab_check:
    st.caption(f"{RECORD_START} ~ {RECORD_END} 중 날짜를 고르면, **그날 실제로 빌린 대수**(서울시 공식 기록)와 "
               "**모델의 예측**을 비교해요. 날씨는 그날의 실제 날씨가 자동으로 들어가요.")
    pick = st.date_input("📅 확인할 날짜", RECORD_END, min_value=RECORD_START, max_value=RECORD_END, key="check_date")
    row = bt[bt['Date'] == pick]
    if row.empty:
        st.warning("이 날은 태풍 등으로 운영을 거의 하지 않아 비교에서 뺐어요. 다른 날짜를 골라 주세요.")
    else:
        row = row.iloc[0]
        kind = "공휴일" if row['Holiday'] else ("주말" if row['Weekend'] else "평일")
        show_strip("예측 vs 실제", f"{kind} · {weather_text(row['Temp'], row['Rain'])}",
                   [("날짜", date_text(pick), f"습도 {row['Humidity']:.0f}%", 17),
                    ("실제로 빌린 대수", f"{int(row['Count']):,}대", "", 26),
                    ("모델의 예측", f"{int(row['Predicted']):,}대", "", 26)],
                   int(row['Count']), int(row['Predicted']))
        st.caption("※ 공정한 비교를 위해, 이 날짜가 속한 **달을 빼고 학습한 모델**이 예측한 값이에요. "
                   "(모델이 정답을 모르는 상태에서 한 예측)")

        st.markdown("#### 앞뒤 2주 흐름")
        near = bt[(bt['Date'] >= pick - datetime.timedelta(days=14)) & (bt['Date'] <= pick + datetime.timedelta(days=14))]
        st.line_chart(near.set_index('Date')[['Count', 'Predicted']].rename(columns={'Count': '실제', 'Predicted': '예측'}))

        # [로그] 예측 + 실제값 피드백 (같은 날짜는 한 번만 기록) → 재학습 데이터
        if st.session_state.get("logged_check") != pick:
            rid = str(uuid.uuid4())
            log_event("prediction", request_id=rid, source="backtest",
                      input={"date": pick, "temp": float(row['Temp']), "humidity": float(row['Humidity']),
                             "wind": float(row['Wind']), "rain_day": float(row['Rain']),
                             "holiday": int(row['Holiday']), "weekend": int(row['Weekend'])},
                      pred_total=int(row['Predicted']))
            log_event("actual", request_id=rid, date=pick, actual_total=int(row['Count']),
                      pred_total=int(row['Predicted']), total_error_rate=round(float(row['err']), 1))
            st.session_state["logged_check"] = pick

# ------------------------------ 탭 3: 전체 성적표 ------------------------------
with tab_score:
    st.caption(f"{RECORD_START} ~ {RECORD_END}, {len(bt):,}일 전체를 모델이 얼마나 맞혔는지 보여줘요.")
    n_ok = int((bt['err'] <= TOTAL_TOL).sum())
    m = st.columns(3)
    m[0].metric("잘 맞힌 날", f"{n_ok / len(bt) * 100:.0f}%", f"{n_ok:,}/{len(bt):,}일 (차이 ±{TOTAL_TOL}% 이내)",
                delta_color="off")
    m[1].metric("하루 총량 차이 (중간값)", f"{bt['err'].median():.1f}%")
    m[2].metric("하루 평균 차이", f"±{(bt['Predicted'] - bt['Count']).abs().mean():,.0f}대")

    if metrics.get('before') and metrics.get('after'):
        st.markdown(f"#### 🔁 재학습 전/후 비교 ({metrics['test_period']}, 학습에 쓰지 않은 기간)")
        b, a = metrics['before'], metrics['after']
        c = st.columns(2)
        c[0].metric("하루 총량 차이 (중간값)", f"{a['median_err']}%",
                    f"{a['median_err'] - b['median_err']:+.1f}%p (재학습 전 {b['median_err']}%)", delta_color="inverse")
        c[1].metric(f"잘 맞힌 날 (±{TOTAL_TOL}%)", f"{a['hit_rate']:.0f}%",
                    f"{a['hit_rate'] - b['hit_rate']:+.0f}%p (재학습 전 {b['hit_rate']:.0f}%)")
        st.caption("재학습 전: 2018년 일부 대여소 데이터로 학습한 모델 · "
                   f"재학습 후: 서울시 공식 {RECORD_START.year}~2025년 데이터로 다시 학습한 모델")

    st.markdown("#### 연도별 성적")
    yr = bt.assign(연도=bt['Date'].map(lambda d: d.year)).groupby('연도').agg(
        실제_평균=('Count', 'mean'), 예측_평균=('Predicted', 'mean'), 차이_중간값=('err', 'median'),
        잘맞힌날=('err', lambda e: (e <= TOTAL_TOL).mean() * 100))
    st.dataframe(yr.rename(columns={'실제_평균': '실제 하루 평균(대)', '예측_평균': '예측 하루 평균(대)',
                                    '차이_중간값': '차이 중간값(%)', '잘맞힌날': '잘 맞힌 날(%)'}).round(1),
                 use_container_width=True)

    st.markdown("#### 날짜별 하루 대여량: 실제 vs 예측")
    st.line_chart(bt.set_index('Date')[['Count', 'Predicted']].rename(columns={'Count': '실제', 'Predicted': '예측'}))

    st.markdown("#### 날씨별 정확도 (차이 중간값, %)")
    rain_kind = pd.cut(bt['Rain'], [-1, 0, 10, 1000], labels=['비 없음', '비 10mm 미만', '비 10mm 이상'])
    st.bar_chart(bt.groupby(rain_kind, observed=True)['err'].median().rename('차이(%)'))
    st.caption("※ 비가 많이 온 날은 실제 대여량이 아주 적어서, 몇천 대만 차이 나도 %가 크게 보여요.")

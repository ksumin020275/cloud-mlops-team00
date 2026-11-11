"""[재학습] 서울시 공식 하루 대여량(2018-01 ~ 2026-06) + 날씨로 '하루 대여량 모델'을 다시 학습
사용법: python train_daily_model.py
필요: daily_dataset.csv, SeoulBikeData.csv(시간대별 비율 계산용)
결과:
  daily_model.pkl          서비스에 쓰는 하루 대여량 모델 (전체 기간으로 학습)
  hour_profile.csv         0~23시 비율 (평일/쉬는 날 × 계절) → 하루 예측을 시간대별로 나눌 때 사용
  daily_backtest.csv       과거 기록 확인용 예측 (그 달을 빼고 학습한 모델의 예측)
  daily_metrics.json       재학습 전/후 성적 (2026년 1~6월, 학습에 쓰지 않은 기간)
"""
import json

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

SEASON_BY_MONTH = {12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
                   6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
TEST_START = pd.Timestamp('2026-01-01').date()     # 이 날짜부터는 학습에 쓰지 않고 시험용으로 남김
TOL = 20


def make_daily_features(dates, temp, humidity, wind, rain, holiday, weekend):
    """하루 단위 입력 변환 (학습과 app.py에서 똑같이 사용)"""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    X = pd.DataFrame({
        'Year': dates.dt.year,                       # 해마다 늘어난 이용 규모를 배우기 위한 입력
        'Month': dates.dt.month,
        'Temp': np.asarray(temp, dtype=float),
        'Humidity': np.asarray(humidity, dtype=float),
        'Wind': np.asarray(wind, dtype=float),
        'Rain': np.asarray(rain, dtype=float),
        'Weekend': np.asarray(weekend, dtype=int),
        'Holiday': np.asarray(holiday, dtype=int),
    })
    X['DayOff'] = ((X['Weekend'] == 1) | (X['Holiday'] == 1)).astype(int)
    X['RainYN'] = (X['Rain'] > 0).astype(int)
    X['HeavyRain'] = (X['Rain'] >= 10).astype(int)   # 로그 분석: 비가 많이 온 날 오차가 가장 컸음
    X['LogRain'] = np.log1p(X['Rain'])
    return X


def new_model():
    return XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5,
                        subsample=0.9, colsample_bytree=0.9, random_state=42)


def fit_predict(train, test):
    """대여량은 날씨 영향이 '비율'로 작용해서 log로 학습하고 되돌림"""
    m = new_model()
    m.fit(features(train), np.log1p(train['Count']))
    return m, np.expm1(m.predict(features(test))).clip(min=0).round().astype(int)


def features(d):
    return make_daily_features(d['Date'], d['Temp'], d['Humidity'], d['Wind'], d['Rain'],
                               d['Holiday'], d['Weekend'])


def score(actual, pred):
    err = np.abs(pred - actual) / actual * 100
    return {'median_err': round(float(np.median(err)), 1), 'hit_rate': round(float((err <= TOL).mean() * 100), 1),
            'mae': int(np.mean(np.abs(pred - actual)))}


def hour_profile():
    """기존 시간대별 데이터(SeoulBikeData.csv)에서 하루 중 시간대별 비율을 계산"""
    try:
        h = pd.read_csv('SeoulBikeData.csv', encoding='utf-8')
    except UnicodeDecodeError:
        h = pd.read_csv('SeoulBikeData.csv', encoding='cp949')
    h = h.iloc[:, [0, 1, 2, 11, 12, 13]]
    h.columns = ['Date', 'Count', 'Hour', 'Season', 'Holiday', 'Func']
    h = h[h['Func'] == 'Yes']
    h['Date'] = pd.to_datetime(h['Date'], format='%d/%m/%Y')
    h['DayOff'] = ((h['Date'].dt.dayofweek >= 5) | (h['Holiday'] == 'Holiday')).astype(int)
    p = h.groupby(['Season', 'DayOff', 'Hour'])['Count'].sum()
    p = (p / p.groupby(level=[0, 1]).transform('sum')).rename('Share').reset_index()
    return p


if __name__ == '__main__':
    df = pd.read_csv('daily_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df = df[df['Count'] > 1000].reset_index(drop=True)   # 태풍 등 운영 중단 수준의 날 제외

    train, test = df[df['Date'] < TEST_START], df[df['Date'] >= TEST_START]
    print(f"학습: {train['Date'].min()} ~ {train['Date'].max()} ({len(train)}일)")
    print(f"시험: {test['Date'].min()} ~ {test['Date'].max()} ({len(test)}일, 학습에 사용 안 함)")

    # 1) 재학습 후 성적 (2026년 1~6월)
    _, pred = fit_predict(train, test)
    after = score(test['Count'].values, pred)

    # 2) 재학습 전 성적 (같은 기간): 기존 시간대별 모델(xgboost_bike_model.pkl)로 예측
    before = None
    try:
        from train_model import make_features
        old = joblib.load('xgboost_bike_model.pkl')
        n = len(test)
        rep = lambda c: np.repeat(test[c].values, 24)
        Xh = make_features(np.tile(np.arange(24), n), rep('Temp'), rep('Humidity'), rep('Wind'), rep('Rain') / 24,
                           np.repeat(test['Date'].map(lambda d: SEASON_BY_MONTH[d.month]).values, 24),
                           rep('Holiday'), rep('Weekend'))
        old_pred = old.predict(Xh).clip(min=0).reshape(n, 24).sum(axis=1).round().astype(int)
        before = score(test['Count'].values, old_pred)
    except Exception as e:   # 파일이 없거나 못 읽어도 재학습 자체는 계속 진행
        print(f"(재학습 전 모델을 불러오지 못해 재학습 전 성적은 생략: {e})")

    print(f"\n[2026년 1~6월 성적]  하루 총량 차이 중간값 / 잘 맞힌 날(±{TOL}%)")
    if before:
        print(f"  재학습 전: {before['median_err']}% / {before['hit_rate']}%")
    print(f"  재학습 후: {after['median_err']}% / {after['hit_rate']}%")

    # 3) 과거 기록 확인용 예측: 한 달씩 빼고 학습 → 그 달 예측 (모든 날짜가 '모르는 상태의 예측')
    month = df['Date'].map(lambda d: (d.year, d.month))
    bt = np.zeros(len(df), dtype=int)
    for key in month.unique():
        hold = (month == key).values
        _, bt[hold] = fit_predict(df[~hold], df[hold])
    out = df[['Date', 'Count', 'Temp', 'Humidity', 'Wind', 'Rain', 'Holiday', 'Weekend']].copy()
    out['Predicted'] = bt
    out.to_csv('daily_backtest.csv', index=False)
    bt_score = score(df['Count'].values, bt)
    print(f"\n[과거 기록 확인용 예측] 2018~2026 전체: 차이 중간값 {bt_score['median_err']}%, "
          f"잘 맞힌 날 {bt_score['hit_rate']}%")

    # 4) 서비스용 최종 모델: 전체 기간으로 학습
    final = new_model()
    final.fit(features(df), np.log1p(df['Count']))
    joblib.dump(final, 'daily_model.pkl')
    hour_profile().to_csv('hour_profile.csv', index=False)
    with open('daily_metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'test_period': f"{test['Date'].min()} ~ {test['Date'].max()}",
                   'before': before, 'after': after, 'backtest_all': bt_score,
                   'data_period': f"{df['Date'].min()} ~ {df['Date'].max()}"}, f, ensure_ascii=False, indent=2)
    print("\ndaily_model.pkl, hour_profile.csv, daily_backtest.csv, daily_metrics.json 저장 완료!")

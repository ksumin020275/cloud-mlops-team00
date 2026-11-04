import json
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

SEASONS = ['Spring', 'Summer', 'Autumn', 'Winter']
PEAK_HOURS = [8, 9, 18, 19, 20, 21]


def make_features(hour, temp, humidity, wind, rain, season, holiday, weekend):
    """학습과 화면(app.py)에서 똑같이 쓰는 입력 변환 함수"""
    X = pd.DataFrame({
        'Hour': hour,
        'Temperature': temp,
        'Humidity': humidity,
        'WindSpeed': wind,
        'Rainfall': rain,
    })
    X['Peak_Hour'] = X['Hour'].isin(PEAK_HOURS).astype(int)
    X['Discomfort_Index'] = 1.8 * X['Temperature'] - 0.55 * (1 - X['Humidity'] / 100) * (1.8 * X['Temperature'] - 26) + 32
    X['Precipitation_YN'] = (X['Rainfall'] > 0).astype(int)
    X['Holiday'] = np.asarray(holiday, dtype=int)
    # [개선] 주말 여부 추가: 토·일은 출퇴근 수요가 없어서 시간대별 패턴이 완전히 다름
    X['Weekend'] = np.asarray(weekend, dtype=int)
    X['DayOff'] = ((X['Weekend'] == 1) | (X['Holiday'] == 1)).astype(int)
    season = pd.Series(np.asarray(season))
    for s in SEASONS:
        X['Seasons_' + s] = (season == s).astype(int).values
    return X


if __name__ == '__main__':
    # 1. 데이터 로드
    try:
        df = pd.read_csv('SeoulBikeData.csv', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv('SeoulBikeData.csv', encoding='cp949')
    df.columns = ['Date', 'Count', 'Hour', 'Temp', 'Humidity', 'Wind', 'Visibility', 'DewPoint',
                  'Solar', 'Rainfall', 'Snowfall', 'Seasons', 'Holiday', 'Functioning']

    # 2. 전처리: 운영한 날만, 날짜순 정렬
    df = df[df['Functioning'] == 'Yes'].copy()
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df = df.sort_values(['Date', 'Hour']).reset_index(drop=True)

    X = make_features(df['Hour'], df['Temp'], df['Humidity'], df['Wind'], df['Rainfall'],
                      df['Seasons'], (df['Holiday'] == 'Holiday'), (df['Date'].dt.dayofweek >= 5))
    y = df['Count']

    # 3. [수정] 시간 순서로 나누기: 앞 80% 기간으로 학습, 가장 최근 20% 기간으로 평가
    #    (기존 train_test_split은 무작위로 섞어서 '미래 데이터'가 학습에 들어가 점수가 부풀려졌음)
    cut = df['Date'].quantile(0.8)
    train_mask = df['Date'] < cut
    test_mask = ~train_mask
    print(f"학습 기간: {df['Date'][train_mask].min().date()} ~ {df['Date'][train_mask].max().date()}")
    print(f"평가 기간: {df['Date'][test_mask].min().date()} ~ {df['Date'][test_mask].max().date()}")

    # 4. 단순 비교 기준: 학습 기간의 '같은 계절·같은 시간대 평균 대여량'
    base_table = df[train_mask].groupby(['Seasons', 'Hour'])['Count'].mean()
    base_pred = [base_table.get((s, h), df[train_mask]['Count'].mean())
                 for s, h in zip(df['Seasons'][test_mask], df['Hour'][test_mask])]
    base_mae = mean_absolute_error(y[test_mask], base_pred)

    # 5. 모델 학습 및 평가
    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X[train_mask], y[train_mask])
    pred = model.predict(X[test_mask])
    model_mae = mean_absolute_error(y[test_mask], pred)
    model_r2 = r2_score(y[test_mask], pred)

    print(f"\n[단순 비교 기준] MAE: {base_mae:.1f}대")
    print(f"[XGBoost 모델]   MAE: {model_mae:.1f}대, R2: {model_r2:.3f}")
    print(f"→ 기준 대비 오차 {(1 - model_mae / base_mae) * 100:.1f}% 감소")

    # 6. 최종 모델은 전체 데이터로 다시 학습해서 저장 (평가 점수는 위에서 구한 값 사용)
    model.fit(X, y)
    joblib.dump(model, 'xgboost_bike_model.pkl')
    with open('metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'mae': round(float(model_mae), 1), 'r2': round(float(model_r2), 3),
                   'baseline_mae': round(float(base_mae), 1)}, f, ensure_ascii=False)
    print("\n모델(xgboost_bike_model.pkl)과 평가 결과(metrics.json) 저장 완료!")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

# 1. 데이터 로드
try:
    df = pd.read_csv('SeoulBikeData.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('SeoulBikeData.csv', encoding='cp949')

# 2. 전처리 및 피처 엔지니어링
df_prep = df[df['Functioning Day'] == 'Yes'].copy()

features = ['Hour', 'Temperature(캜)', 'Humidity(%)', 'Wind speed (m/s)', 'Rainfall(mm)', 'Seasons', 'Holiday']
target = 'Rented Bike Count'

X_raw = df_prep[features]
y = df_prep[target]

X = pd.DataFrame()
X['Hour'] = X_raw['Hour']
X['Temperature'] = X_raw['Temperature(캜)']
X['Humidity'] = X_raw['Humidity(%)']
X['WindSpeed'] = X_raw['Wind speed (m/s)']
X['Rainfall'] = X_raw['Rainfall(mm)']

X['Peak_Hour'] = X['Hour'].apply(lambda x: 1 if x in [8, 9, 18, 19, 20, 21] else 0)
X['Discomfort_Index'] = 1.8 * X['Temperature'] - 0.55 * (1 - X['Humidity']/100) * (1.8 * X['Temperature'] - 26) + 32
X['Precipitation_YN'] = (X['Rainfall'] > 0).astype(int)

X['Holiday'] = X_raw['Holiday'].map({'Holiday': 1, 'No Holiday': 0})
X['Seasons_Spring'] = (X_raw['Seasons'] == 'Spring').astype(int)
X['Seasons_Summer'] = (X_raw['Seasons'] == 'Summer').astype(int)
X['Seasons_Autumn'] = (X_raw['Seasons'] == 'Autumn').astype(int)
X['Seasons_Winter'] = (X_raw['Seasons'] == 'Winter').astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 모델 학습 및 저장
print("모델 학습을 시작합니다...")
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, 'xgboost_bike_model.pkl')
print("새로운 모델 파일 저장 완료!")
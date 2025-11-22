import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


# -----------------------------
# 1) 데이터 로딩
# -----------------------------
def load_air_weather_data(
    air_24_path, air_25_path, weather_24_path, weather_25_path
):
    air_24 = pd.read_csv(air_24_path, sep=",", encoding="cp949")
    air_25 = pd.read_csv(air_25_path, sep=",", encoding="cp949")
    weather_24 = pd.read_csv(weather_24_path, sep=",", encoding="cp949")
    weather_25 = pd.read_csv(weather_25_path, sep=",", encoding="cp949")

    return air_24, air_25, weather_24, weather_25


# -----------------------------
# 2) datetime 파싱 및 time 정렬
# -----------------------------
def create_time_variables(air_24, air_25, weather_24, weather_25):
    # 미세먼지 time 처리
    air_24["time"] = pd.to_datetime(
        air_24["측정일시"].astype(str).str[:10], format="%Y%m%d%H", errors="coerce"
    ) - pd.Timedelta(hours=1)

    air_25["time"] = pd.to_datetime(
        air_25["측정일시"].astype(str).str[:10], format="%Y%m%d%H", errors="coerce"
    ) - pd.Timedelta(hours=1)

    # 날씨 time 처리
    weather_24["time"] = pd.to_datetime(
        weather_24["일시"], format="%Y-%m-%d %H:%M", errors="coerce"
    )
    weather_25["time"] = pd.to_datetime(
        weather_25["일시"], format="%Y-%m-%d %H:%M", errors="coerce"
    )

    return air_24, air_25, weather_24, weather_25


# -----------------------------
# 3) 데이터 병합
# -----------------------------
def merge_air_weather(air_24, air_25, weather_24, weather_25):
    df_24 = pd.merge(air_24, weather_24, on="time", how="left")
    df_25 = pd.merge(air_25, weather_25, on="time", how="left")

    return df_24, df_25


# -----------------------------
# 4) 변수 선택 & 정렬
# -----------------------------
def select_features(df):
    selected = df[
        [
            "time",
            "SO2",
            "CO",
            "O3",
            "NO2",
            "PM10",
            "PM25",
            "기온(°C)",
            "강수량(mm)",
            "습도(%)",
            "풍속(m/s)",
            "풍향(16방위)",
        ]
    ].sort_values("time")
    return selected.reset_index(drop=True)


# -----------------------------
# 5) 결측치 처리
# -----------------------------
def handle_missing_values(df):
    # 중요 기상/오염 변수 결측 → 제거
    df = df.dropna(
        subset=["time", "SO2", "O3", "NO2", "기온(°C)", "습도(%)", "풍속(m/s)", "풍향(16방위)"]
    )
    # 강수량 결측 → 0 (비가 오지 않았다고 가정)
    df["강수량(mm)"] = df["강수량(mm)"].fillna(0)

    return df


# -----------------------------
# 6) 시계열 변수 생성
# -----------------------------
def create_time_features(df):
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["hour"] = df["time"].dt.hour
    return df


# -----------------------------
# 7) lag & target 변수 생성
# -----------------------------
def create_lag_target(df):
    # 전일 같은 시간 (24시간 lag)
    df["PM10_lag1"] = df["PM10"].shift(24)

    # t+1 시점 변수 생성
    df["PM10_1"] = df["PM10"].shift(-1)

    # lag, target 모두 존재하는 행만 사용
    df = df.dropna(subset=["PM10_lag1", "PM10_1"])

    return df


# -----------------------------
# 8) train/test 분리
# -----------------------------
def split_train_test(df_24, df_25, target="PM10_1"):
    train_x = df_24.drop(columns=[target, "time"])
    train_y = df_24[target]

    test_x = df_25.drop(columns=[target, "time"])
    test_y = df_25[target]

    return train_x, train_y, test_x, test_y


# -----------------------------
# 9) 전체 파이프라인 함수
# -----------------------------
def preprocess_pipeline(
    air_24_path, air_25_path, weather_24_path, weather_25_path
):
    # 1) load
    air_24, air_25, weather_24, weather_25 = load_air_weather_data(
        air_24_path, air_25_path, weather_24_path, weather_25_path
    )

    # 2) time 정리
    air_24, air_25, weather_24, weather_25 = create_time_variables(
        air_24, air_25, weather_24, weather_25
    )

    # 3) merge
    df_24, df_25 = merge_air_weather(air_24, air_25, weather_24, weather_25)

    # 4) 변수 선택
    df_24 = select_features(df_24)
    df_25 = select_features(df_25)

    # 5) 결측치 처리
    df_24 = handle_missing_values(df_24)
    df_25 = handle_missing_values(df_25)

    # 6) 시계열 변수 생성
    df_24 = create_time_features(df_24)
    df_25 = create_time_features(df_25)

    # 7) lag & target
    df_24 = create_lag_target(df_24)
    df_25 = create_lag_target(df_25)

    # 8) train/test 분리
    train_x, train_y, test_x, test_y = split_train_test(df_24, df_25)

    return train_x, train_y, test_x, test_y


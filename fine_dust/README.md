# 🌫️ Fine Dust Prediction Project
서울시 **미세먼지(PM10) 농도 예측 모델**을 구축한 프로젝트입니다.  
2024년·2025년 미세먼지 데이터 + 기상 데이터를 활용하여  
시계열 기반 예측 모델을 설계/전처리/학습하였습니다.

---

## 📌 프로젝트 개요

### 🎯 목표  
- 서울 지역의 **1시간 뒤 미세먼지(PM10)** 농도를 예측하는 머신러닝 모델 개발  
- 데이터 전처리 → Feature Engineering → 모델링 → 평가 전체 파이프라인 구축  

### 📁 사용 데이터  
| 데이터 | 설명 |
|-------|------|
| air_2024.csv | 2024년 미세먼지 측정 데이터 |
| air_2025.csv | 2025년 미세먼지 측정 데이터 |
| weather_2024.csv | 2024년 기상 데이터 |
| weather_2025.csv | 2025년 기상 데이터 |

---

## 🧹 1. 데이터 전처리 (data_preprocessing.py)

### 📌 주요 전처리 작업
- `측정일시`, `일시` → `time` 변수 변환 (Datetime)
- 미세먼지(1~24시) / 날씨(0~23시) 시간대 차이 보정
- 결측치 처리 (drop / 강수량은 0으로 대체)
- 필요한 변수만 선택하여 df_24 / df_25 구성
- Feature Engineering  
  - `month`, `day`, `hour` (파생 변수)  
  - `PM10_lag1` (24시간 전 PM10)  
  - `PM10_1` (1시간 후 PM10, 예측 target)

### 📁 출력 파일  
- `train_x.csv`  
- `train_y.csv`  
- `test_x.csv`  
- `test_y.csv`  

---

## 🤖 2. 모델링 (modeling.py)

### 🔍 사용 모델  
- Linear Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- XGBoost Regressor  

### 📈 평가 지표  
- **MSE** (Mean Squared Error)  
- **R² Score**  

### 💾 저장되는 모델 파일  
- `linear_model.pkl`  
- `RFR.pkl`  
- `gbr_model.pkl`  
- `xgbr_model.pkl`

### 📊 Feature Importance  
RandomForest, GradientBoosting, XGBoost 모델에 대해  
중요 변수 시각화를 지원합니다.

---

## 🗂 프로젝트 구조

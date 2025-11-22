# 📱 스마트폰 센서 기반 행동 인식 프로젝트

스마트폰 내 센서(Time-domain + Frequency-domain)를 활용해  
**6가지 행동(Activity)을 분류하고**,  
추가로 **정적/동적(is_dynamic)** 행동을 구분하는 모델을 제작하는 프로젝트입니다.

---

## 📌 프로젝트 목적

- 센서 기반 시계열 특징(feature) 데이터 이해
- EDA 기반 중요 변수 분석
- RandomForest 기반 행동 분류 모델 생성
- feature importance 기반 계층 구조 분석(sensor / agg / axis)
- 정적 vs 동적 행동 구분 모델 추가 파일 생성

---

## 📂 프로젝트 구조
```
sensor_activity/
├── data_preprocessing.py # Step1 - 데이터 로딩, 전처리, EDA
├── modeling.py # Step2 - 모델 학습, 중요도 분석, 병합 저장
├── README.md # 프로젝트 설명 문서
├── feature_importance_merged.pkl # 모델링 결과물
└── requirements.txt # 필요한 패키지
```

## 🚀 실행 방법

### 1) 전처리 실행 (EDA + 데이터 준비)

다음 명령을 실행하면 데이터 로딩, EDA, 변수 그룹 분석 등이 수행됩니다:

```bash
python data_preprocessing.py
```

### 2) 모델 학습 실행 (RandomForest etc.)

RandomForest 모델을 학습하고,
변수 중요도(feature importance) 병합 파일(.pkl)이 생성됩니다:

생성되는 파일:

feature_importance_merged.pkl

실행:
```bash
python modeling.py
```

### 3) requirements 설치
필요한 패키지를 설치하세요 .
```bash
pip install -r requirements.txt
```

📊 데이터 설명
📁 data01_train.csv / data01_test.csv

- 스마트폰 센서 기반 feature 561개

- 활동(Activity) 라벨 포함
(LAYING, SITTING, STANDING, WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS)

📁 features.csv

- 센서 변수의 계층 구조 포함
(sensor / agg / axis)

🧠 수행 내용 요약
✔ Step 1: 데이터 EDA

feature 그룹(sensors, agg, axis) 별 분포 확인

Activity class 단변량 분석

RandomForest 기반 중요 변수 상위 30개 시각화

중요도 상위 5개 변수 KDE plot 분석

중요도 하위 5개 변수 KDE plot 분석

✔ Step 2: 모델링

RandomForestClassifier 기반 기본 모델 생성

Activity(6-class) → is_dynamic(2-class) 이진 분류 모델 생성

두 중요도 결과를 feature.csv와 병합하여 pkl 저장

📦 결과물

| 파일명                           | 설명                                        |
| ----------------------------- | ----------------------------------------- |
| feature_importance_merged.pkl | Activity + is_dynamic 중요도 + 계층정보 merge 결과 |
| data_preprocessing.py         | 전처리 및 전체 EDA                              |
| modeling.py                   | RandomForest 기반 모델 및 중요도 저장               |
| requirements.txt              | 필요한 라이브러리 목록                              |

✨ 기술 스택

Python

Pandas, numpy

seaborn, matplotlib

scikit-learn

joblib

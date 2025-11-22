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

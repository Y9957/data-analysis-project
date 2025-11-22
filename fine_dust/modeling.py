import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


# -----------------------------
# 데이터 로딩
# -----------------------------
def load_train_test(train_x_path, train_y_path, test_x_path, test_y_path):
    train_x = pd.read_csv(train_x_path)
    train_y = pd.read_csv(train_y_path)
    test_x = pd.read_csv(test_x_path)
    test_y = pd.read_csv(test_y_path)
    return train_x, train_y, test_x, test_y


# -----------------------------
# 모델 학습 & 평가 함수
# -----------------------------
def train_and_evaluate(model, train_x, train_y, test_x, test_y):
    model.fit(train_x, train_y)

    y_pred = model.predict(test_x)

    mse = mean_squared_error(test_y, y_pred)
    r2 = r2_score(test_y, y_pred)

    return model, mse, r2, y_pred


# -----------------------------
# 모델 저장 함수
# -----------------------------
def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)


# -----------------------------
# Feature Importance (옵션)
# -----------------------------
def plot_feature_importance(model, feature_names, title="Feature Importances"):
    if not hasattr(model, "feature_importances_"):
        print("⚠ 이 모델은 feature_importances_ 속성이 없습니다.")
        return

    importances = model.feature_importances_
    plt.figure(figsize=(6, 8))
    plt.barh(feature_names, importances)
    plt.title(title)
    plt.show()


# -----------------------------
# 전체 파이프라인
# -----------------------------
def modeling_pipeline(train_x_path, train_y_path, test_x_path, test_y_path):
    # 1) 데이터 로딩
    train_x, train_y, test_x, test_y = load_train_test(
        train_x_path, train_y_path, test_x_path, test_y_path
    )

    results = {}

    # -----------------------------
    # ① Linear Regression
    # -----------------------------
    lr_model = LinearRegression()
    lr_model, mse_lr, r2_lr, pred_lr = train_and_evaluate(
        lr_model, train_x, train_y, test_x, test_y
    )
    save_model(lr_model, "linear_model.pkl")

    results["Linear Regression"] = {
        "MSE": mse_lr,
        "R2": r2_lr,
    }

    # -----------------------------
    # ② RandomForest Regressor
    # -----------------------------
    rfr_model = RandomForestRegressor()
    rfr_model, mse_rfr, r2_rfr, pred_rfr = train_and_evaluate(
        rfr_model, train_x, train_y, test_x, test_y
    )
    save_model(rfr_model, "RFR.pkl")

    results["Random Forest"] = {
        "MSE": mse_rfr,
        "R2": r2_rfr,
        "Feature Importance": rfr_model.feature_importances_,
    }

    # -----------------------------
    # ③ Gradient Boosting Regressor
    # -----------------------------
    gbr_model = GradientBoostingRegressor()
    gbr_model, mse_gbr, r2_gbr, pred_gbr = train_and_evaluate(
        gbr_model, train_x, train_y, test_x, test_y
    )
    save_model(gbr_model, "gbr_model.pkl")

    results["Gradient Boosting"] = {
        "MSE": mse_gbr,
        "R2": r2_gbr,
        "Feature Importance": gbr_model.feature_importances_,
    }

    # -----------------------------
    # ④ XGBoost (Self Choice Model)
    # -----------------------------
    xgb_model = XGBRegressor()
    xgb_model, mse_xgb, r2_xgb, pred_xgb = train_and_evaluate(
        xgb_model, train_x, train_y, test_x, test_y
    )
    save_model(xgb_model, "xgbr_model.pkl")

    results["XGBoost"] = {
        "MSE": mse_xgb,
        "R2": r2_xgb,
        "Feature Importance": xgb_model.feature_importances_,
    }

    return results


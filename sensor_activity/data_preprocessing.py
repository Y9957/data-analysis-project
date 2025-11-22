import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# 1. 데이터 로딩
# ---------------------------------------------------------
train_path = "./data01_train.csv"
test_path = "./data01_test.csv"
feature_path = "./features.csv"

data01_train = pd.read_csv(train_path)
data01_test = pd.read_csv(test_path)
features = pd.read_csv(feature_path)

# subject 삭제
if "subject" in data01_train.columns:
    data01_train.drop(columns=["subject"], inplace=True)
if "subject" in data01_test.columns:
    data01_test.drop(columns=["subject"], inplace=True)

# ---------------------------------------------------------
# 2. 기본 정보 출력 (필요 시 주석 해제)
# ---------------------------------------------------------
print("Train shape:", data01_train.shape)
print("Test shape :", data01_test.shape)
print("Features shape:", features.shape)

# ---------------------------------------------------------
# 3. Target = Activity 기반 Feature Importance 산출
# ---------------------------------------------------------
target = "Activity"

X = data01_train.drop(columns=[target])
y = data01_train[target]

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
feature_names = X.columns

fi_df_activity = pd.DataFrame({
    "feature_name": feature_names,
    "importance_activity": importances
}).sort_values("importance_activity", ascending=False)

# ---------------------------------------------------------
# 4. is_dynamic 변수 생성 후 Feature Importance
# ---------------------------------------------------------
data01_train["is_dynamic"] = data01_train["Activity"].isin(
    ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]
).astype(int)

X2 = data01_train.drop(columns=["Activity", "is_dynamic"])
y2 = data01_train["is_dynamic"]

rf2 = RandomForestClassifier(random_state=42)
rf2.fit(X2, y2)

importances2 = rf2.feature_importances_
feature_names2 = X2.columns

fi_df_dynamic = pd.DataFrame({
    "feature_name": feature_names2,
    "importance_dynamic": importances2
}).sort_values("importance_dynamic", ascending=False)

# ---------------------------------------------------------
# 5. Feature Importance 병합
# ---------------------------------------------------------
fi_merged = pd.merge(fi_df_activity, fi_df_dynamic, on="feature_name", how="outer")
fi_final = pd.merge(fi_merged, features, on="feature_name", how="left")

fi_final["importance_activity"] = fi_final["importance_activity"].fillna(0)
fi_final["importance_dynamic"] = fi_final["importance_dynamic"].fillna(0)
fi_final["axis"] = fi_final["axis"].fillna("None")

# ---------------------------------------------------------
# 6. 저장
# ---------------------------------------------------------
joblib.dump(fi_final, "feature_importance_merged.pkl")

print("✅ feature_importance_merged.pkl 저장 완료")

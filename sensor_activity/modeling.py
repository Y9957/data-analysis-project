import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# 데이터 로딩
# ---------------------------------------------------------
train_path = "./data01_train.csv"
test_path = "./data01_test.csv"

data01_train = pd.read_csv(train_path)
data01_test = pd.read_csv(test_path)

# subject 제거
if "subject" in data01_train.columns:
    data01_train.drop(columns=["subject"], inplace=True)
if "subject" in data01_test.columns:
    data01_test.drop(columns=["subject"], inplace=True)

# ---------------------------------------------------------
# 1. 6-Class 행동 분류 모델
# ---------------------------------------------------------
target = "Activity"
X = data01_train.drop(columns=[target])
y = data01_train[target]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

pred = rf.predict(X_val)

print("\n===== 6-Class Activity Classification =====")
print("Accuracy :", accuracy_score(y_val, pred))
print(classification_report(y_val, pred))

# ---------------------------------------------------------
# 2. is_dynamic (Binary Classification)
# ---------------------------------------------------------
data01_train["is_dynamic"] = data01_train["Activity"].isin(
    ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]
).astype(int)

X2 = data01_train.drop(columns=["Activity", "is_dynamic"])
y2 = data01_train["is_dynamic"]

X_train2, X_val2, y_train2, y_val2 = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2
)

rf2 = RandomForestClassifier(random_state=42)
rf2.fit(X_train2, y_train2)

pred2 = rf2.predict(X_val2)

print("\n===== Binary Classification (is_dynamic) =====")
print("Accuracy :", accuracy_score(y_val2, pred2))
print(classification_report(y_val2, pred2))

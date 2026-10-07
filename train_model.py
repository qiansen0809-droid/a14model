# -*- coding: utf-8 -*-
"""
训练就业合格预测模型
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


# ==============================
# 1. 数据路径
# ==============================
DATA_PATH = "new data2.csv"   # ← 改成你的路径
TARGET = "label"              # ← 改成你的标签列名（1=合格,0=不合格）


# ==============================
# 2. 读取数据
# ==============================
df = pd.read_csv(DATA_PATH)

print("数据形状:", df.shape)

y = df[TARGET]
X = df.drop(columns=[TARGET])


# ==============================
# 3. 特征分类
# ==============================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

print("数值特征:", len(numeric_features))
print("类别特征:", len(categorical_features))


# ==============================
# 4. 预处理
# ==============================
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# ==============================
# 5. 模型
# ==============================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# ==============================
# 6. 训练
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)


# ==============================
# 7. 评估
# ==============================
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\n分类报告:")
print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))


# ==============================
# 8. 保存模型
# ==============================
joblib.dump(pipeline, "best_model.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("\n模型已保存")
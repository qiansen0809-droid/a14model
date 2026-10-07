# -*- coding: utf-8 -*-
"""
A14 赛题：适配“新的整合后 CSV 数据集”的训练脚本

适配说明：
1. 不再读取旧版 zip + 5 张固定表，而是直接读取整合后的 integrated_student_dataset.csv
2. 默认使用“毕业去向”构造二分类风险标签：
   - 低风险：已落实就业/升学/保研/留学/西部计划等明确去向
   - 高风险：暂不就业/求职中/拟参加公招考试/拟出国出境/拟应征入伍/不就业拟升学
3. 自动去掉明显标签泄漏字段（graduate_dest__*）
4. 保留原脚本的：CV 对比、Optuna 调参、Voting 集成、Permutation Importance、4 类行为聚类
5. 输出目录：脚本同目录下 rebuilt_model_outputs_newdata

安装依赖：
pip install pandas numpy scikit-learn lightgbm optuna openpyxl
"""

import os
import warnings
import traceback
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, cross_val_predict
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, VotingClassifier
)
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler as ClusterScaler

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[警告] 未安装 lightgbm，建议执行: pip install lightgbm")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("[警告] 未安装 optuna，建议执行: pip install optuna")

warnings.filterwarnings("ignore")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(CURRENT_DIR, "integrated_student_dataset.csv")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "rebuilt_model_outputs_newdata")
RANDOM_SEED = 42
N_OPTUNA_TRIALS = 40

# ---------------------------------------------------------
# 标签策略
# graduate_dest__BYQXMC__last 来自“毕业去向名称”
# ---------------------------------------------------------
TARGET_STRATEGY = "graduate_risk"   # 可选: graduate_risk / status_change_risk

LOW_RISK_DEST = {
    "签就业协议形式就业",
    "签劳动合同形式就业",
    "保研",
    "考研已录取",
    "境外留学",
    "科研助理、管理助理",
    "第二学士学位",
    "其他录用形式就业",
    "西部计划",
    "地方选调生",
    "研究生",
    "应征义务兵",
    "创立公司",
}

HIGH_RISK_DEST = {
    "暂不就业",
    "求职中",
    "拟参加公招考试",
    "拟出国出境",
    "拟应征入伍",
    "不就业拟升学",
    "自由职业",
}


# =========================
# 工具函数
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def print_step(msg):
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)

def best_f1_threshold(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    if len(thr) == 0:
        return 0.5
    return float(thr[np.argmax(f1[:-1])])

def safe_to_datetime_features(df, col):
    if col not in df.columns:
        return df
    x = pd.to_datetime(df[col], errors="coerce")
    df[f"{col}__year"] = x.dt.year
    df[f"{col}__month"] = x.dt.month
    df[f"{col}__dayofweek"] = x.dt.dayofweek
    return df

def read_csv_auto(path):
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


# =========================
# 读取数据
# =========================
def load_integrated_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"未找到整合后的数据集：{csv_path}\n"
            f"请把 integrated_student_dataset.csv 放到脚本同目录。"
        )
    df = read_csv_auto(csv_path)
    return df


# =========================
# 标签构造
# =========================
def add_risk_label(df):
    df = df.copy()

    if TARGET_STRATEGY == "graduate_risk":
        label_col = "graduate_dest__BYQXMC__last"
        if label_col not in df.columns:
            raise KeyError(
                f"当前策略需要列 {label_col}，但数据中不存在。"
            )

        df = df[df[label_col].isin(LOW_RISK_DEST | HIGH_RISK_DEST)].copy()
        df["risk_label"] = df[label_col].isin(HIGH_RISK_DEST).astype(int)
        return df, label_col

    elif TARGET_STRATEGY == "status_change_risk":
        # 备选：学籍异动次数>0 视为高风险
        label_col = "student_status_change__YDLBDM__count"
        if label_col not in df.columns:
            raise KeyError(
                f"当前策略需要列 {label_col}，但数据中不存在。"
            )
        x = pd.to_numeric(df[label_col], errors="coerce").fillna(0)
        df["risk_label"] = (x > 0).astype(int)
        return df, label_col

    else:
        raise ValueError("未知 TARGET_STRATEGY")


# =========================
# 特征清洗与筛选
# =========================
def clean_and_select_features(df, label_source_col):
    df = df.copy()

    # 去掉明显标签泄漏字段
    leak_prefixes = [
        "graduate_dest__",   # 如果用毕业去向构造标签，这整块都是泄漏
    ]
    if TARGET_STRATEGY == "status_change_risk":
        leak_prefixes.append("student_status_change__")

    leak_cols = [c for c in df.columns if any(c.startswith(p) for p in leak_prefixes)]

    # 去掉无意义或明显脏字段
    bad_keywords = ["Unnamed:", "__sum", "__min", "__max"]
    drop_bad = []
    for c in df.columns:
        if c in ["sid", "student_id_raw", "risk_label", label_source_col]:
            continue
        if "Unnamed:" in c:
            drop_bad.append(c)

    # 先保留一批高价值原始列
    preferred_cols = [
        "gender",
        "ethnicity_name",
        "political_status_name",
        "college_name",
        "major_name",

        "internet_usage__SWLJSC__sum",
        "internet_usage__SWLJSC__mean",
        "internet_usage__SWLJSC__count",

        "discussion_record__creator_post_count",
        "discussion_record__reply_count",

        "running_punch__WEEKNUM__count",
        "running_punch__PUNCH_DAY__nunique",

        "daily_exercise__DKCS__sum",
        "daily_exercise__DKCS__mean",
        "daily_exercise__DKCS__count",

        "door_access__event_count",
        "door_access__active_days",

        "club_activity__event_count",
        "club_activity__active_days",

        "online_learning_overall__BFB__mean",
        "online_learning_overall__BFB__max",
        "online_learning_overall__BFB__count",

        "assignment_submission__SCORE__mean",
        "assignment_submission__SCORE__count",
        "assignment_submission__FULLMARKS__mean",
        "assignment_submission__STATUS__nunique",

        "exam_submission__SCORE__mean",
        "exam_submission__SCORE__count",
        "exam_submission__FULLMARKS__mean",
        "exam_submission__STATUS__nunique",

        "student_score__KCCJ__mean",
        "student_score__KCCJ__max",
        "student_score__KCCJ__count",
        "student_score__JDCJ__mean",
        "student_score__JDCJ__max",
        "student_score__JDCJ__min",
        "student_score__XF__sum",
        "student_score__XF__mean",
        "student_score__KCH__nunique",
        "student_score__TDKCH__nunique",

        "student_signin__ACTIVE_ID__count",
        "student_signin__ACTIVE_STATE__mean",
        "student_signin__ATTEND_TIME__count",
        "student_signin__INSERT_TIME__nunique",

        "scholarship_award__FFJE__sum",
        "scholarship_award__FFJE__max",
        "scholarship_award__JXJMC__nunique",
        "scholarship_award__PDDJ__nunique",

        "cet_score__KS_CJ__mean",
        "cet_score__KS_CJ__max",
        "cet_score__KS_CJ__count",

        "comprehensive_eval__ZCJ__mean",
        "comprehensive_eval__ZCJ__max",
        "comprehensive_eval__ZYNJPM__mean",
        "comprehensive_eval__BJPM__mean",
        "comprehensive_eval__BJRS__mean",

        "physical_test__ZF__mean",
        "physical_test__BMI__nunique",
        "physical_test__FHL__mean",
        "physical_test__WS__mean",
        "physical_test__LDTY__mean",
        "physical_test__ZWTQQ__mean",

        "physical_course__KC__nunique",
        "physical_course__SKSJ__nunique",
        "physical_course__ZKC__nunique",

        "discipline_competition__HJJB__nunique",
        "discipline_competition__HJDJ__nunique",
    ]

    keep_cols = [c for c in preferred_cols if c in df.columns]

    # 保留低缺失的基础列
    base_cols = [c for c in ["sid", "student_id_raw"] if c in df.columns]
    keep_cols = base_cols + keep_cols + ["risk_label"]

    model_df = df[keep_cols].copy()

    # 数值化
    for c in model_df.columns:
        if c in ["sid", "student_id_raw", "gender", "ethnicity_name", "political_status_name", "college_name", "major_name", "risk_label"]:
            continue
        model_df[c] = pd.to_numeric(model_df[c], errors="coerce")

    # 类别列
    for c in ["gender", "ethnicity_name", "political_status_name", "college_name", "major_name"]:
        if c in model_df.columns:
            model_df[c] = model_df[c].astype("object")

    return model_df


# =========================
# 特征工程（新数据集版）
# =========================
def add_feature_engineering(df):
    df = df.copy()

    # 统一可用列
    def col(name):
        return name if name in df.columns else None

    # 学习行为
    c_net = col("internet_usage__SWLJSC__sum")
    c_discuss = col("discussion_record__creator_post_count")
    c_reply = col("discussion_record__reply_count")
    c_online = col("online_learning_overall__BFB__mean")
    c_hw = col("assignment_submission__SCORE__mean")
    c_exam = col("exam_submission__SCORE__mean")
    c_course_score = col("student_score__KCCJ__mean")
    c_gpa = col("student_score__JDCJ__mean")

    # 参与行为
    c_run = col("running_punch__PUNCH_DAY__nunique")
    c_exercise = col("daily_exercise__DKCS__sum")
    c_door = col("door_access__active_days")
    c_club = col("club_activity__active_days")
    c_sign = col("student_signin__ACTIVE_ID__count")

    # 成就行为
    c_scholar = col("scholarship_award__FFJE__sum")
    c_cet = col("cet_score__KS_CJ__mean")
    c_comp = col("discipline_competition__HJDJ__nunique")
    c_eval = col("comprehensive_eval__ZCJ__mean")

    # 用 0 填临时值，生成后仍由预处理器做缺失补
    def s(c):
        if c is None:
            return 0
        return pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["study_engagement_score"] = (
        0.35 * s(c_net) +
        0.20 * s(c_discuss) +
        0.10 * s(c_reply) +
        0.20 * s(c_online) +
        0.15 * s(c_sign)
    )

    df["achievement_score"] = (
        0.35 * s(c_course_score) +
        0.25 * s(c_gpa) +
        0.20 * s(c_hw) +
        0.20 * s(c_exam)
    )

    df["activity_score"] = (
        0.35 * s(c_run) +
        0.30 * s(c_exercise) +
        0.20 * s(c_club) +
        0.15 * s(c_door)
    )

    df["development_score"] = (
        0.40 * s(c_scholar) +
        0.30 * s(c_cet) +
        0.15 * s(c_comp) +
        0.15 * s(c_eval)
    )

    df["study_achievement_gap"] = df["study_engagement_score"] - df["achievement_score"]
    df["engagement_activity_balance"] = df["study_engagement_score"] + df["activity_score"]

    # 分位数排名
    for c in ["study_engagement_score", "achievement_score", "activity_score", "development_score"]:
        df[f"rank_{c}"] = pd.Series(df[c]).rank(pct=True)

    return df


# =========================
# 预处理器
# =========================
def build_preprocessors(X):
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre_linear = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ])

    pre_tree = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ])

    return pre_linear, pre_tree, num_cols, cat_cols


# =========================
# 模型
# =========================
def build_base_models(pre_linear, pre_tree):
    models = {
        "LogisticRegression": Pipeline([
            ("pre", pre_linear),
            ("clf", LogisticRegression(
                max_iter=4000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=RANDOM_SEED
            ))
        ]),
        "RandomForest": Pipeline([
            ("pre", pre_tree),
            ("clf", RandomForestClassifier(
                n_estimators=400,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1
            ))
        ]),
        "ExtraTrees": Pipeline([
            ("pre", pre_tree),
            ("clf", ExtraTreesClassifier(
                n_estimators=400,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1
            ))
        ]),
        "GradientBoosting": Pipeline([
            ("pre", pre_tree),
            ("clf", GradientBoostingClassifier(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_SEED
            ))
        ]),
    }

    if HAS_LGB:
        models["LightGBM"] = Pipeline([
            ("pre", pre_tree),
            ("clf", lgb.LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                num_leaves=15,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.2,
                reg_lambda=0.2,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=-1
            ))
        ])
    return models


def optuna_tune(model_name, X_tr, y_tr, pre_tree, pre_linear):
    if not HAS_OPTUNA:
        return None, None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    if model_name == "LogisticRegression":
        def obj(trial):
            C = trial.suggest_float("C", 0.01, 10.0, log=True)
            model = Pipeline([
                ("pre", pre_linear),
                ("clf", LogisticRegression(
                    max_iter=4000,
                    C=C,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=RANDOM_SEED
                ))
            ])
            return cross_validate(model, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=1)["test_score"].mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(obj, n_trials=25)
        bp = study.best_params
        tuned = Pipeline([
            ("pre", pre_linear),
            ("clf", LogisticRegression(
                max_iter=4000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=RANDOM_SEED,
                **bp
            ))
        ])
        return tuned, bp

    if model_name == "LightGBM" and HAS_LGB:
        def obj(trial):
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 200, 700),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 6),
                num_leaves=trial.suggest_int("num_leaves", 8, 31),
                min_child_samples=trial.suggest_int("min_child_samples", 10, 60),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 0.01, 3.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 0.01, 3.0, log=True),
            )
            model = Pipeline([
                ("pre", pre_tree),
                ("clf", lgb.LGBMClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                    verbose=-1,
                    **params
                ))
            ])
            return cross_validate(model, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=1)["test_score"].mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(obj, n_trials=N_OPTUNA_TRIALS)
        bp = study.best_params
        tuned = Pipeline([
            ("pre", pre_tree),
            ("clf", lgb.LGBMClassifier(
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbose=-1,
                **bp
            ))
        ])
        return tuned, bp

    return None, None


def build_voting(top_models):
    return VotingClassifier(
        estimators=list(top_models.items()),
        voting="soft",
        n_jobs=1
    )


def evaluate_models(models, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for name, model in models.items():
        scores = cross_validate(
            model, X, y, cv=cv,
            scoring=["roc_auc", "accuracy", "precision", "recall", "f1"],
            n_jobs=1
        )
        rows.append({
            "model": name,
            "cv_auc_mean": round(scores["test_roc_auc"].mean(), 6),
            "cv_auc_std": round(scores["test_roc_auc"].std(), 6),
            "cv_accuracy_mean": round(scores["test_accuracy"].mean(), 6),
            "cv_precision_mean": round(scores["test_precision"].mean(), 6),
            "cv_recall_mean": round(scores["test_recall"].mean(), 6),
            "cv_f1_mean": round(scores["test_f1"].mean(), 6),
            "fold_auc_scores": ",".join([f"{x:.4f}" for x in scores["test_roc_auc"]]),
        })
    return pd.DataFrame(rows).sort_values(
        by=["cv_auc_mean", "cv_f1_mean"],
        ascending=False
    ).reset_index(drop=True)


def fit_best_model_and_export(best_model_name, models_ordered, X, y, output_dir, pre_linear, pre_tree):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    best_tuned_model = None
    if HAS_OPTUNA:
        print(f"\n[Optuna] 调参：{best_model_name}")
        best_tuned_model, bp = optuna_tune(best_model_name, X_train, y_train, pre_tree, pre_linear)
        if bp:
            print(f"最优参数：{bp}")

    top3 = {n: m for n, m in list(models_ordered.items())[:3]}
    voting_model = build_voting(top3)

    def eval_candidate(name, model):
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        try:
            oof = cross_val_predict(model, X_train, y_train, cv=5, method="predict_proba", n_jobs=1)[:, 1]
            thr = best_f1_threshold(y_train, oof)
        except Exception:
            thr = 0.5
        pred = (prob >= thr).astype(int)

        return {
            "name": name,
            "model": model,
            "prob": prob,
            "pred": pred,
            "thr": thr,
            "auc": roc_auc_score(y_test, prob),
            "acc": accuracy_score(y_test, pred),
            "prec": precision_score(y_test, pred, zero_division=0),
            "rec": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
        }

    results = []
    results.append(eval_candidate(best_model_name, models_ordered[best_model_name]))
    if best_tuned_model is not None:
        results.append(eval_candidate(f"{best_model_name}_Optuna", best_tuned_model))
    results.append(eval_candidate("Voting_top3", voting_model))

    best = max(results, key=lambda r: r["auc"])

    metrics_df = pd.DataFrame([{
        "best_model": best["name"],
        "opt_threshold": round(best["thr"], 4),
        "holdout_auc": round(best["auc"], 6),
        "holdout_accuracy": round(best["acc"], 6),
        "holdout_precision": round(best["prec"], 6),
        "holdout_recall": round(best["rec"], 6),
        "holdout_f1": round(best["f1"], 6),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }])

    cand_df = pd.DataFrame([{
        "model": r["name"],
        "auc": round(r["auc"], 6),
        "accuracy": round(r["acc"], 6),
        "precision": round(r["prec"], 6),
        "recall": round(r["rec"], 6),
        "f1": round(r["f1"], 6),
        "opt_threshold": round(r["thr"], 4),
    } for r in results])

    perm = permutation_importance(
        best["model"], X_test, y_test,
        scoring="roc_auc", n_repeats=15,
        random_state=RANDOM_SEED, n_jobs=1
    )
    imp_df = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    pred_df = X_test.copy()
    pred_df["y_true"] = y_test.values
    pred_df["risk_probability"] = best["prob"]
    pred_df["y_pred"] = best["pred"]

    metrics_df.to_csv(os.path.join(output_dir, "rebuild_metrics.csv"), index=False, encoding="utf-8-sig")
    cand_df.to_csv(os.path.join(output_dir, "candidate_models_comparison.csv"), index=False, encoding="utf-8-sig")
    imp_df.to_csv(os.path.join(output_dir, "rebuild_permutation_importance.csv"), index=False, encoding="utf-8-sig")
    pred_df.to_csv(os.path.join(output_dir, "prediction_sample.csv"), index=False, encoding="utf-8-sig")

    return metrics_df, imp_df, pred_df, best["model"]


def export_lr_coef(model_name, model, output_dir):
    if "LogisticRegression" not in str(model_name):
        return
    try:
        feat = model.named_steps["pre"].get_feature_names_out()
        coef = model.named_steps["clf"].coef_[0]
        coef_df = pd.DataFrame({"feature": feat, "coef": coef, "abs": np.abs(coef)}).sort_values("abs", ascending=False)
        coef_df.to_csv(os.path.join(output_dir, "logistic_coefficients.csv"), index=False, encoding="utf-8-sig")
    except Exception:
        pass


def build_behavior_cluster_profile(df, output_dir):
    candidate_feats = [
        "internet_usage__SWLJSC__sum",
        "discussion_record__creator_post_count",
        "running_punch__PUNCH_DAY__nunique",
        "daily_exercise__DKCS__sum",
        "door_access__active_days",
        "online_learning_overall__BFB__mean",
        "student_signin__ACTIVE_ID__count",
    ]
    feats = [c for c in candidate_feats if c in df.columns]
    if len(feats) < 4:
        return pd.DataFrame({"message": ["可用于聚类的行为字段不足，跳过聚类。"]})

    tmp = df[feats].copy()
    for c in feats:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
    tmp = tmp.fillna(tmp.median(numeric_only=True))

    X_cl = ClusterScaler().fit_transform(tmp)
    labels = KMeans(n_clusters=4, random_state=RANDOM_SEED, n_init=20).fit_predict(X_cl)
    cdf = tmp.copy()
    cdf["cluster"] = labels
    prof = cdf.groupby("cluster")[feats].mean().round(3)
    prof["size"] = cdf.groupby("cluster").size()

    # 自动命名
    total_eng = prof.sum(axis=1)
    names = {}
    names[prof["internet_usage__SWLJSC__sum"].idxmax()] = "高投入稳健型" if "internet_usage__SWLJSC__sum" in prof.columns else "高投入稳健型"
    if "door_access__active_days" in prof.columns:
        names[prof["door_access__active_days"].idxmax()] = "高活跃参与型"
    if "discussion_record__creator_post_count" in prof.columns:
        names[prof["discussion_record__creator_post_count"].idxmax()] = "互动活跃型"
    names[total_eng.idxmin()] = "低投入低活跃型"
    for i in prof.index:
        if i not in names:
            names[i] = "中投入波动型"

    prof.insert(0, "cluster_name", [names[i] for i in prof.index])
    prof.to_csv(os.path.join(output_dir, "行为聚类画像表_无乱码版.csv"), encoding="utf-8-sig")
    prof.to_excel(os.path.join(output_dir, "行为聚类画像表_无乱码版.xlsx"))
    return prof


def save_readme(output_dir, comp_df, metrics_df, label_source_col, original_shape, model_shape):
    txt = [
        "# 建模结果说明（适配新的整合后 CSV 数据集）",
        "",
        f"- 原始整合后数据集形状：{original_shape[0]} 行 × {original_shape[1]} 列",
        f"- 参与建模数据形状：{model_shape[0]} 行 × {model_shape[1]} 列",
        f"- 标签来源字段：`{label_source_col}`",
        f"- 标签策略：`{TARGET_STRATEGY}`",
        "",
        "## 标签定义",
        "",
        "当前默认采用“毕业去向风险”二分类：",
        "- 低风险：已落实就业/升学/保研/留学/西部计划等明确去向",
        "- 高风险：暂不就业、求职中、拟参加公招考试、拟出国出境、拟应征入伍、不就业拟升学、自由职业",
        "",
        "## 模型对比（5折CV）",
        comp_df.to_markdown(index=False),
        "",
        "## 最终指标",
        metrics_df.to_markdown(index=False),
        "",
        "## 输出文件",
        "- model_comparison.csv",
        "- candidate_models_comparison.csv",
        "- rebuild_metrics.csv",
        "- rebuild_permutation_importance.csv",
        "- prediction_sample.csv",
        "- 行为聚类画像表_无乱码版.csv / .xlsx",
    ]
    with open(os.path.join(output_dir, "README_重新建模说明.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(txt))


def main():
    print_step("1. 读取新的整合后 CSV 数据集")
    df = load_integrated_dataset(CSV_PATH)
    original_shape = df.shape
    ensure_dir(OUTPUT_DIR)
    print(f"原始数据：{df.shape[0]} 行 × {df.shape[1]} 列")

    print_step("2. 构造标签")
    df, label_source_col = add_risk_label(df)
    print(f"标签过滤后：{df.shape[0]} 行")
    print(df["risk_label"].value_counts(dropna=False))

    print_step("3. 特征筛选 + 特征工程")
    df_model = clean_and_select_features(df, label_source_col=label_source_col)
    df_model = add_feature_engineering(df_model)

    drop_cols = [c for c in ["sid", "student_id_raw", "risk_label"] if c in df_model.columns]
    X = df_model.drop(columns=drop_cols)
    y = df_model["risk_label"]
    print(f"参与建模特征数：{X.shape[1]}，样本数：{len(y)}")
    print(f"正样本占比：{y.mean():.2%}")

    print_step("4. 构建预处理器和模型")
    pre_linear, pre_tree, num_cols, cat_cols = build_preprocessors(X)
    print(f"数值特征：{len(num_cols)}，类别特征：{len(cat_cols)}")
    models = build_base_models(pre_linear, pre_tree)
    print(f"参与对比模型：{list(models.keys())}")

    print_step("5. 5 折交叉验证模型对比")
    comp_df = evaluate_models(models, X, y)
    comp_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False, encoding="utf-8-sig")
    print(comp_df.to_string())

    best_name = comp_df.iloc[0]["model"]
    ordered = {row["model"]: models[row["model"]] for _, row in comp_df.iterrows() if row["model"] in models}

    print_step("6. 调参 + 集成 + 最终训练")
    metrics_df, imp_df, pred_df, best_model = fit_best_model_and_export(
        best_name, ordered, X, y, OUTPUT_DIR, pre_linear, pre_tree
    )
    print(metrics_df.to_string())
    print("\nTop 10 特征重要性：")
    print(imp_df.head(10).to_string())

    export_lr_coef(best_name, best_model, OUTPUT_DIR)

    print_step("7. 行为聚类画像")
    print(build_behavior_cluster_profile(df_model, OUTPUT_DIR))

    print_step("8. 生成说明文档")
    save_readme(
        OUTPUT_DIR, comp_df, metrics_df, label_source_col,
        original_shape=original_shape, model_shape=df_model.shape
    )

    print_step("9. 保存模型")
    model_path = os.path.join(OUTPUT_DIR, "best_model.pkl")
    feature_path = os.path.join(OUTPUT_DIR, "feature_columns.pkl")
    joblib.dump(best_model, model_path)
    joblib.dump(list(X.columns), feature_path)
    print(f"模型已保存：{model_path}")
    print(f"特征列名已保存：{feature_path}")
    print(f"共 {len(X.columns)} 个特征列，API 调用时请确保输入特征与此一致")

    print_step("完成")
    print(f"输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n运行失败：", str(e))
        traceback.print_exc()

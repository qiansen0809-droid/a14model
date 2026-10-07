# -*- coding: utf-8 -*-
"""
学生就业风险预测系统（FastAPI + 网页上传 CSV）

启动命令：
    uvicorn main:app --reload --port 8000

访问地址：
    http://127.0.0.1:8000/upload
    http://127.0.0.1:8000/docs
"""

import os
from io import StringIO
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# 1. 模型路径配置
# ============================================================
# 这里改成你模型真实所在的位置
MODEL_DIR = r"C:\Users\11448\Desktop\2\rebuilt_model_outputs_newdata"

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")

print("正在加载模型...")
print("MODEL_PATH =", MODEL_PATH)
print("FEATURE_PATH =", FEATURE_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"找不到模型文件：{MODEL_PATH}")

if not os.path.exists(FEATURE_PATH):
    raise FileNotFoundError(f"找不到特征列文件：{FEATURE_PATH}")

model = joblib.load(MODEL_PATH)
feature_columns: List[str] = joblib.load(FEATURE_PATH)

print(f"模型加载成功，共 {len(feature_columns)} 个特征列")


# ============================================================
# 2. FastAPI 应用
# ============================================================
app = FastAPI(
    title="学生就业风险预测系统",
    description="上传 CSV 文件，批量评估学生是否合格/不合格",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 3. Pydantic 输入模型
# ============================================================
class StudentFeatures(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    gender: Optional[str] = Field(None, description="性别")
    ethnicity_name: Optional[str] = Field(None, description="民族")
    political_status_name: Optional[str] = Field(None, description="政治面貌")
    college_name: Optional[str] = Field(None, description="学院名称")
    major_name: Optional[str] = Field(None, description="专业名称")

    internet_usage__SWLJSC__sum: Optional[float] = None
    internet_usage__SWLJSC__mean: Optional[float] = None
    internet_usage__SWLJSC__count: Optional[float] = None

    discussion_record__creator_post_count: Optional[float] = None
    discussion_record__reply_count: Optional[float] = None

    running_punch__WEEKNUM__count: Optional[float] = None
    running_punch__PUNCH_DAY__nunique: Optional[float] = None

    daily_exercise__DKCS__sum: Optional[float] = None
    daily_exercise__DKCS__mean: Optional[float] = None
    daily_exercise__DKCS__count: Optional[float] = None

    door_access__event_count: Optional[float] = None
    door_access__active_days: Optional[float] = None

    club_activity__event_count: Optional[float] = None
    club_activity__active_days: Optional[float] = None

    online_learning_overall__BFB__mean: Optional[float] = None
    online_learning_overall__BFB__max: Optional[float] = None
    online_learning_overall__BFB__count: Optional[float] = None

    assignment_submission__SCORE__mean: Optional[float] = None
    assignment_submission__SCORE__count: Optional[float] = None
    assignment_submission__FULLMARKS__mean: Optional[float] = None
    assignment_submission__STATUS__nunique: Optional[float] = None

    exam_submission__SCORE__mean: Optional[float] = None
    exam_submission__SCORE__count: Optional[float] = None
    exam_submission__FULLMARKS__mean: Optional[float] = None
    exam_submission__STATUS__nunique: Optional[float] = None

    student_score__KCCJ__mean: Optional[float] = None
    student_score__KCCJ__max: Optional[float] = None
    student_score__JDCJ__mean: Optional[float] = None
    student_score__JDCJ__max: Optional[float] = None
    student_score__JDCJ__min: Optional[float] = None
    student_score__XF__sum: Optional[float] = None
    student_score__XF__mean: Optional[float] = None
    student_score__KCH__nunique: Optional[float] = None
    student_score__TDKCH__nunique: Optional[float] = None

    student_signin__ACTIVE_ID__count: Optional[float] = None
    student_signin__ACTIVE_STATE__mean: Optional[float] = None
    student_signin__ATTEND_TIME__count: Optional[float] = None
    student_signin__INSERT_TIME__nunique: Optional[float] = None

    scholarship_award__FFJE__sum: Optional[float] = None
    scholarship_award__FFJE__max: Optional[float] = None
    scholarship_award__JXJMC__nunique: Optional[float] = None
    scholarship_award__PDDJ__nunique: Optional[float] = None

    cet_score__KS_CJ__mean: Optional[float] = None
    cet_score__KS_CJ__max: Optional[float] = None
    cet_score__KS_CJ__count: Optional[float] = None

    comprehensive_eval__ZCJ__mean: Optional[float] = None
    comprehensive_eval__ZCJ__max: Optional[float] = None
    comprehensive_eval__ZYNJPM__mean: Optional[float] = None
    comprehensive_eval__BJPM__mean: Optional[float] = None
    comprehensive_eval__BJRS__mean: Optional[float] = None

    physical_test__ZF__mean: Optional[float] = None
    physical_test__BMI__nunique: Optional[float] = None
    physical_test__FHL__mean: Optional[float] = None
    physical_test__WS__mean: Optional[float] = None
    physical_test__LDTY__mean: Optional[float] = None
    physical_test__ZWTQQ__mean: Optional[float] = None

    physical_course__KC__nunique: Optional[float] = None
    physical_course__SKSJ__nunique: Optional[float] = None
    physical_course__ZKC__nunique: Optional[float] = None

    discipline_competition__HJJB__nunique: Optional[float] = None
    discipline_competition__HJDJ__nunique: Optional[float] = None


class PredictResponse(BaseModel):
    risk_probability: float
    risk_label: int
    risk_level: str
    threshold_used: float
    status: str


# ============================================================
# 4. 工具函数
# ============================================================
THRESHOLD = 0.5


def get_risk_level(prob: float) -> str:
    if prob < 0.3:
        return "低风险"
    elif prob < 0.6:
        return "中等风险"
    else:
        return "高风险"


def get_status(prob: float, threshold: float = THRESHOLD) -> str:
    return "合格" if prob < threshold else "不合格"


def build_input_df(data: StudentFeatures) -> pd.DataFrame:
    raw = data.model_dump()
    row = {}
    for col in feature_columns:
        row[col] = raw.get(col, np.nan)
    return pd.DataFrame([row])


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    temp_df = df.copy()
    for col in feature_columns:
        if col not in temp_df.columns:
            temp_df[col] = np.nan
    temp_df = temp_df[feature_columns]
    return temp_df


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    model_df = normalize_dataframe(df)
    probs = model.predict_proba(model_df)[:, 1]

    result_df = df.copy().reset_index(drop=True)
    result_df["risk_probability"] = [round(float(p), 4) for p in probs]
    result_df["risk_label"] = [int(float(p) >= THRESHOLD) for p in probs]
    result_df["risk_level"] = [get_risk_level(float(p)) for p in probs]
    result_df["status"] = [get_status(float(p)) for p in probs]
    return result_df


# ============================================================
# 5. 页面
# ============================================================
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    return """
    <html>
    <head>
        <meta charset="utf-8">
        <title>学生就业风险预测系统</title>
    </head>
    <body style="font-family: Arial; max-width: 900px; margin: 40px auto;">
        <h2>学生就业风险预测系统</h2>
        <p><a href="/upload">打开上传评估页面</a></p>
        <p><a href="/docs">查看接口文档</a></p>
        <p><a href="/health">健康检查</a></p>
    </body>
    </html>
    """


@app.get("/upload", response_class=HTMLResponse, include_in_schema=False)
def upload_page():
    return """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8" />
        <title>学生就业风险评估系统</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 30px auto;
                padding: 20px;
                background: #f7f9fc;
            }
            h1 { color: #1f2d3d; }
            .card {
                background: white;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            }
            button {
                background: #1677ff;
                color: white;
                border: none;
                padding: 10px 18px;
                border-radius: 8px;
                cursor: pointer;
                margin-right: 10px;
                margin-top: 10px;
            }
            button:hover {
                background: #0958d9;
            }
            input[type=file] {
                margin: 12px 0;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                background: white;
            }
            th, td {
                border: 1px solid #e5e7eb;
                padding: 8px;
                text-align: center;
                font-size: 14px;
            }
            th {
                background: #f3f4f6;
                position: sticky;
                top: 0;
            }
            .ok {
                color: #0f9d58;
                font-weight: bold;
            }
            .bad {
                color: #d93025;
                font-weight: bold;
            }
            .muted {
                color: #6b7280;
            }
            .summary {
                font-size: 16px;
                margin-top: 10px;
            }
            .table-wrap {
                max-height: 520px;
                overflow: auto;
                border: 1px solid #e5e7eb;
                border-radius: 8px;
            }
            .tip {
                color: #374151;
                line-height: 1.7;
            }
            .code {
                background: #f3f4f6;
                padding: 2px 6px;
                border-radius: 4px;
            }
        </style>
    </head>
    <body>
        <h1>学生就业风险评估系统</h1>

        <div class="card">
            <div class="tip">
                <p>请上传与模型特征一致的 <span class="code">.csv</span> 文件。</p>
                <p>系统会输出每位学生的风险概率、风险等级，以及“合格 / 不合格”结果。</p>
            </div>
            <input type="file" id="csvFile" accept=".csv" />
            <div>
                <button onclick="uploadFile()">开始评估</button>
                <button onclick="downloadResult()">导出结果 CSV</button>
                <button onclick="loadFeatures()">查看模型特征列</button>
            </div>
        </div>

        <div class="card">
            <h3>评估汇总</h3>
            <p id="summary" class="summary muted">暂未上传文件</p>
        </div>

        <div class="card">
            <h3>评估结果明细</h3>
            <div id="tableContainer" class="table-wrap"></div>
        </div>

        <div class="card">
            <h3>模型特征列</h3>
            <pre id="featuresBox" style="white-space: pre-wrap; line-height: 1.6; color: #374151;">点击“查看模型特征列”后显示</pre>
        </div>

        <script>
            let latestResults = [];

            async function uploadFile() {
                const fileInput = document.getElementById("csvFile");
                if (!fileInput.files.length) {
                    alert("请先选择 CSV 文件");
                    return;
                }

                const formData = new FormData();
                formData.append("file", fileInput.files[0]);

                const response = await fetch("/predict/csv", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();

                if (!response.ok) {
                    alert(data.detail || "上传失败");
                    return;
                }

                latestResults = data.results || [];
                const passCount = latestResults.filter(r => r.status === "合格").length;
                const failCount = latestResults.filter(r => r.status === "不合格").length;

                document.getElementById("summary").innerText =
                    `总人数：${data.count}，合格：${passCount}，不合格：${failCount}`;

                renderTable(latestResults);
            }

            function renderTable(rows) {
                const container = document.getElementById("tableContainer");
                if (!rows.length) {
                    container.innerHTML = '<p class="muted">暂无结果</p>';
                    return;
                }

                const columns = Object.keys(rows[0]);
                let html = "<table><thead><tr>";

                columns.forEach(col => {
                    html += `<th>${col}</th>`;
                });

                html += "</tr></thead><tbody>";

                rows.forEach(row => {
                    html += "<tr>";
                    columns.forEach(col => {
                        const value = row[col] ?? "";
                        if (col === "status") {
                            const cls = value === "合格" ? "ok" : "bad";
                            html += `<td class="${cls}">${value}</td>`;
                        } else {
                            html += `<td>${value}</td>`;
                        }
                    });
                    html += "</tr>";
                });

                html += "</tbody></table>";
                container.innerHTML = html;
            }

            async function loadFeatures() {
                const response = await fetch("/features");
                const data = await response.json();
                document.getElementById("featuresBox").innerText =
                    (data.feature_columns || []).join("\\n");
            }

            async function downloadResult() {
                const fileInput = document.getElementById("csvFile");
                if (!fileInput.files.length) {
                    alert("请先上传并评估 CSV 文件");
                    return;
                }

                const formData = new FormData();
                formData.append("file", fileInput.files[0]);

                const response = await fetch("/predict/csv/download", {
                    method: "POST",
                    body: formData
                });

                if (!response.ok) {
                    const err = await response.json();
                    alert(err.detail || "导出失败");
                    return;
                }

                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = "prediction_results.csv";
                document.body.appendChild(a);
                a.click();
                a.remove();
                window.URL.revokeObjectURL(url);
            }
        </script>
    </body>
    </html>
    """


# ============================================================
# 6. 接口
# ============================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "feature_count": len(feature_columns)
    }


@app.get("/features")
def get_features():
    return {
        "feature_columns": feature_columns,
        "count": len(feature_columns)
    }


@app.post("/predict", response_model=PredictResponse)
def predict(data: StudentFeatures):
    try:
        input_df = build_input_df(data)
        prob = float(model.predict_proba(input_df)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型推理失败：{str(e)}")

    return PredictResponse(
        risk_probability=round(prob, 4),
        risk_label=int(prob >= THRESHOLD),
        risk_level=get_risk_level(prob),
        threshold_used=THRESHOLD,
        status=get_status(prob)
    )


@app.post("/predict/batch")
def predict_batch(students: List[StudentFeatures]):
    if len(students) > 200:
        raise HTTPException(status_code=400, detail="单次批量最多 200 条")

    try:
        rows = [build_input_df(student).iloc[0] for student in students]
        batch_df = pd.DataFrame(rows).reset_index(drop=True)
        probs = model.predict_proba(batch_df)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量推理失败：{str(e)}")

    results = []
    for i, prob in enumerate(probs):
        prob = float(prob)
        results.append({
            "index": i,
            "risk_probability": round(prob, 4),
            "risk_label": int(prob >= THRESHOLD),
            "risk_level": get_risk_level(prob),
            "status": get_status(prob)
        })

    return {
        "count": len(results),
        "results": results
    }


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="请上传 .csv 文件")

    try:
        contents = await file.read()

        try:
            df = pd.read_csv(StringIO(contents.decode("utf-8-sig")))
        except UnicodeDecodeError:
            df = pd.read_csv(StringIO(contents.decode("gbk")))

        result_df = predict_dataframe(df)

        return {
            "count": len(result_df),
            "results": result_df.to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV 批量推理失败：{str(e)}")


@app.post("/predict/csv/download")
async def predict_csv_download(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="请上传 .csv 文件")

    try:
        contents = await file.read()

        try:
            df = pd.read_csv(StringIO(contents.decode("utf-8-sig")))
        except UnicodeDecodeError:
            df = pd.read_csv(StringIO(contents.decode("gbk")))

        result_df = predict_dataframe(df)
        csv_data = result_df.to_csv(index=False, encoding="utf-8-sig")

        return StreamingResponse(
            iter([csv_data]),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=prediction_results.csv"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"结果导出失败：{str(e)}")


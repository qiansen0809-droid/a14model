# -*- coding: utf-8 -*-
"""
将多源 xlsx 数据整合为一个学生粒度 CSV，并输出一份 Markdown 说明。

特点：
1. 直接读取已解压的数据文件夹，不读 zip。
2. 自动扫描 .xlsx / .xls 文件。
3. 输出到桌面：
   - integrated_student_dataset.csv
   - integrated_student_dataset_readme.md
4. CSV 使用 UTF-8-SIG 编码。
5. 输出列名尽量不含中文；统一使用英文前缀 + 原字段代码。
6. 对能够可靠映射到学生粒度（student_id/sid）的表进行聚合整合。
7. 对无法可靠映射到学生粒度的表记录到 Markdown 中，便于后续二次开发。

你需要修改的只有 DATA_DIR。
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


# =========================================================
# 1) 这里改成你的 xlsx 文件夹路径
# =========================================================
DATA_DIR = r"C:\Users\Lenovo\Desktop\a14基于多源数据的大学生行为分析与干预模型设计\数据集及类型"


# =========================================================
# 2) 输出路径：桌面
# =========================================================
DESKTOP_DIR = os.path.join(os.path.expanduser("~"), "Desktop")
OUTPUT_CSV = os.path.join(DESKTOP_DIR, "integrated_student_dataset.csv")
OUTPUT_MD = os.path.join(DESKTOP_DIR, "integrated_student_dataset_readme.md")


# =========================================================
# 3) 元数据文件（可选，不存在也能跑）
#    用于补充说明，不影响主流程
# =========================================================
SUMMARY_XLSX = os.path.join(DATA_DIR, "xlsx文件信息汇总.xlsx")
FIELD_DESC_MD = os.path.join(DATA_DIR, "字段说明汇总.md")


# =========================================================
# 4) 文件名前缀映射：避免 CSV 中出现中文列名前缀
# =========================================================
FILE_PREFIX_MAP = {
    "学生基本信息": "student_basic",
    "本科生综合测评": "comprehensive_eval",
    "毕业去向": "graduate_dest",
    "奖学金获奖": "scholarship_award",
    "考勤-汇总表": "attendance_summary",
    "考试提交记录": "exam_submission",
    "课程信息": "course_info",
    "课堂任务参与": "class_task_participation",
    "门禁数据": "door_access",
    "跑步打卡": "running_punch",
    "日常锻炼": "daily_exercise",
    "上课信息统计表": "course_class_stats",
    "上网统计": "internet_usage",
    "社团活动": "club_activity",
    "四六级成绩": "cet_score",
    "讨论记录": "discussion_record",
    "体测数据": "physical_test",
    "体育课": "physical_course",
    "图书馆打卡记录": "library_gate",
    "线上学习（综合表现）": "online_learning_overall",
    "选课信息": "course_selection",
    "学籍异动": "student_status_change",
    "学科竞赛": "discipline_competition",
    "学生成绩": "student_score",
    "学生签到记录": "student_signin",
    "学生体能考核数据": "physical_assessment",
    "学生作业提交记录": "assignment_submission",
}

# 兼容文件名小差异
FILE_PREFIX_FALLBACK = {
    "线上学习": "online_learning_overall",
    "线上学习(综合表现)": "online_learning_overall",
    "图书馆打卡": "library_gate",
    "考勤汇总表": "attendance_summary",
    "学生成绩信息": "student_score",
}


# =========================================================
# 5) 可能的学号字段
# =========================================================
SID_CANDIDATES = [
    "XH", "XSBH", "SID", "USERNUM", "KS_XH", "IDSERTAL", "cardId",
    "LOGIN_NAME", "CREATER_LOGIN_NAME", "REPLY_LOGIN_NAME", "XHHGH"
]

# 某些表里学号列不标准，需要特殊指定
SPECIAL_ID_COLUMN = {
    "学生基本信息": "XH",
    "本科生综合测评": "XH",
    "毕业去向": "SID",
    "奖学金获奖": "XSBH",
    "考勤-汇总表": "XH",
    "课堂任务参与": "LOGIN_NAME",
    "门禁数据": "IDSERTAL",
    "跑步打卡": "USERNUM",
    "日常锻炼": "XH",
    "上网统计": "XSBH",
    "社团活动": "XSBH",
    "四六级成绩": "KS_XH",
    "体测数据": "XM",     # 按字段说明，这里 XM 实际是学号
    "体育课": "XH",
    "图书馆打卡记录": "cardId",
    "线上学习（综合表现）": "LOGIN_NAME",
    "选课信息": "XH",
    "学籍异动": "XH",
    "学科竞赛": "XHHGH",
    "学生成绩": "XH",
    "学生签到记录": "LOGIN_NAME",
    "学生体能考核数据": "XH",
    "考试提交记录": "CREATER_LOGIN_NAME",
    "学生作业提交记录": "CREATER_LOGIN_NAME",
}

# 纯课程级/教师级，默认不直接进学生粒度主表
SKIP_DIRECT_MERGE = {
    "课程信息",
    "上课信息统计表",
}

# 需要特殊处理的表
SPECIAL_TABLES = {
    "讨论记录",
    "考试提交记录",
    "学生作业提交记录",
    "选课信息",
    "图书馆打卡记录",
    "门禁数据",
    "社团活动",
}


def read_excel_auto(path: str) -> pd.DataFrame:
    """
    读取 xlsx/xls，统一为字符串列名。
    """
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception:
        df = pd.read_excel(path)

    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_filename(stem: str) -> str:
    stem = stem.strip()
    if stem in FILE_PREFIX_MAP:
        return FILE_PREFIX_MAP[stem]
    if stem in FILE_PREFIX_FALLBACK:
        return FILE_PREFIX_FALLBACK[stem]
    # 兜底：只保留英文数字下划线
    stem_ascii = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    return stem_ascii if stem_ascii else "table"


def find_sid_column(df: pd.DataFrame, table_name: str):
    if table_name in SPECIAL_ID_COLUMN and SPECIAL_ID_COLUMN[table_name] in df.columns:
        return SPECIAL_ID_COLUMN[table_name]
    for c in SID_CANDIDATES:
        if c in df.columns:
            return c
    return None


def normalize_sid(series: pd.Series) -> pd.Series:
    """
    统一 student_id：
    - 转字符串
    - 去空格
    - 去 .0
    - 去非字母数字下划线连字符
    """
    s = series.astype(str).str.strip()
    s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.replace(r"\s+", "", regex=True)
    return s


def to_numeric_if_possible(df: pd.DataFrame) -> pd.DataFrame:
    """
    尝试把明显数值列转成数值。
    """
    out = df.copy()
    for c in out.columns:
        if c == "sid":
            continue
        # 只对 object 列尝试转换
        if out[c].dtype == object:
            sample = out[c].dropna().astype(str).head(30)
            if len(sample) == 0:
                continue
            # 如果大多数样本长得像数字，就转
            ok = sample.str.match(r"^-?\d+(\.\d+)?$").mean()
            if ok >= 0.7:
                out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def aggregate_generic(df: pd.DataFrame, sid_col: str, prefix: str) -> pd.DataFrame:
    """
    通用聚合：
    - 数值列：sum / mean / max / min / count_nonnull
    - 非数值列：nunique
    """
    tmp = df.copy()
    tmp["sid"] = normalize_sid(tmp[sid_col])
    tmp = tmp[tmp["sid"].notna()].copy()

    # 去掉完全空列
    tmp = tmp.dropna(axis=1, how="all")

    # 避免重复保留 sid 原列
    drop_cols = [sid_col] if sid_col in tmp.columns else []
    tmp = tmp.drop(columns=drop_cols, errors="ignore")

    tmp = to_numeric_if_possible(tmp)

    value_cols = [c for c in tmp.columns if c != "sid"]
    if not value_cols:
        return pd.DataFrame(columns=["sid"])

    num_cols = [c for c in value_cols if pd.api.types.is_numeric_dtype(tmp[c])]
    obj_cols = [c for c in value_cols if c not in num_cols]

    agg_parts = []

    if num_cols:
        num_agg = tmp.groupby("sid")[num_cols].agg(["sum", "mean", "max", "min", "count"])
        num_agg.columns = [f"{prefix}__{col}__{stat}" for col, stat in num_agg.columns]
        agg_parts.append(num_agg)

    if obj_cols:
        # 只保留低风险的聚合：唯一值数 + 最后一个非空值（少量）
        obj_nunique = tmp.groupby("sid")[obj_cols].nunique(dropna=True)
        obj_nunique.columns = [f"{prefix}__{col}__nunique" for col in obj_nunique.columns]
        agg_parts.append(obj_nunique)

        # 对字段数量做一点控制，避免太宽
        keep_last_cols = [c for c in obj_cols if c.upper() in {
            "XB", "XBMC", "XSM", "ZYM", "DEPARTMENT", "MAJOR", "CLASS_NAME", "BJMC",
            "GRADE", "NJ", "YX", "ZY", "BJ", "JKZKMC", "BYQXMC", "DJ"
        }]
        if keep_last_cols:
            obj_last = tmp.groupby("sid")[keep_last_cols].agg(lambda x: x.dropna().astype(str).iloc[-1] if x.dropna().shape[0] else np.nan)
            obj_last.columns = [f"{prefix}__{col}__last" for col in obj_last.columns]
            agg_parts.append(obj_last)

    if not agg_parts:
        return pd.DataFrame(columns=["sid"])

    out = pd.concat(agg_parts, axis=1).reset_index()
    return out


def aggregate_discussion(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    讨论记录特殊处理：
    - 发帖人（CREATER_LOGIN_NAME）统计发帖数
    - 回帖人（REPLY_LOGIN_NAME）统计回帖数
    """
    frames = []

    if "CREATER_LOGIN_NAME" in df.columns:
        a = df.copy()
        a["sid"] = normalize_sid(a["CREATER_LOGIN_NAME"])
        a = a[a["sid"].notna()]
        if len(a):
            tmp = a.groupby("sid").size().reset_index(name=f"{prefix}__creator_post_count")
            frames.append(tmp)

    if "REPLY_LOGIN_NAME" in df.columns:
        b = df.copy()
        b["sid"] = normalize_sid(b["REPLY_LOGIN_NAME"])
        b = b[b["sid"].notna()]
        if len(b):
            tmp = b.groupby("sid").size().reset_index(name=f"{prefix}__reply_count")
            frames.append(tmp)

    if not frames:
        return pd.DataFrame(columns=["sid"])

    out = frames[0]
    for x in frames[1:]:
        out = out.merge(x, on="sid", how="outer")
    return out


def aggregate_submission(df: pd.DataFrame, sid_col: str, prefix: str) -> pd.DataFrame:
    """
    考试/作业提交记录：
    - 提交数、已批阅数、平均分、最高分、满分均值
    """
    tmp = df.copy()
    tmp["sid"] = normalize_sid(tmp[sid_col])
    tmp = tmp[tmp["sid"].notna()].copy()

    for c in ["SCORE", "FULLMARKS", "STATUS", "TYPE"]:
        if c in tmp.columns:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    agg_dict = {}
    if "SCORE" in tmp.columns:
        agg_dict["SCORE"] = ["count", "mean", "max", "min"]
    if "FULLMARKS" in tmp.columns:
        agg_dict["FULLMARKS"] = ["mean", "max"]
    if "STATUS" in tmp.columns:
        agg_dict["STATUS"] = ["nunique"]

    if not agg_dict:
        return pd.DataFrame(columns=["sid"])

    out = tmp.groupby("sid").agg(agg_dict)
    out.columns = [f"{prefix}__{col}__{stat}" for col, stat in out.columns]
    out = out.reset_index()
    return out


def aggregate_access_records(df: pd.DataFrame, sid_col: str, prefix: str, time_col: str = None) -> pd.DataFrame:
    """
    门禁 / 图书馆 / 社团活动等事件型表：
    - 记录数
    - 去重日期数
    - 最早/最晚时间（如果有）
    """
    tmp = df.copy()
    tmp["sid"] = normalize_sid(tmp[sid_col])
    tmp = tmp[tmp["sid"].notna()].copy()
    if len(tmp) == 0:
        return pd.DataFrame(columns=["sid"])

    out = tmp.groupby("sid").size().reset_index(name=f"{prefix}__event_count")

    # 尝试从时间列衍生去重日期数
    if time_col and time_col in tmp.columns:
        dt = pd.to_datetime(tmp[time_col], errors="coerce")
        tmp["_dt_"] = dt
        tmp["_date_"] = dt.dt.date
        daily = tmp.groupby("sid")["_date_"].nunique(dropna=True).reset_index(name=f"{prefix}__active_days")
        first_t = tmp.groupby("sid")["_dt_"].min().reset_index(name=f"{prefix}__first_time")
        last_t = tmp.groupby("sid")["_dt_"].max().reset_index(name=f"{prefix}__last_time")
        out = out.merge(daily, on="sid", how="left").merge(first_t, on="sid", how="left").merge(last_t, on="sid", how="left")

    return out


def aggregate_course_selection(selection_df: pd.DataFrame, course_df: pd.DataFrame = None, class_stats_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    选课信息特殊处理。
    如果存在课程信息 / 上课信息统计表，则补充课程层面的聚合特征。
    """
    tmp = selection_df.copy()
    if "XH" not in tmp.columns:
        return pd.DataFrame(columns=["sid"])

    tmp["sid"] = normalize_sid(tmp["XH"])
    tmp = tmp[tmp["sid"].notna()].copy()

    for c in ["SFCX"]:
        if c in tmp.columns:
            tmp[c] = tmp[c].astype(str)

    base = tmp.groupby("sid").agg(
        course_selection__row_count=("sid", "size"),
        course_selection__distinct_kch=("KCH", pd.Series.nunique) if "KCH" in tmp.columns else ("sid", "size"),
        course_selection__distinct_jxbh=("JXBH", pd.Series.nunique) if "JXBH" in tmp.columns else ("sid", "size"),
        course_selection__retake_count=("SFCX", lambda x: x.astype(str).isin(["1", "是", "Y", "True", "true"]).sum()) if "SFCX" in tmp.columns else ("sid", "size"),
    ).reset_index()

    # 如果有课程信息：按 KCH 连接，给学生汇总学分/学时
    if course_df is not None and "KCH" in course_df.columns:
        cdf = course_df.copy()
        for c in ["XF", "XS"]:
            if c in cdf.columns:
                cdf[c] = pd.to_numeric(cdf[c], errors="coerce")

        cols = ["KCH"] + [c for c in ["XF", "XS"] if c in cdf.columns]
        cdf = cdf[cols].drop_duplicates(subset=["KCH"])
        join_df = tmp.merge(cdf, on="KCH", how="left")
        extra = join_df.groupby("sid").agg(
            course_selection__credit_sum=("XF", "sum") if "XF" in join_df.columns else ("sid", "size"),
            course_selection__credit_mean=("XF", "mean") if "XF" in join_df.columns else ("sid", "size"),
            course_selection__hours_sum=("XS", "sum") if "XS" in join_df.columns else ("sid", "size"),
        ).reset_index()
        base = base.merge(extra, on="sid", how="left")

    # 如果有上课信息统计：按课程号连接，给学生汇总低头率/抬头率等
    if class_stats_df is not None and "COURSE_CODE" in class_stats_df.columns and "KCH" in tmp.columns:
        sdf = class_stats_df.copy()
        for c in ["ATTENDEES_NUM", "ARRIVE_NUM", "BOWING_RATE", "HEAD_UP_RATE", "FRONT_ROW_RATE", "TOTAL_SCORE"]:
            if c in sdf.columns:
                sdf[c] = pd.to_numeric(sdf[c], errors="coerce")

        keep_cols = ["COURSE_CODE"] + [c for c in ["ATTENDEES_NUM", "ARRIVE_NUM", "BOWING_RATE", "HEAD_UP_RATE", "FRONT_ROW_RATE", "TOTAL_SCORE"] if c in sdf.columns]
        sdf = sdf[keep_cols]
        course_level = sdf.groupby("COURSE_CODE").mean(numeric_only=True).reset_index()
        join_df = tmp.merge(course_level, left_on="KCH", right_on="COURSE_CODE", how="left")
        extra = join_df.groupby("sid").mean(numeric_only=True).reset_index()
        rename_map = {}
        for c in extra.columns:
            if c != "sid":
                rename_map[c] = f"course_selection__{c.lower()}__mean"
        extra = extra.rename(columns=rename_map)
        base = base.merge(extra, on="sid", how="left")

    return base


def build_student_base(tables: dict) -> pd.DataFrame:
    """
    构造学生主表：
    优先用“学生基本信息”；
    如果不存在，则从所有可识别 sid 的表做并集。
    """
    if "学生基本信息" in tables:
        df = tables["学生基本信息"].copy()
        if "XH" in df.columns:
            df["sid"] = normalize_sid(df["XH"])
            df = df[df["sid"].notna()].copy()
            keep_cols = [c for c in ["sid", "XH", "XB", "MZMC", "ZZMMMC", "CSRQ", "JG", "XSM", "ZYM"] if c in df.columns]
            base = df[keep_cols].drop_duplicates(subset=["sid"]).copy()
            rename_map = {
                "XH": "student_id_raw",
                "XB": "gender",
                "MZMC": "ethnicity_name",
                "ZZMMMC": "political_status_name",
                "CSRQ": "birth_date",
                "JG": "native_place",
                "XSM": "college_name",
                "ZYM": "major_name",
            }
            base = base.rename(columns=rename_map)
            return base

    all_sid = []
    for name, df in tables.items():
        sid_col = find_sid_column(df, name)
        if sid_col and sid_col in df.columns:
            s = normalize_sid(df[sid_col]).dropna().drop_duplicates()
            all_sid.append(pd.DataFrame({"sid": s}))

    if all_sid:
        base = pd.concat(all_sid, ignore_index=True).drop_duplicates(subset=["sid"])
        return base

    return pd.DataFrame(columns=["sid"])


def safe_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if right is None or right.empty:
        return left
    return left.merge(right, on="sid", how="left")


def scan_xlsx_folder(folder: str):
    files = []
    for root, _, filenames in os.walk(folder):
        for fn in filenames:
            if fn.startswith("~$"):
                continue
            if fn.lower().endswith((".xlsx", ".xls")):
                files.append(os.path.join(root, fn))
    return sorted(files)


def load_all_tables(data_dir: str):
    """
    读取所有 xlsx/xls 到 dict:
    {文件名(不带扩展名): DataFrame}
    """
    tables = {}
    file_paths = scan_xlsx_folder(data_dir)
    for fp in file_paths:
        stem = Path(fp).stem.strip()
        try:
            df = read_excel_auto(fp)
            tables[stem] = df
        except Exception as e:
            print(f"[WARN] 读取失败: {fp} -> {e}")
    return tables


def make_markdown_report(
    integrated_df: pd.DataFrame,
    merged_files: list,
    skipped_files: dict,
    output_csv: str,
    output_md: str
):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# 预处理后数据集说明")
    lines.append("")
    lines.append(f"- 生成时间：{now_str}")
    lines.append(f"- 数据集文件：`{output_csv}`")
    lines.append(f"- 说明文件：`{output_md}`")
    lines.append(f"- 编码说明：CSV 使用 `UTF-8-SIG`，Markdown 使用 `UTF-8`。")
    lines.append("")
    lines.append("## 一、数据集粒度")
    lines.append("")
    lines.append("本次整合后的数据集为**学生粒度（one row per student）**，主键字段为：")
    lines.append("")
    lines.append("- `sid`：统一后的学生标识")
    lines.append("")
    lines.append("也就是说，最终 CSV 中每一行代表一个学生；原始多表中的明细记录（如门禁、打卡、作业提交、考试提交、图书馆进出等）已被聚合为学生级统计特征。")
    lines.append("")
    lines.append("## 二、预处理原则")
    lines.append("")
    lines.append("1. **统一主键**：将不同表中的学号字段（如 `XH`、`XSBH`、`SID`、`USERNUM`、`cardId`、`LOGIN_NAME` 等）尽量标准化为 `sid`。")
    lines.append("2. **学生粒度聚合**：")
    lines.append("   - 数值字段：按学生聚合为 `sum / mean / max / min / count` 等统计量；")
    lines.append("   - 类别字段：保留 `nunique`，少量关键字段保留 `last`；")
    lines.append("   - 事件型表：聚合为事件次数、活跃天数、最早/最晚时间等。")
    lines.append("3. **列名英文化**：最终 CSV 列名尽量不使用中文，采用 `英文前缀__原字段代码__聚合方式` 的形式。")
    lines.append("4. **尽量保留信息**：可可靠映射到学生粒度的表已纳入整合；无法可靠映射到学生粒度的表在下方单独列出。")
    lines.append("")
    lines.append("## 三、整合后数据集形状")
    lines.append("")
    lines.append(f"- 行数：`{integrated_df.shape[0]}`")
    lines.append(f"- 列数：`{integrated_df.shape[1]}`")
    lines.append("")
    lines.append("## 四、已整合的数据表")
    lines.append("")
    for name in merged_files:
        lines.append(f"- {name}")
    lines.append("")
    lines.append("## 五、未直接并入学生粒度主表的数据表")
    lines.append("")
    if skipped_files:
        for name, reason in skipped_files.items():
            lines.append(f"- **{name}**：{reason}")
    else:
        lines.append("- 无")
    lines.append("")
    lines.append("## 六、字段命名规则")
    lines.append("")
    lines.append("最终 CSV 中主要有三类字段：")
    lines.append("")
    lines.append("1. **基础字段**")
    lines.append("   - `sid`")
    lines.append("   - `student_id_raw`")
    lines.append("   - `gender`")
    lines.append("   - `college_name`")
    lines.append("   - `major_name`")
    lines.append("   - 等")
    lines.append("")
    lines.append("2. **聚合字段**")
    lines.append("   - 形式：`tableprefix__fieldcode__agg`")
    lines.append("   - 例如：")
    lines.append("     - `internet_usage__SWLJSC__sum`")
    lines.append("     - `student_score__KCCJ__mean`")
    lines.append("     - `library_gate__event_count`")
    lines.append("")
    lines.append("3. **特殊逻辑字段**")
    lines.append("   - 例如选课表与课程表、上课统计表联动产生的学生级特征。")
    lines.append("")
    lines.append("## 七、建议")
    lines.append("")
    lines.append("1. 在建模前先检查 `sid` 的缺失率和重复率。")
    lines.append("2. 对时间字段可进一步派生学年、学期、月份、是否夜间等特征。")
    lines.append("3. 对高缺失、高稀疏字段建议再做一次筛选。")
    lines.append("4. 如需更强的课程行为特征，可进一步把 `课程信息`、`上课信息统计表` 与 `选课信息` 做更细粒度联结。")
    lines.append("")
    lines.append("## 八、字段预览（前 80 列）")
    lines.append("")
    preview_cols = integrated_df.columns.tolist()[:80]
    for c in preview_cols:
        lines.append(f"- `{c}`")
    if integrated_df.shape[1] > 80:
        lines.append(f"- ……其余共 {integrated_df.shape[1] - 80} 列未在此展开。")
    lines.append("")

    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print("=" * 80)
    print("开始整合多源 xlsx 数据")
    print("=" * 80)
    print(f"DATA_DIR = {DATA_DIR}")
    print(f"OUTPUT_CSV = {OUTPUT_CSV}")
    print(f"OUTPUT_MD  = {OUTPUT_MD}")

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"数据文件夹不存在：{DATA_DIR}")

    tables = load_all_tables(DATA_DIR)
    if not tables:
        raise FileNotFoundError("未找到任何 xlsx/xls 文件，请检查文件夹路径。")

    print(f"共读取到 {len(tables)} 张表。")

    # 单独取课程表和上课统计表，供选课信息联动使用
    course_info_df = tables.get("课程信息")
    course_class_stats_df = tables.get("上课信息统计表")

    # 构造学生主表
    student_df = build_student_base(tables)
    if student_df.empty:
        raise ValueError("未能构造学生主表，请检查是否存在可识别的学号字段。")

    merged_files = []
    skipped_files = {}

    # 逐表处理
    for table_name, df in tables.items():
        if table_name == "学生基本信息":
            # 已作为基础主表使用
            merged_files.append(table_name)
            continue

        if table_name in SKIP_DIRECT_MERGE:
            skipped_files[table_name] = "该表为课程级/教师级数据，默认不直接并入学生粒度主表。"
            continue

        prefix = clean_filename(table_name)

        try:
            if table_name == "讨论记录":
                agg = aggregate_discussion(df, prefix)

            elif table_name in {"考试提交记录", "学生作业提交记录"}:
                sid_col = find_sid_column(df, table_name)
                if sid_col is None:
                    skipped_files[table_name] = "未识别到可用的学生标识字段。"
                    continue
                agg = aggregate_submission(df, sid_col, prefix)

            elif table_name == "选课信息":
                agg = aggregate_course_selection(
                    selection_df=df,
                    course_df=course_info_df,
                    class_stats_df=course_class_stats_df
                )

            elif table_name == "图书馆打卡记录":
                sid_col = find_sid_column(df, table_name)
                agg = aggregate_access_records(df, sid_col=sid_col, prefix=prefix, time_col="visittime")

            elif table_name == "门禁数据":
                sid_col = find_sid_column(df, table_name)
                agg = aggregate_access_records(df, sid_col=sid_col, prefix=prefix, time_col="LOGINTIME")

            elif table_name == "社团活动":
                sid_col = find_sid_column(df, table_name)
                agg = aggregate_access_records(df, sid_col=sid_col, prefix=prefix, time_col="HDRQ")

            else:
                sid_col = find_sid_column(df, table_name)
                if sid_col is None:
                    skipped_files[table_name] = "未识别到可用的学生标识字段。"
                    continue
                agg = aggregate_generic(df, sid_col=sid_col, prefix=prefix)

            if agg is None or agg.empty:
                skipped_files[table_name] = "聚合后为空，可能该表无法稳定映射到学生粒度。"
                continue

            student_df = safe_merge(student_df, agg)
            merged_files.append(table_name)
            print(f"[OK] 已整合：{table_name} -> 新增 {agg.shape[1]-1} 个字段")

        except Exception as e:
            skipped_files[table_name] = f"处理失败：{e}"
            print(f"[WARN] {table_name} 处理失败：{e}")

    # 列名去重/清洗
    student_df.columns = [str(c).strip() for c in student_df.columns]
    student_df = student_df.loc[:, ~student_df.columns.duplicated()].copy()

    # 时间列转字符串，避免 Excel/CSV 写出报错
    for c in student_df.columns:
        if pd.api.types.is_datetime64_any_dtype(student_df[c]):
            student_df[c] = student_df[c].astype(str)

    # 输出 CSV（UTF-8-SIG）
    student_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # 输出 Markdown（UTF-8）
    make_markdown_report(
        integrated_df=student_df,
        merged_files=merged_files,
        skipped_files=skipped_files,
        output_csv=OUTPUT_CSV,
        output_md=OUTPUT_MD
    )

    print("=" * 80)
    print("处理完成")
    print(f"CSV 已输出到：{OUTPUT_CSV}")
    print(f"说明已输出到：{OUTPUT_MD}")
    print("=" * 80)


if __name__ == "__main__":
    main()

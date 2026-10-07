# A14：大学生就业风险预测

学生表格数据的二分类风险预测、模型比较与行为聚类项目。这里发布源码与说明，不包含学生数据、训练产物或个人记录。

## 机器学习架构

本项目不是深度神经网络。数值特征经过缺失值填充和（在线性模型中）标准化，类别特征使用 One-Hot 编码，再进入 scikit-learn Pipeline。

- `train_model.py`：随机森林分类基线。
- `adapted_train_for_new_integrated_dataset.py`：整合数据训练版本，包含交叉验证、可选 LightGBM、Optuna 和软投票集成。
- `model_4_1_1.py`：比较逻辑回归、随机森林、ExtraTrees、GBDT，以及可选 LightGBM、XGBoost、CatBoost；执行五折交叉验证、部分模型 Optuna 调参、前三模型软投票，并比较候选模型。
- KMeans 四类聚类是辅助行为分析，不是风险分类器本身。最终导出的分类器由运行时的候选比较决定，不能把它固定称为某一种神经网络。
- `main(2).py`：FastAPI 推理及网页上传接口，加载已训练模型，不负责训练。

## 使用

```bash
pip install -r requirements.txt
python integrate_xlsx_to_csv.py
python model_4_1_1.py
```

运行前根据脚本说明调整数据路径并准备自己的授权数据。旧训练脚本读取 `new data2.csv`，整合版本读取 `integrated_student_dataset.csv`。数据整合步骤详见 `integrate_xlsx_to_csv_README.md`。

推理服务需先生成模型，再修改 `main(2).py` 中的 `MODEL_DIR`。由于原始文件名带括号，可使用：

```bash
uvicorn 'main(2):app' --host 127.0.0.1 --port 8000
```

`tools/pkl_to_joblib.py` 保留了原模型转换工具，使用前同样需要调整路径。仅加载可信来源的 pickle/joblib 文件。

## 发布范围

保留原始训练及推理逻辑；排除 CSV/XLSX 数据、模型文件、训练输出、压缩包、IDE 配置。源码经过 Python 语法检查，未在此次发布中重新训练或验证预测质量。依赖列表不代表已经验证全部历史脚本在同一环境中兼容。

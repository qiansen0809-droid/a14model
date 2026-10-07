# 这是唯一能解决你问题的脚本
import joblib

# 直接用 joblib 加载你的 pkl 文件
model = joblib.load("best_model.pkl")

# 重新保存为真正的 joblib 格式
joblib.dump(model, "best_model.joblib")

print("✅ 转换完成！已生成 best_model.joblib")
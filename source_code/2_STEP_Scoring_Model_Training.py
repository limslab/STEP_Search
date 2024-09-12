import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# 从positive.xlsx中读取数据
positive_df = pd.read_excel('positive_labeled.xlsx')

# 删除Sequence和Charge列中任意存在空值的行
positive_df = positive_df.dropna(subset=['Sequence', 'Charge'])

# 按照Sequence和Charge列分组并保留恰好有2行的组
positive_groups = positive_df.groupby(['Sequence', 'Charge', 'label']).filter(lambda x: len(x) == 2)

# 计算Retention time、Ratio_D、CCS之差的绝对值，并存入numpy数组
positive_diff = []
positive_pairs = []
for name, group in positive_groups.groupby(['Sequence', 'Charge', 'label']):
    row1, row2 = group.iloc[0], group.iloc[1]
    retention_time_diff = abs(row1['Retention time'] - row2['Retention time'])
    ratio_d_diff = min([row1['Ratio_D'], row2['Ratio_D']])
    ccs_diff = abs(row1['CCS'] - row2['CCS'])
    positive_diff.append([retention_time_diff, ratio_d_diff, ccs_diff])
    positive_pairs.append((row1, row2))

positive_diff = np.array(positive_diff)

# 创建行向量，长度为positive_diff的行数，所有值均为1
positive_labels = np.ones(positive_diff.shape[0])

# 从filtered_negative.xlsx中读取数据
negative_df = pd.read_excel('filtered_negative_labeled.xlsx')

# 删除Sequence和Charge列中任意存在空值的行
negative_df = negative_df.dropna(subset=['Sequence', 'Charge'])

# 按照Sequence和Charge列分组并保留恰好有2行的组
negative_groups = negative_df.groupby(['Sequence', 'Charge', 'label']).filter(lambda x: len(x) == 2)

# 计算delta_Retention time、Ratio_D、delta_CCS，并存入numpy数组
negative_diff = []
negative_pairs = []
for name, group in negative_groups.groupby(['Sequence', 'Charge', 'label']):
    row1, row2 = group.iloc[0], group.iloc[1]
    retention_time_diff = abs(row1['Retention time'] - row2['Retention time'])
    ratio_d_diff = min([row1['Ratio_D'], row2['Ratio_D']])
    ccs_diff = abs(row1['CCS'] - row2['CCS'])
    negative_diff.append([retention_time_diff, ratio_d_diff, ccs_diff])
    negative_pairs.append((row1, row2))

negative_diff = np.array(negative_diff)

# 将positive_diff与negative_diff连接
combined_diff = np.vstack((positive_diff, negative_diff))

# 延长positive_labels向量，延长的所有值均为0
negative_labels = np.zeros(negative_diff.shape[0])
combined_labels = np.concatenate((positive_labels, negative_labels))

# 对combined_diff的每一列进行平方
combined_diff_squared = combined_diff ** 2

# 计算每个特征与标签之间的p值
p_values = []
for i in range(combined_diff_squared.shape[1]):
    # 对每个特征进行独立样本t检验（正类 vs 负类）
    _, p_value = ttest_ind(combined_diff_squared[combined_labels == 1, i],
                           combined_diff_squared[combined_labels == 0, i])
    p_values.append(p_value)

# 应用Benjamini-Hochberg方法控制FDR
_, p_values_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

# 筛选显著的特征
significant_features = np.where(p_values_corrected < 0.05)[0]
print(f"Selected significant features based on FDR control: {significant_features}")

# 使用显著特征进行后续分析
X_train, X_test, y_train, y_test = train_test_split(combined_diff_squared[:, significant_features], combined_labels, test_size=0.2, random_state=42)

# 创建逻辑回归模型并进行训练
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# 输出模型的系数和截距
print("Model Coefficients (train/test split):", model.coef_)
print("Model Intercept (train/test split):", model.intercept_)

# 进行概率预测
y_proba = model.predict_proba(X_test)

# 输出前10个样本的预测概率
print("Predicted probabilities for the first 10 samples (train/test split):\n", y_proba[:10])

# 输出分类报告与准确率
y_pred = model.predict(X_test)
print("Accuracy (train/test split):", accuracy_score(y_test, y_pred))
print("Classification Report (train/test split):\n", classification_report(y_test, y_pred))

# 计算ROC曲线数据
fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba[:, 1])
roc_auc = auc(fpr, tpr)

# 保存ROC曲线数据到txt文件
with open("roc_curve_data.txt", "w") as f:
    f.write("FPR\tTPR\tThreshold\n")
    for i in range(len(fpr)):
        f.write(f"{fpr[i]:.6f}\t{tpr[i]:.6f}\t{thresholds_roc[i]:.6f}\n")

print("ROC curve data saved as roc_curve_data.txt")

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# 输出AUC值
print(f"AUC (train/test split): {roc_auc:.2f}")

# 计算FDR曲线数据
thresholds_fdr = np.linspace(0, 1, 100)
fdr_values = []
tprs_fdr = []
for threshold in thresholds_fdr:
    y_pred_fdr = (y_proba[:, 1] >= threshold).astype(int)
    tp = np.sum((y_pred_fdr == 1) & (y_test == 1))
    fp = np.sum((y_pred_fdr == 1) & (y_test == 0))
    fn = np.sum((y_pred_fdr == 0) & (y_test == 1))
    if tp + fp > 0:
        fdr = fp / (fp + tp)
    else:
        fdr = 0
    tpr_fdr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fdr_values.append(fdr)
    tprs_fdr.append(tpr_fdr)

# 保存FDR曲线数据到txt文件
with open("fdr_curve_data.txt", "w") as f:
    f.write("Threshold\tFDR\tTPR\n")
    for i in range(len(thresholds_fdr)):
        f.write(f"{thresholds_fdr[i]:.6f}\t{fdr_values[i]:.6f}\t{tprs_fdr[i]:.6f}\n")

print("FDR curve data saved as fdr_curve_data.txt")

# 绘制FDR曲线
plt.figure()
plt.plot(thresholds_fdr, fdr_values, color='blue', lw=2, label='FDR curve')
plt.xlabel('Threshold')
plt.ylabel('FDR')
plt.title('False Discovery Rate Curve')
plt.legend(loc="upper right")
plt.show()

# 输出FDR曲线数据的最后一个值（作为参考）
print(f"Last FDR value at threshold 1.0: {fdr_values[-1]:.6f}")

# 重新确定cutoff值
# 找到tpr与fpr相差最小的阈值作为新的cutoff值
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds_roc[optimal_idx]
print(f"Optimal threshold based on ROC curve: {optimal_threshold:.2f}")

# 使用新的cutoff值重新进行预测
y_new_pred = (y_proba[:, 1] >= optimal_threshold).astype(int)

# 输出新的分类报告与准确率
print("Accuracy with new cutoff (train/test split):", accuracy_score(y_test, y_new_pred))
print("Classification Report with new cutoff (train/test split):\n", classification_report(y_test, y_new_pred))

# 合并训练集和测试集，重新训练总的模型
total_model = LogisticRegression(class_weight='balanced')
total_model.fit(combined_diff_squared[:, significant_features], combined_labels)

# 输出总模型的系数和截距
print("Total Model Coefficients:", total_model.coef_)
print("Total Model Intercept:", total_model.intercept_)

# 计算归一化后的模型系数
scaler = StandardScaler()
combined_diff_scaled = scaler.fit_transform(combined_diff_squared[:, significant_features])
normalized_coefficients = total_model.coef_ / scaler.scale_[significant_features]
print("Normalized Total Model Coefficients:", normalized_coefficients)

# 保存训练好的total_model模型
joblib.dump(total_model, 'total_model.pkl')
print("Total model saved as total_model.pkl")
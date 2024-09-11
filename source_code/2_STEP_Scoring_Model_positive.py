import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

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

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(combined_diff_squared, combined_labels, test_size=0.2, random_state=42)

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

# 合并训练集和测试集，重新训练总的模型
total_model = LogisticRegression(class_weight='balanced')
total_model.fit(combined_diff_squared, combined_labels)

# 输出总模型的系数和截距
print("Total Model Coefficients:", total_model.coef_)
print("Total Model Intercept:", total_model.intercept_)

# 计算归一化后的模型系数
scaler = StandardScaler()
combined_diff_scaled = scaler.fit_transform(combined_diff_squared)
normalized_coefficients = total_model.coef_ / scaler.scale_
print("Normalized Total Model Coefficients:", normalized_coefficients)

# 保存训练好的total_model模型
joblib.dump(total_model, 'total_model.pkl')
print("Total model saved as total_model.pkl")

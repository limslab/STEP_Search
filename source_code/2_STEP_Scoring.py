import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def calculate_scores(df, model):
    # 按Sequence、Charge和label分组并保留恰好有2行的组
    groups = df.groupby(['Sequence', 'Charge', 'label']).filter(lambda x: len(x) == 2)
    scores = []

    # 计算每个组的特征差异并预测得分
    for name, group in groups.groupby(['Sequence', 'Charge', 'label']):
        row1, row2 = group.iloc[0], group.iloc[1]
        retention_time_diff = abs(row1['Retention time'] - row2['Retention time'])
        ratio_d_diff = min([row1['Ratio_D'], row2['Ratio_D']])
        ccs_diff = abs(row1['CCS'] - row2['CCS'])
        diff = np.array([[retention_time_diff, ratio_d_diff, ccs_diff]])
        diff_squared = diff ** 2
        score = model.predict_proba(diff_squared)[0, 1]
        scores.append((row1.name, row2.name, score))
        
    return scores

def add_scores_to_excel(input_file, model_file, output_file):
    # 读取模型和标准化器
    model = joblib.load(model_file)
    
    # 读取输入的Excel文件
    df = pd.read_excel(input_file)
    
    # 删除Sequence和Charge列中任意存在空值的行
    df = df.dropna(subset=['Sequence', 'Charge'])

    # 计算得分
    scaler = StandardScaler()
    combined_diff = np.array([[0, 0, 0]])  # 用于fit scaler的占位数组
    combined_diff_squared = combined_diff ** 2
    combined_diff_scaled = scaler.fit_transform(combined_diff_squared)
    scores = calculate_scores(df, model)
    
    # 添加得分到DataFrame
    score_dict = {index: score for index1, index2, score in scores for index in [index1, index2]}
    df['Score'] = df.index.map(score_dict)
    
    # 保存更新后的DataFrame到新文件
    df.to_excel(output_file, index=False)
    print(f"Updated data with scores saved as {output_file}")

# 使用示例
input_file = 'evidence_N2a_Ctrl.xlsx_Run3_brief_updated_labeled.xlsx'
model_file = 'total_model.pkl'
output_file = 'evidence_N2a_Ctrl_Run3_with_New_score.xlsx'

add_scores_to_excel(input_file, model_file, output_file)

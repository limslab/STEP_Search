import pandas as pd

# 读取Excel文件
df = pd.read_excel('negative_with_scores_new.xlsx')

# 删除Label列
df.drop(columns=['label'], inplace=True)

# 将Score列转换为字符串类型
df['Score'] = df['Score'].astype(str)

# 按Sequence和Charge列分组
grouped = df.groupby(['Sequence', 'Charge'])

# 初始化一个空的DataFrame来存储结果
result_df = pd.DataFrame()

# 遍历每个分组
for _, group in grouped:
    # 删除Intensity相同的行，只保留一个
    group = group.drop_duplicates(subset=['Intensity'])
    
    # 找到Intensity最大的行的索引
    max_intensity_idx = group['Intensity'].idxmax()
    
    # 将该行的Score列改为"-"
    group.loc[max_intensity_idx, 'Score'] = "-"
    
    # 将修改后的组添加到结果DataFrame中
    result_df = pd.concat([result_df, group], ignore_index=True)

# 保存结果到新的Excel文件
result_df.to_excel('negative_result.xlsx', index=False)
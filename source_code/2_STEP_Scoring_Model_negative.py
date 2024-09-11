import pandas as pd

# 读入excel文件
df = pd.read_excel('negative.xlsx')

# 删除任意提到的列中存在空值的行
df = df.dropna(subset=['Sequence', 'Charge', 'Intensity'])

# 找出Sequence和Charge列相同的行，并分组
grouped = df.groupby(['Sequence', 'Charge'])

# 创建一个空的DataFrame来存储结果
result_df = pd.DataFrame()

for name, group in grouped:
    if len(group) >= 2:
        # 按Intensity降序排序，并保留最大的2行
        top2 = group.nlargest(2, 'Intensity')
        
        # 计算2行的Intensity值之和
        intensity_sum = top2['Intensity'].sum()
        
        # 添加Ratio_D列
        top2['Ratio_D'] = top2['Intensity'] / intensity_sum
        
        # 将结果添加到result_df中
        result_df = pd.concat([result_df, top2])

# 将结果保存到一个新的excel文件中
result_df.to_excel('filtered_negative.xlsx', index=False)
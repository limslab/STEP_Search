import pandas as pd

# 读取Excel文件
df = pd.read_excel('evidence_N2a_300.xlsx_Run3_brief_updated.xlsx')

# 计算每个 unique 'Sequence' 和 'Charge' 组合的 'Ratio_D' 的最大值，然后获取相应的 'Retention time' 值
grouped = df.loc[df.groupby(['Sequence', 'Charge'])['Ratio_D'].idxmax()][['Sequence', 'Charge', 'Retention time']]

# 将结果合并回原始数据框，基于 'Sequence' 和 'Charge'，然后计算 'd_RT%'
df = df.merge(grouped, on=['Sequence', 'Charge'], suffixes=('', '_max'))
df['d_RT%'] = (abs(df['Retention time'] - df['Retention time_max']) / df['Retention time_max']) * 100

# 将每个Sequence和Charge组合的Retention time值都更改为该组合中Ratio_D最大的Retention time值
df['Retention time'] = df['Retention time_max']

# 删除不再需要的列
df = df.drop(columns='Retention time_max')

# 将结果写入新的Excel文件
df.to_excel('dRT_WT_N2a_300_Run3.xlsx', index=False)
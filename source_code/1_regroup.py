import pandas as pd

# 从positive.xlsx中读取数据
positive_df = pd.read_excel('evidence_N2a_300.xlsx_Run3_brief_updated.xlsx')

# 删除Sequence和Charge列中任意存在空值的行
positive_df = positive_df.dropna(subset=['Sequence', 'Charge'])

# 创建一个空的DataFrame用于存储结果
labeled_df = pd.DataFrame()

# 按照Sequence和Charge列分组
groups = positive_df.groupby(['Sequence', 'Charge'])

# 遍历每个分组
for name, group in groups:
    if len(group) == 2:
        # 如果组中恰好有两行，label设置为0
        group['label'] = 0
        labeled_df = pd.concat([labeled_df, group])
    elif len(group) > 2:
        # 如果组中有多于两行，找到Ratio_D最大的一行
        max_ratio_d_row = group.loc[group['Ratio_D'].idxmax()]
        remaining_rows = group.drop(max_ratio_d_row.name)
        
        # 配对形成2行的分组，并设置label
        label_counter = 1
        for _, row in remaining_rows.iterrows():
            pair = pd.DataFrame([max_ratio_d_row, row])
            pair['label'] = label_counter
            labeled_df = pd.concat([labeled_df, pair])
            label_counter += 1

# 保存结果到positive_labeled.xlsx
labeled_df.to_excel('evidence_N2a_300.xlsx_Run3_brief_updated_labeled.xlsx', index=False)

print("Processing complete. The labeled data has been saved to 'positive_labeled.xlsx'.")

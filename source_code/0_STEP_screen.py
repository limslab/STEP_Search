import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# file and sheet name
FILENAME = "evidence_N2a_300.xlsx"
SHEET_NAME = "Run1"

# RT-CUTOFF in determining isotope
RT_CUTOFF_ISO = 0.02

# whether plot RT distribution
# if true, then we do not output anything
CALRULATE_RT_DISTRIBUTION = False

# if calculate RT distribution
PLOT_BINS = 50

# if not calculate RT distribution 
RT_CUTOFF_LOWER = 0.005
RT_CUTOFF_UPPER = 100

# if RE-screen results with similar RTs and CCSs
RE_SCREEN = True
RE_SCREEN_CCS_CUTOFF = 0.005

# cut-off for re-screen ratio
RE_SCREEN_RATIO_CUTOFF = 0.01

data = pd.read_excel(FILENAME, sheet_name=SHEET_NAME)

# step 0, drop all the rows that has a zero intensity
data = data.dropna(subset=['Intensity'])

# step 1, screen "Modifications"
screened_data_1 = data[data["Modifications"] == "Unmodified"]

# step 2, get possible charges of each sequence
sequences = screened_data_1["Sequence"].unique()
print(f"Dealing with {len(sequences)} unmodified sequences")

# loop over these sequences
unique_sequence_charge_list = []
for sequence in sequences:
    each_sequence = screened_data_1[screened_data_1["Sequence"] == sequence]

    # loop over all possible charges
    charges = each_sequence["Charge"].unique()

    for charge in charges:
        each_charge = each_sequence[each_sequence["Charge"] == charge]
        if len(each_charge) == 1:
            continue
        else:
            unique_sequence_charge_list.append(each_charge)

print(f"Dealing with {len(unique_sequence_charge_list)} charged sequences with possible different RT")

# step 3: remove isotopic peak
unique_sequence_charge_isotope_list = []
for unique_sequence_charge in unique_sequence_charge_list:
    # remove adjacent rows with diff in MS/MS m/z in [0.97, 1.03]
    # record which rows to be deleted
    to_delete = pd.Series(False, index=unique_sequence_charge.index)

    for i, row_i in unique_sequence_charge.iterrows():
        for j, row_j in unique_sequence_charge.iterrows():
            iso_mass = (row_i['MS/MS m/z'] - row_j['MS/MS m/z']) * row_i['Charge']
            if ((iso_mass >= 0.98 and iso_mass <= 1.02) or (iso_mass >= 1.96 and iso_mass <= 2.04)) and ((row_i['Retention time'] - row_j['Retention time']) / row_i['Retention time'] < RT_CUTOFF_ISO):
                to_delete[i] = True

    unique_sequence_charge = unique_sequence_charge[~to_delete]

    if len(unique_sequence_charge) == 1:
            continue
    else:
        unique_sequence_charge_isotope_list.append(unique_sequence_charge)
print(f"Dealing with {len(unique_sequence_charge_isotope_list)} charged sequences with possible different RT after removing isotopic peaks")

# step 4: calculate RT distribution
if CALRULATE_RT_DISTRIBUTION:
    
    relative_rt_differences = []
    # loop over all charged sequences with the same charge
    for unique_sequence_charge in unique_sequence_charge_isotope_list:
        retention_time = unique_sequence_charge["Retention time"].values

        relative_differences_group = [abs(j-i)/(0.5*(j+i)) for i in retention_time for j in retention_time]
        relative_rt_differences += relative_differences_group

    plt.hist(relative_rt_differences, bins=PLOT_BINS, edgecolor='black')
    plt.title('Distribution of Relative Differences')
    plt.xlabel('Relative Difference')
    plt.ylabel('Frequency')
    plt.show()

# step 5: screen RT
if not CALRULATE_RT_DISTRIBUTION:
    chirality_list = []
    for unique_sequence_charge in unique_sequence_charge_isotope_list:

        unique_sequence_charge = unique_sequence_charge.copy()

        max_intensity_index = unique_sequence_charge['Intensity'].idxmax()
        max_intensity_time = unique_sequence_charge.loc[max_intensity_index, 'Retention time']
        absolute_cutoff = max_intensity_time * RT_CUTOFF_LOWER
        bins = np.arange(unique_sequence_charge['Retention time'].min() - 10e-8, unique_sequence_charge['Retention time'].max() + absolute_cutoff * 2 , absolute_cutoff)
        unique_sequence_charge['binned_Retention_time'] = pd.cut(unique_sequence_charge['Retention time'], bins)
        binned_RTs = unique_sequence_charge['binned_Retention_time'].unique()

        # guarantee bin_index < int(RT_CUTOFF_UPPER / RT_CUTOFF_LOWER)
        max_allowed_bin_number = int(RT_CUTOFF_UPPER / RT_CUTOFF_LOWER)
        if len(binned_RTs.categories) > max_allowed_bin_number:
            top_categories = binned_RTs.categories[:max_allowed_bin_number]
            unique_sequence_charge['binned_Retention_time'] = unique_sequence_charge['binned_Retention_time'].cat.set_categories(top_categories)
            unique_sequence_charge = unique_sequence_charge[unique_sequence_charge['binned_Retention_time'].isin(top_categories)]
        binned_RTs = unique_sequence_charge['binned_Retention_time'].unique()

        if len(binned_RTs) == 1:
            continue
        else:
            if unique_sequence_charge.empty:
                continue

            if RE_SCREEN:
                grouped = unique_sequence_charge.groupby('binned_Retention_time')

                idx = []

                # iterate over each 'binned_Retention_time' group
                for name, group in unique_sequence_charge.groupby('binned_Retention_time'):
                    if not group.empty:
                        # check if 'CCS' column exists and further group by 'CCS' if it does
                        if 'CCS' in group.columns:
                            for name_ccs, group_ccs in group.groupby(
                                pd.cut(
                                    group['CCS'], 
                                    np.arange(
                                        group['CCS'].min() - 10e-8 , 
                                        group['CCS'].max() + group['CCS'].max() * RE_SCREEN_CCS_CUTOFF,
                                        group['CCS'].max() * RE_SCREEN_CCS_CUTOFF)
                                    )
                                ):
                                if not group_ccs.empty:
                                    idx.append(group_ccs['Intensity'].idxmax())
                        else:
                            idx.append(group['Intensity'].idxmax())

                # Use the indices to select only the rows with maximum 'Intensity' in each group
                unique_sequence_charge_max_intensity = unique_sequence_charge.loc[idx]
            else:
                unique_sequence_charge_max_intensity = unique_sequence_charge

            total_intensity = unique_sequence_charge_max_intensity["Intensity"].sum()
            unique_sequence_charge_max_intensity["Ratio_D"] = unique_sequence_charge_max_intensity["Intensity"] / total_intensity
            chirality_list.append(unique_sequence_charge_max_intensity)

    print(f"We have {len(chirality_list)} sequences with chiral modification")

    # result = pd.concat(chirality_list)
    
    # print(f"writing result to {FILENAME}_{SHEET_NAME}.xlsx")
    # result.to_excel(f'{FILENAME}_{SHEET_NAME}.xlsx', index=False)


    # # columns to be outputted
    # info = ["Sequence", "Proteins", "Charge", "m/z", "Retention time", "Intensity", "Ratio_D"]
    # if "CCS" in result.columns:
    #     info.append("CCS")

    # result_brief = result[info]
    # print(f"writing brief result to {FILENAME}_{SHEET_NAME}_brief.xlsx")
    # result_brief.to_excel(f"{FILENAME}_{SHEET_NAME}_brief.xlsx", index=False)

# step 6: re-screen ratio_D
if not CALRULATE_RT_DISTRIBUTION:
    
    result = pd.concat(chirality_list)

    # 删除 Ratio_D 小于 1% 的行
    result = result[result["Ratio_D"] >= RE_SCREEN_RATIO_CUTOFF]

    # 删除长度等于 1 的组
    group_lengths = result.groupby(['Sequence', 'Charge']).size()
    valid_groups = group_lengths[group_lengths > 1].index
    result = result.set_index(['Sequence', 'Charge'])
    result = result.loc[valid_groups].reset_index()

    # 剩下的组，重新根据 Intensity 计算 Ratio_D
    # 下面这行代码是触发错误的代码行
    # result['Ratio_D'] = result.groupby(['Sequence', 'Charge'])['Intensity'].apply(lambda x: x / x.sum())

    # 为了修复这个错误，我们可以使用 transform 方法，它会返回与原 DataFrame 相同长度的 Series
    result['Ratio_D'] = result.groupby(['Sequence', 'Charge'])['Intensity'].transform(lambda x: x / x.sum())

    # 输出更新后的结果
    print(f"writing updated result to {FILENAME}_{SHEET_NAME}_updated.xlsx")
    result.to_excel(f'{FILENAME}_{SHEET_NAME}_updated.xlsx', index=False)

    # 输出简要的结果
    info = ["Sequence", "Proteins", "Charge", "m/z", "Retention time", "Intensity", "Ratio_D"]
    if "CCS" in result.columns:
        info.append("CCS")

    result_brief = result[info]
    print(f"writing updated brief result to {FILENAME}_{SHEET_NAME}_brief_updated.xlsx")
    result_brief.to_excel(f"{FILENAME}_{SHEET_NAME}_brief_updated.xlsx", index=False)
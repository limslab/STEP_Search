# STEP_Search (v1)

### STEP_Search is a set of data processing tools for untargeted extracting stereoisomeric and other isomeric data from proteomics database searching output files, which is developed by LimsLab @ Nankai University (The Gongyu Li Research Group), through collaboration with Dr. Haohao Fu @ Nankai University. 

### Code Availability. All code used in this study is available on GitHub (https://github.com/limslab/STEP_Search.git). The repository contains Python scripts for data processing and analysis. The code was developed using Python 3.10, with key dependencies listed in the requirements.txt file. A comprehensive README.md provides step-by-step instructions for reproducing our analyses. The raw data processed by this code is available from both the Supporting Information and the GitHub repository. For any questions regarding the code, please contact Prof. Dr. Gongyu Li (ligongyu@nankai.edu.cn). 

Stereoproteome (STEP)/isomeric data searching workflow: 
1.	Data Acquisition: Obtain raw omics data (.raw, .wiff, .mzML) from biological samples (tissues, cells, urine) via ProteomeXchange or direct proteomics experiments for protein database retrieval. 
2.	Data Processing: Import raw data into MaxQuant software. Select appropriate database and set parameters to generate .txt/.xlsx files for isomeric retrieval.
3.	Initial Data Screening: Execute "0_STEP_screen" script. Retrieve "evidence" file (demo data: evidence_N2a_Ctrl_Run2), filter contaminants/reverse entries, and convert .txt to .xlsx format with separate sheets for each sample injection. This spreadsheet serves as the input for isomer retrieval.
4.	Data Refinement:
a. Exclude entries with zero or null Intensity values.
b. Selectively remove or retain entries with "modified/unmodified" in the modification column based on experimental requirements.
c. Group remaining entries by sequence and charge. Remove low-intensity charge information to ensure unique primary charge peaks for each sequence.
d. Eliminate entries within groups if iso-mass interpolation is 1 or 2 with low intensity.
e. Apply retention time (RT) difference criterion to remove entries with relative RT shift < 0.5% and low intensity to exclude isotope and abnormal tailing peaks.
5.	Isomer Identification:
a. Calculate RT distribution between adjacent peaks: ΔRT% = (RT1 - RT2) / RT1 * 100.
b. Filter peptides with isomeric modifications using ΔRT > 0.5% (default based on STEP standards).
c. If applicable, apply ΔCCS > 3% criterion (ΔCCS% = (CCS1 - CCS2) / CCS1 * 100) while maintaining constant RT.
6.	Quantification:
a. Calculate relative proportions of target peaks within sequences.
b. Remove false-positive noise (ratio < 1%).
c. Recalculate ratios to determine isomerization ratio and number of isomers per sequence. 
7.	Data Compilation:
a. Generate concise isomeric modification list with protein/gene name, sequence, charge, m/z, RT, CCS, intensity, and isomer ratio.
b. Produce comprehensive list with all input details.
c. Export both Excel files (evidence_N2a_Ctrl.xlsx_Run2_brief_updated.xlsx; evidence_N2a_Ctrl.xlsx_Run2_updated.xlsx).
8.	Retention Time Analysis: Execute "0_dRT_WT" script to calculate the difference in retention time between all-L and isomer configurations (output file: dRT_all-L_N2a_Ctrl_Run2.xlsx). 
9.	Scoring Model Development:
a. Analyze demo data to obtain a scoring model (negative.xlsx, positive.xlsx).
b. Execute "1_STEP_Scoring_Model_negative" script to obtain filtered negative dataset (filtered_negative.xlsx). 
c. Run "2_regroup" script to organize datasets into groups based on isomer entities. (Input files: positive.xlsx/filtered_negative.xlsx; Output files: positive_labeled.xlsx, filtered_negative_labeled.xlsx)
d. Execute "2_STEP_Scoring_Model_Training" script to train labeled positive and negative data (positive_labeled.xlsx, filtered_negative_labeled.xlsx) to obtain a training model (Table 1, total model.pkl). The cutoff condition (STEPScore > 0.19) was further derived based on the Receiver Operating Characteristic (ROC) curve from the algorithm’s defaults (STEPScore > 0.5). 
e. Determine optimal independent variable features using False Discovery Rate (FDR) calculation (p < 0.05 cutoff) based on the Benjamini-Hochberg method (roc_curve_data.txt, fdr_curve_data.txt). 
10.	Model Application:
a. Execute "3_STEP_Scoring_Model_Prediction" script to predict STEPscores for model/demo data (positive_with_new_score.xlsx).
b. Run "4_reformat_output" script to optimize the output format and obtain the final output list (positive_with_new_score_result.xlsx). 

Done!




# Configurations for the Main_project
"""
Author: Zhang Lu
Version: 1.0.0
Date:2025-02-14
Description: Happy Valentine's Day.
"""

## locations of the dictionary and the data
f = r'D:\YLZC\Models1\贷前'
data_f = r'D:\YLZC\Models1\数据\data\output\data_all_0726.pkl'

## classify the variables
### 模型Y值
y = 'flag'
### 月份变量
yearmonth = 'yearmonth'
### 后续无用的标签变量
KEY_list = ['tag1','tag2']
### 除模型Y值以外的，其他验证Y标签
Y_list = ['fpd_k30','mob2_k30','mob4_k30','mob5_k30','mob6_k30']
### 直接删除，对后续子客群等，都无用的变量
del_list=['feature_time']
### 不能入模型的变量，但是后续 子客群，其他逾期口径验证需要
ext_list = ['cust_id', 'yearmonth', 'G0', 'G1', 'fpd_k30', 'mob2_k30', 'mob4_k30', 'mob5_k30', 'mob6_k30']


# 项目名称
part_nm: str = '贷前_LGB'
# 验证数据集年月
vldt_ym = ['202401']
# 离散变量定义
list_dis = ['UPPB027']
# train test oot标签名（function中直接写死）
samp_col = 'samp_type'
# 概况输出文件名称
output_file = 'output/%s_标签统计结果.csv' % part_nm
# 报告 输出文件名称
report_file1 = 'result/%s_Required_report1.xlsx' % part_nm
report_file2 = 'result/%s_Required_report2.xlsx' % part_nm
#模型部署 所需文档保存路径
model_folder = 'result/%s' % part_nm
model_var_file  = r'result/%s/model_vars.csv' % part_nm
score_file = r'result/%s/oral_model_vars.csv' % part_nm
pkl_file = r'result/%s/model.pkl' % part_nm

#.....................................Report 1 Config .............................................


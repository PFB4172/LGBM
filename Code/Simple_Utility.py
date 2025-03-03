import numpy as np
import pandas as pd
import os

"""***************
Split Functions
**************"""
## 1.去除两边tail，后等频
def split_without_tails(series, lower_quantile, upper_quantile, split_num):
    # 注意，使用时split num要比最终分段数，少1。因为已经去掉了tails
    q_low = series.quantile(lower_quantile)  # Lower quantile
    q_high = series.quantile(upper_quantile)  # Upper quantile
    filtered_values = series[(series >= q_low) & (series <= q_high)]
    min_val, max_val = filtered_values.min(), filtered_values.max()
    bins = np.linspace(min_val, max_val, split_num)

    bin_name = []
    bin_name.append(f"(-, {bins[0]}]")
    for i in range(len(bins) - 1):
        bin_name.append(f"[{bins[i]}, {bins[i + 1]})")
    bin_name.append(f"({bins[-1]}, +)")
    return bins,bin_name
## 2.等距
def split_equal_freq(series, split_num):
    _,bins = pd.qcut(series, q=split_num, retbins=True)
    bins = bins[1:-1]
    bin_name = []
    bin_name.append(f"(-, {bins[0]}]")
    for i in range(len(bins) - 1):
        bin_name.append(f"[{bins[i]}, {bins[i + 1]})")
    bin_name.append(f"({bins[-1]}, +)")
    return bins, bin_name
## 3.人工分组
def split_manual_distance(min_val, max_val, split_num):
    step = int((max_val - min_val) / (split_num-2))
    bins = list(np.arange(min_val, max_val, step))
    if bins[-1] < max_val:
        bins.append(max_val)
    bin_name = []
    bin_name.append(f"(-, {bins[0]}]")
    for i in range(len(bins) - 1):
        bin_name.append(f"[{bins[i]}, {bins[i + 1]})")
    bin_name.append(f"({bins[-1]}, +)")
    return bins, bin_name

"""***************
Drop High Similarity Rate Columns
**************"""
## 1. Calculate
def calculate_similarity_rate(column):
    """
    同值率计算
    :param column: 待计算dataframe列 df[]
    :return:similarity_rate 同值率
    """
    unique_values = column.unique()
    if len(unique_values) == 1:
        return 1.0
    else:
        most_common_value_count = column.value_counts().iloc[0]
        total_count = len(column)
        similarity_rate = most_common_value_count / total_count
        return similarity_rate
# 2. Drop High Similarity Rate Columns
def drop_same_rate_high_column(data, cannot_delete_list, rate):
    """
    同值率处理
    data: 数据集  dataframe
    rate: 同值率阈值
    """

    # 定义同值率集合
    similarity_rates = {}
    # 计算每列同值率
    print("原始数据集：", data.shape)
    print("数据集总共：", len(data.columns), "列")
    for i, column_name in enumerate(data.columns):
        print("   >>第", i, "列", column_name, "处理中")
        similarity_rates[column_name] = calculate_similarity_rate(data[column_name])

    high_similarity_columns = [column_name for column_name, value in similarity_rates.items() if value > rate]
    print("删除同值率高于", rate, "的值")
    processed_data = data.drop(columns=list(set(high_similarity_columns)-set(cannot_delete_list)))
    print("最终数据集为：", processed_data.shape)
    return processed_data

"""***************
Drop High Missing Rate Columns
**************"""
def calculate_missing_rate(data,cannot_delete_list, rate):
    """
    缺失率处理 删除缺失率高于 rate的值
    data: 数据集  dataframe
    rate: 缺失率阈值
    """
    # 数据集情况
    print("原始数据集：", data.shape)
    print("数据集总共：", len(data.columns), "列")

    # 计算每列的缺失率
    missing_rates = data.isnull().mean()
    # 找出缺失率高的列，可以根据实际情况设定一个阈值，比如缺失率大于 0.5
    high_missing_rate_columns = [col for col, rate_eg in missing_rates.items() if rate_eg > rate]
    print("删除缺失率高于", rate, "的值")
    processed_data = data.drop(columns=list(set(high_missing_rate_columns)-set(cannot_delete_list)))
    print("最终数据集为：", processed_data.shape)
    return processed_data

"""***************
Calculate non-Missing Rate Columns
**************"""
def has_value_rate(df):
    rates = []
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            rate = (df[column] > 0).mean()
        else:
            rate = df[column].notnull().mean()
        rates.append([column, rate])
    result_df = pd.DataFrame(rates, columns=['var', 'Rate'])
    return result_df

"""***************
Over_Sampling
**************"""
def under_sampling(data , labels , bad_sample_target):
    """
    过采样（Oversampling）
        过采样是一种通过增加少数类样本数量来平衡数据集的方法。其主要思想是生成新的少数类样本，使少数类的样本数量增加到与多数类相同或相近。
    随机过采样
    data: 数据集 x
    labels 数据集 y
    bad_sample_target: 期望的坏样本占比
    """
    # 坏样本占比
    bad_proportion = len(np.where(labels == 1)[0])/len(labels)
    # 坏样本数量
    bad_len = len(np.where(labels == 1)[0])
    # 好样本数量
    good_len = len(np.where(labels != 1)[0])
    # 抽样后好样本比例
    good_sample_target = 1 - bad_sample_target
    if bad_proportion < bad_sample_target:
        print("坏样本较少——过抽坏样本 随机抽样")
        # 总样本数
        total_target_num = round(good_len / good_sample_target)
        # 缺少的坏样本数
        need_bad = total_target_num - len(labels)
        print(" --过抽坏样本为：", need_bad)
        # 坏样本
        minority_class_indices = np.where(labels == 1)[0]
        # 固定随机数
        np.random.seed(42)

        # 针对坏数据随机过抽样
        oversampled_indices = np.random.choice(minority_class_indices,
                                               size = need_bad,
                                               replace = True)
        oversampled_data = data[oversampled_indices]
        oversampled_labels = labels[oversampled_indices]

        ## 数据集为 array
        combined_data = np.vstack((data, oversampled_data))
        combined_target = np.hstack((labels, oversampled_labels))
    else:
        print("好样本较少——过抽好样本 随机抽样")
        # 样本总数
        total_target_num = round(bad_len / bad_sample_target)
        # 缺少好样本数量
        need_good = total_target_num - len(labels)
        print(" --过抽好样本为：" , need_good)
        # 好样本
        minority_class_indices = np.where(labels != 1)[0]
        # 固定随机数
        np.random.seed(42)
        # 随机过抽样
        oversampled_indices = np.random.choice(minority_class_indices,
                                               size = need_good,
                                               replace = True)
        oversampled_data = data[oversampled_indices]
        oversampled_labels = labels[oversampled_indices]
        combined_data = np.vstack((data, oversampled_data))
        combined_target = np.hstack((labels, oversampled_labels))

    return combined_data , combined_target

"""***************
Add image
**************"""
def add_image(ws, im, std_width, std_high, scale, row_num, col_num):
    from PIL import Image
    image = Image.open(im)

    x_scale = std_width * scale / image.size[0]
    y_scale = std_high * scale / image.size[1]
    img_format = {
        "x_scale": x_scale,  # 水平缩放比例
        "y_scale": y_scale  # 垂直缩放比例
    }
    ws.insert_image(row_num, col_num, im, img_format)

"""***************
Save model files
**************"""
def model_auto_file(df, model, model_folder, pkl_file, model_var_file, score_file):
    import pickle
    #模型部署所需文档保存
    if os.path.exists(model_folder):
        pass
    else:
        os.mkdir(model_folder)
    #原始入模变量，分箱后入模变量，WOE后入模变量，prob,score
    model_var = model.feature_name_
    y_pred = model.predict_proba(df[model_var])[:, 1]
    df['y_pred'] = y_pred
    df['y_pred_score'] = np.round(400 - 35 / np.log(2) * np.log(y_pred / (1 - y_pred)))
    df.to_csv(score_file,
                  encoding='utf8',
                  index=False)
    model_vars = pd.DataFrame({
        'col_nm': list(model.feature_name_),
        'importance': list(model.feature_importances_)
    })
    model_vars.to_csv(model_var_file,
                      encoding='utf8',
                      index=False)
    with open(pkl_file, 'wb') as f:
        pickle.dump(model, f)
"""
Author: Zhang Lu
Version: 1.0.0
Date:2025-02-14
Description: Happy Valentine's Day.
"""

import pandas as pd
import numpy as np
import toad
import matplotlib.pyplot as plt


def dist_calc(df_x_bin_temp, df_y, col):
    import re
    """
    判断违约率是否单调

    Arguments:
    df_bin:离散化后的DataFrame
    col:分析变量
    tgt_col:因变量

    Returns:
    iv:iv值
    mono_flag:0,不单调;1,单调递增;-1,单调递减
    df_temp:DataFrame,最终分布
    """
    df_y_ = pd.DataFrame(df_y)
    df_y_.columns = ['y']
    df_xy_bin_temp = pd.concat([df_x_bin_temp, df_y_], axis=1)
    df_temp = df_xy_bin_temp.groupby(col)['y'].agg(['count', 'sum']).reset_index().rename(columns={'sum': 'bad'})
    df_temp = df_temp.assign(
        good=df_temp['count'] - df_temp['bad'],
        d_all=df_temp['count'] / sum(df_temp['count']),
        d_bad=df_temp['bad'] / sum(df_temp['bad'])
    ).assign(
        d_good=lambda x: x['good'] / sum(x['good']),
        p_bad=lambda x: x['bad'] / sum(x['count']),
        p_good=lambda x: x['good'] / sum(x['count']),
        bad_rate=lambda x: x['bad'] / x['count']
    ).assign(
        bin_iv=lambda x: (x['d_bad'] - x['d_good']) *
                         np.log(x['d_bad'] / x['d_good']),
        bin_woe=lambda x: np.log(x['d_bad'] / x['d_good'])
    )

    iv = df_temp['bin_iv'].sum()
    # nan值排除分析
    bad_rate_arr = df_temp['bad_rate'].values
    bad_rate_arr = bad_rate_arr[:-1] if re.search('nan', df_temp[col].map(str).values[-1]) else bad_rate_arr

    arr_diff = np.diff(bad_rate_arr)
    if np.all(arr_diff >= 0):
        mono_flag = -1
    elif np.all(arr_diff <= 0):
        mono_flag = 1
    else:
        mono_flag = 0

    return iv, mono_flag, df_temp

def dist_combiner(df_x, df_y, col, new_bin):
    """
    计算指定bins的分布

    全局变量:
    c_temp:Combiner类，用于分箱转化
    train:DataFrame,原始建模数据集
    samp_col:样本类型列
    tgt_col:因变量列
    Arguments:
    col:指定的变量
    new_bin:指定的分箱规则
    """
    c_temp = toad.transform.Combiner()
    c_temp.set_rules({col: new_bin})
    df_x_bin_temp = c_temp.transform(df_x[[col]], labels=True)
    # 训练集计算分布
    return dist_calc(df_x_bin_temp, df_y, col)

def bins2mono(df_x, df_y, col, bins):
    import itertools
    """
    目标:最大化IV；约束:除nan外，各箱违约率单调

    全局变量:
    c_temp:Combiner类，用于分箱转化
    train:DataFrame,原始建模数据集
    samp_col:样本类型列
    tgt_col:因变量列

    Arguments:
    col:分析目标变量
    new_bin:当前的分箱规则
    """
    bin_cnt = len(bins)
    bin_arr = np.array(bins)
    bin_last = []
    # nan值排除分析
    if np.isnan(bins[-1]):
        bin_last = [np.nan]
        bin_cnt = bin_cnt - 1
        bin_arr = bin_arr[:-1]

    mono_flag = False

    while (bin_cnt + len(bin_last) > 1 and not mono_flag):
        bin_cnt -= 1
        bin_list = []
        flag = np.array([])
        iv = np.array([])

        for sub_bin in itertools.combinations(bin_arr, bin_cnt):
            sub_bin_list = list(sub_bin) + bin_last
            bin_list.append(sub_bin_list)
            # 指定bins的分布
            temp = dist_combiner(df_x, df_y, col, sub_bin_list)
            flag = np.append(flag, temp[1])
            iv = np.append(iv, temp[0])
            mono_flag = np.any(flag)
            if mono_flag:
                ix = iv[flag != 0].argmax()
                return np.array(bin_list)[flag != 0][ix]

def bin_set(df_x, df_y, all_cols, all_bins_update, show_plt=True):
    """
    单变量分箱更新，训练集fit分箱woe，并更新至全数据集上

    全局变量:
    c_temp:Combiner类，用于分箱转化
    t_temp:WOETransformer类，用于woe转化
    ds_all:DataFrame,原始建模数据集
    samp_col:样本类型列
    tgt_col:因变量列
    train_y:因变量值
    train_slc1_bin:DataFrame,粗分箱的离散化训练集
    df_new:DataFrame,更新的离散化数据集
    df_slc1_woe:DataFrame,更新的woe数据集

    Arguments:
    col:更新的变量
    new_bin:更新的分箱规则
    """
    c_temp = toad.transform.Combiner()
    t_temp = toad.transform.WOETransformer()
    c_temp.set_rules(all_bins_update)
    df_bin_update_x = c_temp.transform(df_x[all_cols], labels=True)
    # 计算新分箱woe
    t_temp.fit_transform(df_bin_update_x, df_y)
    df_woe_update_x = t_temp.transform(df_bin_update_x)
    return df_woe_update_x, c_temp, t_temp

def monolize(df_x, df_y, all_cols, all_bins):
    import itertools
    """
    批量更新分箱

    全局变量：同dist_combiner&bins2mono

    Arguments:
    all_cols:list,变量列表
    all_bins:dict,变量初始的分箱

    Returns:
    mono_list:list,单个分段点的变量列表
    """
    all_bins_update = all_bins.copy()
    for col in all_cols:
        if df_x[col].dtypes == 'object':
            continue
        rst_temp = dist_combiner(df_x, df_y, col, all_bins[col])
        mono_flag = rst_temp[1]
        if mono_flag == 0:
            new_bin = bins2mono(df_x, df_y, col, all_bins[col])
            print(all_bins[col])
            print(new_bin)
            all_bins_update[col] = new_bin
    print(all_bins_update)
    df_woe_update_x, c_temp, t_temp = bin_set(df_x, df_y, all_cols, all_bins_update)

    return df_woe_update_x, all_bins_update, c_temp, t_temp

def discrete_type(df_x,df_y,col,bin_set=True):
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    # 分箱:利用分箱阈值进行粗分箱
    c=toad.transform.Combiner()
    c.fit(df_x[col],df_y,method='chi',min_samples=0.05,empty_separate=True)
    # 导出箱的节点
    bins=c.export()
    df_bin_x=c.transform(df_x[col],labels=True)
    t=toad.transform.WOETransformer()
    t.fit_transform(df_bin_x,df_y)
    df_woe_x=t.transform(df_bin_x)
    if bin_set:
        df_woe_x_update,bins_update,c_update,t_update=monolize(df_x,df_y,col,bins)
        return df_woe_x_update,bins_update,c_update,t_update,df_woe_x,bins,c,t
    else:
        return df_woe_x,bins,c,t,df_woe_x,bins,c,t

"""***************
调整分箱结果图展示（函数）
**************"""
def plot_bin(df_bin, df_y, col, title=None, show_iv=True):
    """
    单变量样本分布及违约趋势图

    Params
    ------
    df_bin:离散化后的DataFrame
    col:分析变量
    tgt_col:因变量
    title:标题前置文本
    show_iv:显示iv值

    Returns
    ------
    matplotlib fig object
    """
    # matplotlib.use('nbAgg')
    # import matplotlib.pyplot as plt
    plt.style.use('default')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    temp = dist_calc(df_bin, df_y, col)
    iv = temp[0]
    dist_binx = temp[2]

    x_list = dist_binx[col]
    bad_rate = dist_binx['bad_rate']
    bin_dist = dist_binx['d_all']
    bin_count = dist_binx['count']
    good_dist = dist_binx['p_good']
    bad_dist = dist_binx['p_bad']

    y_right_max = np.ceil(bad_rate.max() * 10)
    if y_right_max % 2 == 1:
        y_right_max = y_right_max + 1
    if y_right_max - bad_rate.max() * 10 <= 0.3:
        y_right_max = y_right_max + 2

    y_right_max = y_right_max / 10
    if y_right_max > 1 or y_right_max <= 0 or y_right_max is np.nan or y_right_max is None:
        y_right_max = 1
    y_left_max = np.ceil(bin_dist.max() * 10) / 10
    if y_left_max > 1 or y_left_max <= 0 or y_left_max is np.nan or y_left_max is None:
        y_left_max = 1

    # title
    title_string = col + "  (iv:" + str(round(iv, 4)) + ")" if show_iv else col
    title_string = title + '-' + title_string if title is not None else title_string

    # param
    ind = np.arange(len(bad_rate))  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    ###### plot ######
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    # ax1
    p1 = ax1.bar(ind, good_dist, width, color=(24 / 254, 192 / 254, 196 / 254))
    p2 = ax1.bar(ind,
                 bad_dist,
                 width,
                 bottom=good_dist,
                 color=(246 / 254, 115 / 254, 109 / 254))
    for i in ind:
        ax1.text(i,
                 bin_dist[i] * 1.02,
                 str(round(bin_dist[i] * 100, 1)) + '%, ' + str(bin_count[i]),
                 ha='center')
    # ax2
    ax2.plot(ind, bad_rate, marker='o', color='blue')
    for i in ind:
        ax2.text(i,
                 bad_rate[i] * 1.02,
                 str(round(bad_rate[i] * 100, 1)) + '%',
                 color='blue',
                 ha='center')
    # settings
    ax1.set_ylabel('Bin count distribution')
    ax2.set_ylabel('Bad probability', color='blue')
    ax1.set_yticks(np.arange(0, y_left_max + 0.2, 0.2))
    ax2.set_yticks(np.arange(0, y_right_max + 0.2, 0.2))
    ax2.tick_params(axis='y', colors='blue')

    plt.xticks(ind, x_list)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha="right")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=40, ha="right")
    plt.title(title_string, loc='left')
    plt.legend((p2[0], p1[0]), ('bad', 'good'), loc='upper right')
    # fig.tight_layout()
    # show plot
    # plt.show()
    return fig
#调整分箱结果图展示
def bin_update_show_plt(df_x,df_y,c_update,c):
    train_slc1_bin=c.transform(df_x,labels=True)
    train_temp=c_update.transform(df_x,labels=True)
    for i in c:
        if (len(c[i])>0) and  (i not in list_dis):
            if (list(c[i][:-1]) if np.isnan(c[i][-1]) else list(c[i]))==(list(c_update[i][:-1]) if np.isnan(c_update[i][-1]) else list(c_update[i])):
                continue
            plot_bin(train_slc1_bin,df_y,i)
            plot_bin(train_temp,df_y,i)

"""***************
PSI,VIF 特征筛选（函数）for 罗辑回归
**************"""
#PSI计算
def get_psi(df_base, df_target, threshold):
    psi_df = toad.metrics.PSI(df_base, df_target).sort_values(key = 0).reset_index().rename(columns={'index': 'feature', 0: 'psi'}).assign(psi_flag=lambda x: x.psi < threshold)
    return psi_df

#VIF检验
def get_vif(df, threshold):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
# 区分变量中的 categorical变量，不计算vif，并临时处理nan为0，防止报错
    x = df.select_dtypes(include=[float, int])
    if x.empty:
        raise ValueError("No numerical variables detected in the dataframe.")
    x['c'] = 1
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0)
    name = x.columns
    vif_list = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
    vif = pd.DataFrame({'feature': name, 'vif': vif_list})        .query("feature!='c'")        .assign(vif_flag=lambda x: x.vif <= threshold)
    return vif

#汇总分析
def get_psi_vif(train_slc2_woe,test_slc2_woe,threshold1=0.01,threshold2=10):
    psi_df=get_psi(train_slc2_woe, test_slc2_woe, threshold1)
    vif_df=get_vif(train_slc2_woe,threshold2)
    vif_psi_df=vif_df.merge(psi_df,on='feature')
    return vif_psi_df

"""***************
变量描述（函数）
**************"""
def val_describe(df):
    x=df.describe()
    #x.to_csv(r'\\192.168.7.108\Ray\模型文档\风险模型\第三方支付建模\result1.csv',index=False)
    #1%
    y_1=df.quantile(0.01).reset_index().T
    y_1.columns=y_1.loc['index',:]
    y_1=y_1.drop('index')
    y_1.index=['1%']
    #10%
    y_10=df.quantile(0.1).reset_index().T
    y_10.columns=y_10.loc['index',:]
    y_10=y_10.drop('index')
    y_10.index=['10%']
    #90%
    y_90=df.quantile(0.9).reset_index().T
    y_90.columns=y_90.loc['index',:]
    y_90=y_90.drop('index')
    y_90.index=['90%']
    #99%
    y_99=df.quantile(0.99).reset_index().T
    y_99.columns=y_99.loc['index',:]
    y_99=y_99.drop('index')
    y_99.index=['99%']
    #type
    types=df.dtypes.reset_index().T
    types.columns=types.loc['index',:]
    types=types.drop('index')
    types.index=['type']

    list_val={}
    for i in df.columns:
        print(i)
        temp=df[i].value_counts().sort_values(ascending=False).reset_index()
        unique=temp[i].count()
        top1_name=''
        try:
            top1_name=temp.loc[0,'index']
        except:
            pass
        top1_val=0
        try:
            top1_val=temp.loc[0,i]
        except:
            pass
        top2_name=''
        try:
            top2_name=temp.loc[1,'index']
        except:
            pass
        top2_val=0
        try:
            top2_val=temp.loc[1,i]
        except:
            pass
        top3_name=''
        try:
            top3_name=temp.loc[2,'index']
        except:
            pass
        top3_val=0
        try:
            top3_val=temp.loc[2,i]
        except:
            pass
        top4_name=''
        try:
            top4_name=temp.loc[3,'index']
        except:
            pass
        top4_val=0
        try:
            top4_val=temp.loc[4,i]
        except:
            pass
        top5_name=''
        try:
            top5_name=temp.loc[4,'index']
        except:
            pass
        top5_val=0
        try:
            top5_val=temp.loc[4,i]
        except:
            pass
        list_val[i]=[unique,top1_name,top1_val,top2_name,top2_val,top3_name,top3_val,                            top4_name,top4_val,top5_name,top5_val]
    z=pd.DataFrame(list_val,index=['unique','top1_name','top1_val','top2_name','top2_val'                                   ,'top3_name','top3_val',                            'top4_name','top4_val','top5_name','top5_val'])

    final=pd.concat([x,y_1,y_10,y_90,y_99,types,z]).T
    final['size']=final['count'].max()
    final['missing']=final['count']/final['size']
    final=final.reset_index()
    final.rename(columns={'index':'col_name'},inplace=True)
    return final

def val_describe_tot(df,ext_list,output_file,y):
    df_describe=val_describe(df)
    tot_cols=[i for i in list(df_describe[df_describe.unique>1]['col_name']) if i not in ext_list]
    iv_result=toad.quality(df[tot_cols],target=y,iv_only=True,cpu_cores=16)
    iv_result.reset_index(inplace=True)
    iv_result.columns=['col_name','iv','gini','entropy','unique']
    df_describe_output=df_describe.merge(iv_result[['col_name','iv']],how='left',on=['col_name'])
    df_describe_output=df_describe_output[['col_name','iv','type','size','missing','unique','mean','std','min','1%','10%','25%'       ,'50%','75%','90%','99%','max','top1_name','top1_val','top2_name','top2_val'       ,'top3_name','top3_val','top4_name','top4_val','top5_name','top5_val']].sort_values(by=["iv"],ascending=False)
    #df_describe_output=df_describe[['col_name','type','iv',size','missing','unique','mean','std','min','1%','10%','25%'\
    #   ,'50%','75%','90%','99%','max','top1_name','top1_val','top2_name','top2_val'\
    #   ,'top3_name','top3_val','top4_name','top4_val','top5_name','top5_val']]
    df_describe_output.to_csv(output_file,index=False,encoding='gbk')
    return df_describe_output,tot_cols

"""***************
根据标签拆分样本
**************"""
def df_split(df,tgt_col):
    unique_tags = df[tgt_col].unique()
    split_dfs = {tag: df[df[tgt_col] == tag] for tag in unique_tags}
    return split_dfs

"""***************
训练，测试及外推样本选择（函数）
**************"""
def sample_select(df, y, vldt_ym=None, ym=None):
    from sklearn.model_selection import train_test_split
    if vldt_ym is None or ym is None:
        train_x, test_x, train_y, test_y = train_test_split(df.drop(y, axis=1), df[y], test_size=0.3,
                                                            random_state=100)
        train = pd.concat([train_x, train_y], axis=1)
        test = pd.concat([test_x, test_y], axis=1)
        train['samp_type'] = '01.train'
        test['samp_type'] = '02.test'
        ds_all = pd.concat([train, test], axis=0)
        return train, train_x, train_y, test, test_x, test_y, ds_all
    else:
        vldt=df[df[ym].isin(vldt_ym)]
        vldt_x=vldt.drop(y,axis=1)
        vldt_y=vldt[y]
        train_test=df[~df[ym].isin(vldt_ym)]
        train_x,test_x,train_y,test_y=train_test_split(train_test.drop(y,axis=1),train_test[y],test_size=0.3,random_state=100)
        train=pd.concat([train_x,train_y],axis=1)
        test=pd.concat([test_x,test_y],axis=1)
        train['samp_type']='01.train'
        test['samp_type']='02.test'
        vldt['samp_type']='03.vldt'
        ds_all = pd.concat([train, test, vldt], axis=0)
        return vldt,vldt_x,vldt_y,train,train_x,train_y,test,test_x,test_y,ds_all

"""***************
模型评价指标计算（KS,AUC,PSI)(函数)
**************"""

def model_verify(model, x, y, valx, valy, offx, offy, output_flag=False, bucket_num=10):
    from toad.metrics import KS, AUC, PSI
    """
    KS
    Args:
        score (array-like): list of score or probability that the model predict
        target (array-like): list of real target

    AUC
    Args:
        score (array-like): list of score or probability that the model predict
        target (array-like): list of real target
        return_curve (bool): if need return curve data for ROC plot

    PSI
    Args:
        test (array-like): data to test PSI
        base (array-like): base data for calculate PSI
        combiner (Combiner|list|dict): combiner to combine data
        return_frame (bool): if need to return frame of proportion
    """

    y_pred = model.predict_proba(x)[:, 1]
    val_y_pred = model.predict_proba(valx)[:, 1]
    off_y_pred = model.predict_proba(offx)[:, 1]

    train_ks = KS(y_pred, y)
    train_auc = AUC(y_pred, y)

    test_ks = KS(val_y_pred, valy)
    test_auc = AUC(val_y_pred, valy)

    vldt_ks = KS(off_y_pred, offy)
    vldt_auc = AUC(off_y_pred, offy)

    if output_flag:
        toad.KS_bucket(y_pred, y, bucket=bucket_num)[
            ['min', 'max', 'bads', 'goods', 'total', 'bad_rate', 'good_rate', 'odds', 'bad_prop', 'good_prop',
             'total_prop', 'cum_bads_prop', 'cum_goods_prop', 'cum_total_prop', 'ks']].to_csv('output/lr_train_ks.csv',
                                                                                              index=None, sep='\t')
        toad.KS_bucket(val_y_pred, valy, bucket=bucket_num)[
            ['min', 'max', 'bads', 'goods', 'total', 'bad_rate', 'good_rate', 'odds', 'bad_prop', 'good_prop',
             'total_prop', 'cum_bads_prop', 'cum_goods_prop', 'cum_total_prop', 'ks']].to_csv('output/lr_test_ks.csv',
                                                                                              index=None, sep='\t')
        toad.KS_bucket(off_y_pred, offy, bucket=bucket_num)[
            ['min', 'max', 'bads', 'goods', 'total', 'bad_rate', 'good_rate', 'odds', 'bad_prop', 'good_prop',
             'total_prop', 'cum_bads_prop', 'cum_goods_prop', 'cum_total_prop', 'ks']].to_csv('output/lr_vldt_ks.csv',
                                                                                              index=None, sep='\t')
        toad.KS_bucket(list(y_pred) + list(val_y_pred), list(y) + list(valy), bucket=bucket_num)[
            ['min', 'max', 'bads', 'goods', 'total', 'bad_rate', 'good_rate', 'odds', 'bad_prop', 'good_prop',
             'total_prop', 'cum_bads_prop', 'cum_goods_prop', 'cum_total_prop', 'ks']].to_csv('output/lr_dev_ks.csv',
                                                                                              index=None, sep='\t')

    psi_gap = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    train_test_psi = PSI(y_pred, val_y_pred, psi_gap)
    train_vldt_psi = PSI(y_pred, off_y_pred, psi_gap)
    traintest_vldt_psi = PSI(list(y_pred) + list(val_y_pred), off_y_pred, psi_gap)

    return train_ks, train_auc, test_ks, test_auc, vldt_ks, vldt_auc, train_test_psi, train_vldt_psi, traintest_vldt_psi

def rst_print(model, x, y, valx, valy, offx, offy):
    train_ks, train_auc, test_ks, test_auc, vldt_ks, vldt_auc, train_test_psi, train_vldt_psi, traintest_vldt_psi = model_verify(
        model, x, y, valx, valy, offx, offy)
    print('\n')
    print('训练集:')
    print('KS:', train_ks)
    print('AUC:', train_auc)
    print('测试集:')
    print('KS:', test_ks)
    print('AUC:', test_auc)
    print('验证集:')
    print('KS:', vldt_ks)
    print('AUC:', vldt_auc)
    print('\n')
    print('训练->测试,PSI:', train_test_psi)
    print('训练->验证,PSI:', train_vldt_psi)
    print('训练+测试->验证,PSI:', traintest_vldt_psi)
    return train_ks, train_auc, test_ks, test_auc, vldt_ks, vldt_auc, train_test_psi, train_vldt_psi, traintest_vldt_psi

"""***************
模型KS图，LIFT图
**************"""

def plot_lift(model, x, y, title, save_fig=False):
    import math
    y_pred = model.predict_proba(x)[:, 1]
    y_true = y
    df = pd.DataFrame({'pred': y_pred, 'bad': y_true}).sort_values(by='pred', ascending=False).reset_index(
        drop=True).assign(obs=lambda x: x.index + 1).assign(
        obs_pct=lambda x: (x['obs'] / x['obs'].max()).apply(lambda x: math.ceil(10 * x))).groupby('obs_pct').agg(
        {'bad': ['sum', 'count']})
    df.columns = ['bad_cnt', 'obs_cnt']
    df = df.assign(bad_rate=lambda x: x.bad_cnt / x.bad_cnt.sum()).assign(
        rand_rate=lambda x: x.obs_cnt / x.obs_cnt.sum())

    p1 = df[['bad_rate', 'rand_rate']].plot.bar(title=title)
    if save_fig:
        fig = p1.get_figure()
        fig.savefig(title + ".jpg", bbox_inches="tight")


def plot_ks(fpr, tpr, title, save_fig=False):
    df = pd.DataFrame({'fpr': fpr, 'tpr': tpr}).assign(ks=lambda x: x.tpr - x.fpr).reset_index().assign(
        index=lambda x: x.index / max(x.index)).set_index('index')

    ks = round(df['ks'].max(), 4)

    loc = df['ks'].values.argmax()
    p1 = df[['fpr', 'tpr', 'ks']].plot(title=title)
    p1.axvline(df.index[loc], linestyle='--')
    p1.text(df.index[loc] + 0.03, ks - 0.1, 'ks=%s' % ks, color='red')
    if save_fig:
        fig = p1.get_figure()
        fig.savefig(title + ".jpg", bbox_inches="tight")


def plot_all(model, x, y, valx, valy, offx, offy, save_fig=False):
    from sklearn.metrics import roc_curve
    y_pred = model.predict_proba(x)[:, 1]
    fpr_dev, tpr_dev, _ = roc_curve(y, y_pred, drop_intermediate=True)

    val_y_pred = model.predict_proba(valx)[:, 1]
    fpr_val, tpr_val, _ = roc_curve(valy, val_y_pred, drop_intermediate=True)

    off_y_pred = model.predict_proba(offx)[:, 1]
    fpr_off, tpr_off, _ = roc_curve(offy, off_y_pred, drop_intermediate=True)
    figure = plt.figure()
    plt.plot(fpr_dev, tpr_dev, label='dev')
    plt.plot(fpr_val, tpr_val, label='val')
    plt.plot(fpr_off, tpr_off, label='off')

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')

    if save_fig:
        plt.savefig("ROC_curve.jpg", bbox_inches="tight")

    plot_ks(fpr_dev, tpr_dev, 'KS Curve_train', save_fig)
    plot_ks(fpr_val, tpr_val, 'KS Curve_test', save_fig)
    plot_ks(fpr_off, tpr_off, 'KS Curve_vldt', save_fig)

    plot_lift(model, x, y, 'Lift_train', save_fig)
    plot_lift(model, valx, valy, 'Lift_test', save_fig)
    plot_lift(model, offx, offy, 'Lift_vldt', save_fig)

"""***************
模型不同组别单变量违约率
**************"""

def cross_calc(df_bin, by_col, col, tgt_col):
    import re
    """
    判断分组后每组的违约率趋势是否相同

    Arguments:
    df_bin:离散化后的DataFrame
    by_col:分组变量
    col:分析变量
    tgt_col:因变量

    """
    df_cross = (df_bin.groupby([by_col, col])[tgt_col].agg(['count', 'sum']).reset_index().rename(
        columns={'sum': 'bad'}).assign(bad_rate=lambda x: x['bad'] / x['count'])
                .drop(columns=['count', 'bad'])
                .pivot(index=by_col, columns=col, values='bad_rate'))

    # nan值排除分析
    if re.search('nan', df_cross.columns[-1]):
        temp = df_cross.iloc[:, :-1].values
    else:
        temp = df_cross.values

    a = np.all(np.diff(temp, axis=1) <= 0, axis=0)
    b = np.all(np.diff(temp, axis=1) >= 0, axis=0)

    return np.all(a | b), df_cross


def plot_badrate(df_bin, by_col, col, tgt_col):
    import matplotlib.style as psl
    psl.use('seaborn-ticks')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    temp = cross_calc(df_bin, by_col, col, tgt_col)[1]
    ax = temp.plot(title=col,
                   # figsize=(6.5, 4),
                   style=['*-', '^-', '.-', 'p-', 'o-', '*--', '^--', '-.', ':', '.', '-'])
    ax.set_ylabel('bad_rate')
    ax.legend(loc='best')
    return ax

# 每组违约率趋势检查
def cross_vars(df_bins, all_cols, by_col, tgt_col):
    """
    批量判断分组后每组的违约率趋势是否相同

    Arguments:
    df_bin:离散化后的DataFrame
    all_cols:分析变量列表
    by_col:分组变量
    tgt_col:因变量
    """
    cross_list = []
    for col in all_cols:
        temp = cross_calc(df_bins, by_col, col, tgt_col)
        cross_flag = temp[0]
        if not cross_flag:
            print(col)
            cross_list.append(col)
            plot_badrate(df_bins, by_col, col, tgt_col)
    return cross_list

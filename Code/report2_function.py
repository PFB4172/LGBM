import toad

def describe_report_tag(df, tag, tgt_col):
    data_all_tag = df[tag]
    tag_list = sorted(list(set(data_all_tag)))

    describe_df = pd.DataFrame(columns=['月份', '好样本数', '坏样本数', '总样本数', '好样本占比',
                                      '坏样本占比', '总样本月度占比'])
    y = df[tgt_col]

    # 分月循环
    for i in tag_list:
        subset = y.iloc[np.where(data_all_tag == i)]
        good = subset[subset == 0].count()
        bad = subset[subset == 1].count()
        total = subset.count()
        good_pct = good / total
        bad_pct = bad / total
        total_subset_pct = total / len(y)
        describe_df.loc[len(describe_df)] = [i, good, bad, total, good_pct, bad_pct, total_subset_pct]
    return describe_df

def describe_report_tag2(df, tag1, tag2, tgt_col):
    data_all_tag1 = df[tag1]
    tag_list1 = sorted(list(set(data_all_tag1)))

    describe_df = pd.DataFrame(columns=['标签一','月份', '好样本数', '坏样本数', '总样本数', '好样本占比',
                                      '坏样本占比', '总样本月度占比'])
    y = df[tgt_col]

    # 分月循环
    for i in tag_list1:
        data_all_tag2 = df.iloc[np.where(data_all_tag1==i)][tag2]
        tag_list2 = sorted(list(set(data_all_tag2)))
        for j in tag_list2:
            subset = y.iloc[np.where((data_all_tag1 == i) & (data_all_tag2 == j))]
            good = subset[subset == 0].count()
            bad = subset[subset == 1].count()
            total = subset.count()
            good_pct = good / total
            bad_pct = bad / total
            total_subset_pct = total / len(y)
            describe_df.loc[len(describe_df)] = [i, j, good, bad, total, good_pct, bad_pct, total_subset_pct]
    return describe_df

def bucket_distribution_tag(df, tag, y_pred, tgt_col, bench_mark, split_num, methods):
    y_pred = df[y_pred]
    y = df[tgt_col]

    # 标签月份明细list
    data_all_tag = df[tag]
    tag_list = sorted(list(set(data_all_tag)))

    # 设定 分段对benchmark，无benchmark则用全量的 等频，等距作为分段
    if bench_mark == '':
        y_pred_bench = y_pred
    else:
        y_pred_bench = y_pred[np.where(data_all_tag == bench_mark)]

    # 两种分段方式
    if methods == 'steps':
        cut_point, bin_names = split_without_tails(pd.Series(y_pred_bench), lower_quantile=0.05, upper_quantile=0.95,
                                                   split_num=split_num - 1)
    elif methods == 'quantile':
        cut_point, bin_names = split_equal_freq(y_pred_bench, split_num=split_num)
    elif methods == 'manual':
        cut_point, bin_names = split_manual_distance(465, 735, split_num)

    # 附上分组名称并创建空表格
    bads_table = pd.DataFrame(bin_names, columns=['bin_rule'])
    total_prop_table = pd.DataFrame(bin_names, columns=['bin_rule'])
    total_table = pd.DataFrame(bin_names, columns=['bin_rule'])
    lift_table = pd.DataFrame(bin_names, columns=['bin_rule'])

    # 分月循环
    for i in tag_list:
        print(i)
        y_pred_tmp = y_pred[np.where(data_all_tag == i)]
        A = y.iloc[np.where(data_all_tag == i)]
        y_tmp = A.values
        try:
            KS_bucket_table = toad.KS_bucket(y_pred_tmp, y_tmp, bucket=sorted(cut_point))
            # 统计bad个数
            bads_table_tmp = KS_bucket_table[['bads']]
            bads_table_tmp.rename(columns={'bads': i + '_bads'}, inplace=True)
            bads_table = pd.concat([bads_table, bads_table_tmp], axis=1)

            # 统计total_prop 区间样本占比
            total_prop_table_tmp = KS_bucket_table[['total_prop']]
            total_prop_table_tmp.rename(columns={'total_prop': i + '_total_prop'}, inplace=True)
            total_prop_table = pd.concat([total_prop_table, total_prop_table_tmp], axis=1)

            # 统计total个数
            total_table_tmp = KS_bucket_table[['total']]
            total_table_tmp.rename(columns={'total': i + '_total'}, inplace=True)
            total_table = pd.concat([total_table, total_table_tmp], axis=1)

            # 统计lift
            bad_r = KS_bucket_table['bads'].sum() / KS_bucket_table['total'].sum()
            lift_table_tmp = pd.DataFrame()
            lift_table_tmp[i + '_lift'] = KS_bucket_table['bad_rate'] / bad_r
            lift_table = pd.concat([lift_table, lift_table_tmp], axis=1)
        except:
            print(i + "此分类下无坏人")
    bads_table = bads_table.sort_index(ascending=False)
    total_prop_table = total_prop_table.sort_index(ascending=False)
    total_table = total_table.sort_index(ascending=False)
    lift_table = lift_table.sort_index(ascending=False)
    return bads_table, total_prop_table, total_table, lift_table

def bucket_lift_tag(df, tag, y_pred, tgt_col, split_num, methods = 'quantile'):
    from toad.metrics import KS, AUC, Lift

    def calculate_metrics(y_pred, y, split_num, methods):
        ks = KS(y_pred, y)
        auc = AUC(y_pred, y)
        bucket_table = toad.KS_bucket(y_pred, y, bucket=split_num, method=methods)[['total', 'bads', 'bad_rate']]
        bad = bucket_table['bads'].sum()
        total = bucket_table['total'].sum()
        bad_r = bad / total
        lift = list(bucket_table['bad_rate'] / bad_r)[-1]
        return total, bad, bad_r, ks, auc, lift

    y_pred = df[y_pred]
    y = df[tgt_col]

    result_table = pd.DataFrame(columns=['标签', '总样本数', '坏样本数', '坏样本率', 'KS', 'AUC', f'{100 / split_num}%_tail'])

    total, bad, bad_r, ks, auc, lift = calculate_metrics(y_pred, y, split_num, methods)
    result_table.loc[len(result_table)] = ['Total', total, bad, bad_r, ks, auc, lift]
    if tag != '':
        data_all_tag = df[tag]
        tag_list = sorted(list(set(data_all_tag)))
        for i in tag_list:
            y_pred_tmp = y_pred[np.where(data_all_tag == i)]
            y_tmp = y.iloc[np.where(data_all_tag == i)].values
            try:
                total, bad, bad_r, ks, auc, lift = calculate_metrics(y_pred_tmp, y_tmp, split_num, methods)
                result_table.loc[len(result_table)] = [i, total, bad, bad_r, ks, auc, lift]
            except:
                print(f"{i}此分类下有异常")
    return result_table

def loop_report_subgroup(df, tag, y_pred, group_column, my_dict):
    '''
    tag: 二级拆分标签。如：月份，城市等; 如果不需要，填空
    group_column: 如果组标签是单个变量，就填此变量group_column；
                        如果组标签是分开的变量(如变量G1，G2, G3)，则填空
    my_dict: 案例 {'G1':'y','G2':'y2','G3':'y3'}
    '''
    df_result = pd.DataFrame()
    for key,value in my_dict.items():
        print(key,value)
        if group_column == '':
            try:
                df_sub = df.iloc[np.where(df[key] == key)]
            except:
                print(f'{key} is not in the data, use element==1 to continue')
                df_sub = df.iloc[np.where(df[key] == 1)]
        else:
            df_sub = df.iloc[np.where(df[group_column] == key)]
        temp1 = Bucket_lift_Tag(df_sub, tag, y_pred, value, split_num=10, methods='quantile')
        temp2 = Bucket_lift_Tag(df_sub, tag, y_pred, value, split_num=20, methods='quantile')
        temp3 = pd.concat([temp1,temp2.iloc[:,5]],axis=1)
        temp3['sub_group'] = key
        df_result = pd.concat([df_result,temp3],axis=0)
    return df_result

def psi_ks_auc(df, tag, y_pred, tgt_col, bench_mark, split_num = 10):
    '''
    tag: 不能为空；拆分标签。如：月份，城市等。
    bench_mark: psi计算的benchmark。
                'last_row'表示每个tag中的最后一行作为benchmark；
                ''表示全量数据作为benchmark;
                其他值以tag中的一个项表示。
    '''
    result_table = pd.DataFrame(columns=['标签', 'benchmark', 'KS', 'AUC', 'PSI'])
    from toad.metrics import KS, AUC, PSI
    y_pred = df[y_pred]
    y = df[tgt_col]
    ks = KS(y_pred, y)
    auc = AUC(y_pred, y)
    psi = 0
    result_table.iloc[len(result_table)] = ['Total', 'Total', ks, auc, psi]
    '''get split rules'''
    cut_point, _ = split_equal_freq(y_pred, split_num=split_num)

    if bench_mark in ['', 'last_row']:
        y_pred_bench = y_pred
    else:
        y_pred_bench = y_pred[np.where(df[tag] == bench_mark)]

    tag_list = sorted(list(set(df[tag])))
    for i in tag_list:
        print(i)
        y_pred_tmp = y_pred[np.where(df[tag] == i)]
        y_tmp = y.iloc[np.where(df[tag] == i)].values
        try:
            ks = KS(y_pred_tmp, y_tmp)
            auc = AUC(y_pred_tmp, y_tmp)
            psi = PSI(y_pred_tmp, y_pred_bench, bucket=sorted(cut_point)) if bench_mark else 0
            if bench_mark == 'last_row':
                y_pred_bench = y_pred_tmp
            result_table.loc[len(result_table)] = [i, bench_mark, ks, auc, psi]
        except Exception as e:
            print(f"{i}此分类下有异常: {e}")
    return result_table
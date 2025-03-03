import File_utility_1
import Simple_Utility
import os
import Complex_Utility
import importlib
import toad.plot

importlib.reload(File_utility_1)
from File_utility_1 import *

importlib.reload(Simple_Utility)
from Simple_Utility import *

importlib.reload(Complex_Utility)
from Complex_Utility import *

"""***************
读取文件，写入excel
**************"""

def tot_vars_cal(wb,input_file,output_sheet):
    df=pd.read_csv(input_file,encoding='gbk')
    wb_sheet=wb.add_worksheet(output_sheet)
    row_flag=0
    col_flag=0
    write_table(wb_sheet,df,start_row=row_flag,start_col=col_flag)

"""***************
样本分布情况
**************"""

def sample_distribute(wb,ds_all,output_sheet,tgt_col,samp_type='samp_type',yearmonth='yearmonth'):
    samp_type_dict={'01.train':'训练集','02.test':'测试集','03.vldt':'外推集'}
    wb_sheet=wb.add_worksheet(output_sheet)
    col_add=0
    row_flag=-1
    for i,j in ds_all.groupby(samp_type):
        df=j
        T = df.shape[0]
        B = df[df[tgt_col]==1].shape[0]
        G = T-B
        temp = df.groupby([yearmonth,tgt_col])[samp_type].count().reset_index()
        temp[tgt_col] = temp[tgt_col].apply(lambda x: '好样本' if x==0 else '坏样本')
        temp = temp.pivot(index=yearmonth,columns=tgt_col,values=samp_type)
        temp['样本总数'] = temp['好样本']+temp['坏样本']
        temp['好样本比例'] = temp['好样本']/temp['样本总数']
        temp['好样本占总好样本比例'] = temp['好样本']/G
        temp['坏样本比例'] = temp['坏样本']/temp['样本总数']
        temp['坏样本占总坏样本比例'] = temp['坏样本']/B
        temp['占总样本比例'] = temp['样本总数']/T
        temp = temp.reset_index()
        temp = temp.sort_values(by=[yearmonth],ascending=False)
        temp2 = temp[[yearmonth,'好样本','好样本比例','好样本占总好样本比例','坏样本','坏样本比例','坏样本占总坏样本比例','样本总数','占总样本比例']].fillna(0).sort_values(by=yearmonth)
        col_add=temp2.shape[0]
        #####样本概况写入
        row_flag+=2
        wb_sheet.write(row_flag,0,"%s" %samp_type_dict[i],text_title_format)
        row_flag+=1
        temp2.rename(columns={yearmonth:"年月"},inplace=True)
        write_table(wb_sheet,temp2,start_row=row_flag,start_col=0,ch_col=[],str_col=[],int_col=[0,1,4,7],decimal_col=[],pct_col=[2,3,5,6,8])
        row_flag+=col_add

"""***************
入模变量概况
**************"""

def model_vars_analyse(wb,tgt_col,train_model_var,test_model_var,vldt_model_var,train_y,test_y,vldt_y,output_file,output_sheet):
    #入模变量VIF
    model_var_vif = get_vif(train_model_var,10)
    model_var_vif.rename(columns={"feature":"var"},inplace=True)
    model_var_vif = model_var_vif[["var","vif"]]
    model_var_vif
    #变量iv、gini、entropy、分箱个数
    devx_pre = pd.concat([train_model_var,train_y],axis=1)
    testx_pre = pd.concat([test_model_var,test_y],axis=1)
    ootx_pre = pd.concat([vldt_model_var,vldt_y],axis=1)
    all_set_pre = pd.concat([devx_pre,testx_pre,ootx_pre])

    devx_iv = toad.quality(devx_pre,target=tgt_col,iv_only=True)
    testx_iv = toad.quality(testx_pre,target=tgt_col,iv_only=True)
    ootx_iv = toad.quality(ootx_pre,target=tgt_col,iv_only=True)
    all_set_iv = toad.quality(all_set_pre,target=tgt_col,iv_only=False).reset_index()
    all_set_iv.rename(columns={"index":"var","iv":"all_set_iv"},inplace=True)

    devx_iv = devx_iv["iv"].reset_index()
    testx_iv = testx_iv["iv"].reset_index()
    ootx_iv = ootx_iv["iv"].reset_index()

    devx_iv.columns = ["var","train_iv"]
    testx_iv.columns = ["var","test_iv"]
    ootx_iv.columns = ["var","oot_iv"]

    model_var_iv = devx_iv.merge(testx_iv,on=["var"],how="left")
    model_var_iv = model_var_iv.merge(ootx_iv,on=["var"],how="left")
    model_var_iv = model_var_iv.merge(all_set_iv,on=["var"],how="left")
    model_var_iv
    #PSI
    model_var_psi_train_to_test = get_psi(train_model_var, test_model_var, 0.1)
    model_var_psi_train_to_oot = get_psi(train_model_var, vldt_model_var, 0.1)
    model_var_psi_train_to_test = model_var_psi_train_to_test[["feature","psi"]]
    model_var_psi_train_to_oot = model_var_psi_train_to_oot[["feature","psi"]]
    model_var_psi_train_to_test.rename(columns={"feature":"var","psi":"train_to_test_psi"},inplace=True)
    model_var_psi_train_to_oot.rename(columns={"feature":"var","psi":"train_to_oot_psi"},inplace=True)
    model_var_psi_train_to_oot
    # Non_missing_rate
    devx_rate = has_value_rate(train_model_var)
    testx_rate = has_value_rate(test_model_var)
    ootx_rate = has_value_rate(vldt_model_var)
    valued_rate = devx_rate.merge(testx_rate,on=["var"],how="left")
    valued_rate = valued_rate.merge(ootx_rate,on=["var"],how="left")
    valued_rate.columns = ["var","train_rate","test_rate","oot_rate"]
    #表格汇总输出
    model_var_stat = model_var_iv.merge(model_var_vif,on=["var"],how="left")
    model_var_stat = model_var_stat.merge(model_var_psi_train_to_test,on=["var"],how="left")
    model_var_stat = model_var_stat.merge(model_var_psi_train_to_oot,on=["var"],how="left")
    model_var_stat = model_var_stat.merge(valued_rate, on=["var"], how="left")
    model_var_stat.to_csv(output_file,index=False,sep=",",encoding="utf8")
    model_var_stat
    #文件汇总输出
    ws4 = wb.add_worksheet(output_sheet)
    model_var_stat = model_var_stat[["var","unique","train_iv","test_iv","oot_iv","all_set_iv","vif","gini","entropy","train_to_test_psi","train_to_oot_psi","train_rate","test_rate","oot_rate"]].sort_values(by=['train_iv'],ascending=False).reset_index(drop=True)
    model_var_stat.rename(columns={"var":"变量名","unique":"分箱数量","train_iv":"训练集iv","test_iv":"测试集iv","oot_iv":"外推集iv","all_set_iv":"全数据集iv","vif":"VIF","gini":"基尼系数","entropy":"信息熵","train_to_test_psi":"PSI(训练->测试)","train_to_oot_psi":"PSI(训练->外推)","train_rate":"训练-有值率","test_rate":"测试-有值率","oot_rate":"验证-有值率"},inplace=True)
    row_flag=0
    write_table(ws4,model_var_stat,start_row=row_flag,start_col=0,ch_col=[1],str_col=[0],int_col=[1],decimal_col=[2,3,4,5,6,7,8,9,10],pct_col=[11,12,13])
    return model_var_stat

"""***************
入模变量-单变量分析
**************"""

def model_vars_csv(woe,slc_cols,train_model_var,test_model_var,vldt_model_var,train_y,test_y,vldt_y,outputfile):
    #model_var_ks.csv
    ks_res=pd.DataFrame()
    for col in slc_cols:
        print('col:',col)
        temp = dict(zip(woe[col].values(),woe[col].keys()))
        print('temp:',temp)
        ks_temp_res = toad.KS_bucket(pd.concat([train_model_var[col],test_model_var[col],vldt_model_var[col]]),pd.concat([train_y,test_y,vldt_y]),bucket=sorted(list(set(train_model_var[col]))))
        print('bucket:',sorted(list(set(train_model_var[col]))))
        print('ks_temp_res:',ks_temp_res)
        ks_temp_res['bin'] = ks_temp_res['min'].apply(lambda x: temp[x])
        ks_temp_res['var'] = col
        ks_temp_res = ks_temp_res[['var','bin','bads','goods','total','bad_rate','good_rate','odds','bad_prop','good_prop','total_prop','cum_bads_prop','cum_goods_prop','cum_total_prop','ks']]
        ks_temp_res.sort_values(by='bin')
        ks_res=pd.concat([ks_res,ks_temp_res])
    ks_res=ks_res[['var','bin','bads','goods','total','bad_rate','good_rate','odds','bad_prop','good_prop'                       ,'total_prop','cum_bads_prop','cum_goods_prop','cum_total_prop','ks']]
    ks_res.columns=["变量名","分箱","坏样本量","好样本量","总样本量","坏样本率","好样本率","赔率","坏样本占全部坏样本比例"                        ,"好样本占全部好样本比例","组内样本量占比","累计坏样本率","累计好样本率","累计样本率","单变量ks"]
    return ks_res

def model_vars_sheet(wb,woe,slc_cols,train_model_var,test_model_var,vldt_model_var,train_y,test_y,vldt_y,outputfile,outputsheet):
    ks_res=model_vars_csv(woe,slc_cols,train_model_var,test_model_var,vldt_model_var,train_y,test_y,vldt_y,outputfile)
    ks_res.to_csv(outputfile,sep=",",index=False,encoding="utf8")
    wb_sheet = wb.add_worksheet(outputsheet)
    row_flag=0
    write_table(wb_sheet,ks_res,start_row=row_flag,start_col=0,ch_col=[0],str_col=[0,1],int_col=[2,3,4],decimal_col=[13],pct_col=[5,6,7,8,9,10,11,12])

"""***************
绘图 - 离散后数据分组绘图
**************"""

def bin_plt(df, col, samp_col, tgt_col, save_fig=False):
    """
    汇总绘图
    df:离散化后的DataFrame,包含训练、测试、验证
    """
    # %matplotlib inline
    for samp_tp in df[samp_col].drop_duplicates().tolist():
        df_part = df[df[samp_col] == samp_tp]
        p1 = plot_bin(df_part, df_part[tgt_col], col)
        if save_fig:
            # fig = p1.get_figure()
            p1.savefig(col + "_" + samp_tp + "_bins.jpg", bbox_inches="tight")
    # df_train = df[df[samp_col] == '01.train']
    # p1 = plot_bin(df_train, col, tgt_col)
    # def plot_badrate(df_bin, by_col, col, tgt_col):
    p2 = plot_badrate(df, samp_col, col, tgt_col)
    if save_fig:
        fig = p2.get_figure()
        fig.savefig(col + "_badrate.jpg", bbox_inches="tight")


def bin_plt_export(wb, train_model_var, df_slc2_woe_xy, output_sheet, model_var_stat,samp_col,tgt_col):
    for col in train_model_var.columns.tolist():
        bin_plt(df_slc2_woe_xy, col, samp_col, tgt_col, save_fig=True)

    ws6 = wb.add_worksheet(output_sheet)

    # 图片标准长宽
    std_width = 640
    std_high = 480
    # 缩放比例
    scale = 0.75

    # 表格标准长宽
    ws6.set_column(0, 50, 10)  # 设置列宽
    ws6.set_row(0, 20)  # 设置行高
    ws6.hide_gridlines(2)

    # 位置标识初始位置
    row_flag = 0
    col_flag1 = 1
    col_flag2 = 10

    # model_var_nm=pd.DataFrame({'var':train_model_var.columns,'name':pd.Series(train_model_var.columns)})
    string_pre = "变量释义："

    for i in range(len(model_var_stat)):
        var_nm = model_var_stat["变量名"][i]
        string = string_pre + var_nm
        ws6.write(row_flag, col_flag1, string, title_format)
        row_flag += 1
        ws6.write(row_flag, col_flag1, "训练集", title_format)
        im = var_nm + "_01.train_bins.jpg"
        row_flag += 1
        add_image(ws6, im, std_width, std_high, scale, row_flag, col_flag1)

        ws6.write(row_flag - 1, col_flag2, "测试集", title_format)
        im = var_nm + "_02.test_bins.jpg"
        add_image(ws6, im, std_width, std_high, scale, row_flag, col_flag2)

        ####调整图片位置
        row_flag += 19
        ws6.write(row_flag, col_flag1, "外推集", title_format)
        im = var_nm + "_03.vldt_bins.jpg"
        row_flag += 1
        add_image(ws6, im, std_width, std_high, scale, row_flag, col_flag1)

        ws6.write(row_flag - 1, col_flag2, "不同数据集上分箱表现", title_format)
        im = var_nm + "_badrate.jpg"
        add_image(ws6, im, std_width, std_high, scale, row_flag, col_flag2)

        row_flag += 20

    # wb.close()

    # 入模变量-变量相关性

"""***************
入模变量-变量相关性
**************"""

def model_var_corr(wb, train_model_var, output_sheet):
    # 图片标准长宽
    std_width = 640
    std_high = 480
    # 缩放比例
    scale = 0.75
    model_var_corr = train_model_var.corr().reset_index()
    model_var_corr.rename(columns={"index": "model_var"}, inplace=True)
    model_var_corr.to_csv("output/model_var_corr.csv", encoding="utf8", sep=",", index=False)
    model_var_corr

    corr_plot = toad.plot.corr_plot(train_model_var)
    fig = corr_plot.get_figure()
    fig.savefig("output/model_var_corr.jpg", bbox_inches="tight")

    ws9 = wb.add_worksheet(output_sheet)
    df = pd.read_csv("output/model_var_corr.csv")
    row_flag = 0
    col_flag = 0
    decimal_col = [x + 1 for x in range(len(df))]
    write_table(ws9, df, start_row=row_flag, start_col=col_flag, ch_col=[], str_col=[0], int_col=[],
                decimal_col=decimal_col, pct_col=[])

    row_flag += df.shape[0]

    row_flag += 2
    scale = 1.5
    im = "output/model_var_corr.jpg"
    add_image(ws9, im, std_width, std_high, scale, row_flag, col_flag)

    # 模型结果-模型评价指标

"""***************
模型结果-模型评价指标
**************"""

def model_result(wb, train_ks, test_ks, vldt_ks, train_auc, test_auc, vldt_auc, train_test_psi, train_vldt_psi,
                 traintest_vldt_psi, train_x, model, output_sheet):
    set_list = ["训练集", "测试集", "验证集"]
    ks_list = [train_ks, test_ks, vldt_ks]
    auc_list = [train_auc, test_auc, vldt_auc]
    model_ks_auc_df = pd.DataFrame({"DATASET": set_list, "KS": ks_list, "AUC": auc_list})

    set_list = ["训练集->测试集", "训练集->验证集", "训练集+测试集->验证集"]
    psi_list = [train_test_psi, train_vldt_psi, traintest_vldt_psi]
    model_psi_df = pd.DataFrame({"DATASET": set_list, "PSI": psi_list})

    print(model_ks_auc_df)
    print(model_psi_df)

    model_ks_auc_df.to_csv("output/model_ks_auc_df.csv", index=False, sep=",", encoding="utf8")
    model_psi_df.to_csv("output/model_psi_df.csv", index=False, sep=",", encoding="utf8")

    # 模型变量及系数导出
    coef = pd.DataFrame({'var': list(model.feature_name_), 'importance': list(model.feature_importances_)})
    coef = coef[["var", "importance"]]
    coef.to_csv("output/model_var_coef.csv", index=False)
    coef

    ws7 = wb.add_worksheet(output_sheet)

    df = pd.read_csv("output/model_ks_auc_df.csv")
    df.columns = ["数据集", "KS", "AUC"]
    table_decimal_format.set_num_format("0.0000")
    write_table(ws7, df, start_row=1, start_col=1, ch_col=[0], str_col=[0], int_col=[], decimal_col=[1, 2],
                pct_col=[])

    df = pd.read_csv("output/model_psi_df.csv")
    df.columns = ["数据集", "PSI"]
    write_table(ws7, df, start_row=6, start_col=1, ch_col=[0], str_col=[0], int_col=[], decimal_col=[1], pct_col=[])

    df = pd.read_csv("output/model_var_coef.csv")
    df.columns = ["变量", "重要性"]
    write_table(ws7, df, start_row=1, start_col=5, ch_col=[1], str_col=[0], int_col=[], decimal_col=[1], pct_col=[])

    title_list = ["ROC曲线", "训练集ROC曲线", "测试集ROC曲线", "外推集ROC曲线", "训练集lift", "测试集lift",
                  "验证集lift"]
    im_list = ["ROC_curve.jpg", "KS Curve_train.jpg", "KS Curve_test.jpg", "KS Curve_vldt.jpg", "Lift_train.jpg",
               "Lift_test.jpg", "Lift_vldt.jpg"]

    # 图片标准长宽
    std_width = 640
    std_high = 480
    # 缩放比例
    scale = 0.75

    row_flag = df.shape[0] + 3

    for i in range(len(title_list)):
        if i % 2 == 0:
            ws7.write(row_flag, 1, title_list[i], title_format)
            add_image(ws7, im_list[i], std_width, std_high, scale, row_flag + 1, 1)
        else:
            row_flag -= 19
            ws7.write(row_flag, 6, title_list[i], title_format)
            add_image(ws7, im_list[i], std_width, std_high, scale, row_flag + 1, 6)
        row_flag += 19

    row_flag += 1
    title_list = ["训练集", "测试集", "验证集"]
    df_list = ["output/lr_train_ks.csv", "output/lr_test_ks.csv", "output/lr_vldt_ks.csv"]
    for i in range(len(df_list)):
        df = pd.read_csv(df_list[i], sep="\t").reset_index()
        df.columns = ["分组序号", "分组最小值", "分组最大值", "组内坏样本量", "组内好样本量", "组内样本量",
                      "组内坏样本占比", "组内好样本占比", "赔率", "坏样本占全部坏样本比例",
                      "好样本占全部好样本比例", "组内样本量占比", "累计坏样本率", "累计好样本率", "累计样本率",
                      "ks"]
        ws7.write(row_flag, 1, title_list[i], title_format)
        row_flag += 1
        write_table(ws7, df, start_row=row_flag, start_col=1, ch_col=[], str_col=[], int_col=[0, 3, 4, 5],
                    decimal_col=[1, 2, 15], pct_col=[6, 7, 8, 9, 10, 11, 12, 13, 14], auto_adj=False)
        row_flag += df.shape[0] + 2

"""***************
模型结果-模型稳定性
**************"""

def crss_prd_vld(model, df_x, df_y, loc_var, baseline, stp_slc, tgt_col, out_flag=False, bucket_num=10):
    from toad.metrics import KS, AUC, PSI
    ym_ks = []
    ym_auc = []
    ym_psi = []
    # print('df_x:',df_x)
    # print('df_y:',df_y)
    ym_list = np.unique(df_x[loc_var])
    cut_bin = list(pd.qcut(list(baseline), 10, retbins=True, labels=False)[1])
    psi_gap = cut_bin[1:-1]

    for ym in ym_list:
        # x = df_x.loc[loc_var == ym, :][stp_slc]
        x = df_x[df_x[loc_var] == ym][stp_slc]
        y = df_y[df_y[loc_var] == ym][tgt_col]

        y_pred = model.predict_proba(x)[:, 1]
        y_pred_train = y_pred.copy

        ym_auc.append(AUC(y_pred, y))
        ym_ks.append(KS(y_pred, y))
        ym_psi.append(PSI(baseline, y_pred, psi_gap))
        if out_flag:
            toad.KS_bucket(y_pred, list(y), bucket=bucket_num)[
                ['min', 'max', 'bads', 'goods', 'total', 'bad_rate', 'good_rate', 'odds', 'bad_prop', 'good_prop',
                 'total_prop', 'cum_bads_prop', 'cum_goods_prop', 'cum_total_prop', 'ks']].to_csv(
                str(ym) + '_ks.csv', index=None, sep='\t')

    return pd.DataFrame({'ym_ks': ym_ks, 'ym_auc': ym_auc, 'ym_psi': ym_psi}, index=ym_list)


def model_stability(wb, lr_model, train_x, test_x, vldt_x, train_y, test_y, vldt_y, stp_slc, output_sheet,
                    yearmonth, tgt_col):
    # 跨周期分档
    #####以训练集为基准计算跨周期psi
    cross_ym_y_pred = lr_model.predict_proba(train_x.drop(yearmonth, axis=1))[:, 1]
    #####以某个数据集的某个年月为基准
    # ym_x = devx.loc[train_x["yearmonth"]==201902,:]
    # cross_ym_y_pred = lr_model.predict_proba(ym_x)[:, 1]
    print('测试集：')
    test_crss_prd_vld_df = crss_prd_vld(lr_model, test_x, test_y, yearmonth, cross_ym_y_pred, stp_slc, tgt_col,
                                        out_flag=True, bucket_num=10).reset_index()
    print(test_crss_prd_vld_df)
    test_crss_prd_vld_df.to_csv("output/test_crss_prd_vld_df.csv", index=False)
    print('验证集：')
    vldt_crss_prd_vld_df = crss_prd_vld(lr_model, vldt_x, vldt_y, yearmonth, cross_ym_y_pred, stp_slc, tgt_col,
                                        out_flag=True, bucket_num=10).reset_index()
    print(vldt_crss_prd_vld_df)
    vldt_crss_prd_vld_df.to_csv("output/vldt_crss_prd_vld_df.csv", index=False)
    ws8 = wb.add_worksheet(output_sheet)
    df1 = pd.read_csv("output/test_crss_prd_vld_df.csv")
    df2 = pd.read_csv("output/vldt_crss_prd_vld_df.csv")
    df1["dataset"] = "测试集"
    df2["dataset"] = "验证集"
    df = pd.concat([df1, df2])
    df = df[["index", "dataset", "ym_ks", "ym_auc", "ym_psi"]]
    df.columns = ["年月", "数据集", "ks", "auc", "psi"]

    row_flag = 0
    col_flag = 1

    ws8.write(row_flag, col_flag, "模型分年月效果", text_title_format)
    row_flag += 1
    write_table(ws8, df, start_row=row_flag, start_col=col_flag, ch_col=[1], str_col=[1], int_col=[0],
                decimal_col=[2, 3, 4], pct_col=[])
    row_flag += df.shape[0]
    row_flag += 1
    ws8.write(row_flag, col_flag, "注:该表中ks为根据p值按离散型计算,略高于十等分分箱计算ks", text_content_format)

    row_flag += 2
    ym_list = test_x[yearmonth].drop_duplicates().tolist() + vldt_x[yearmonth].drop_duplicates().tolist()
    ym_list.sort()
    for ym in ym_list:
        ws8.write(row_flag, col_flag, str(ym), text_content_format)
        row_flag += 1
        try:
            df = pd.read_csv(str(ym) + "_ks.csv", sep="\t")
            write_table(ws8, df, start_row=row_flag, start_col=col_flag, ch_col=[], str_col=[],
                        int_col=[0, 3, 4, 5], decimal_col=[1, 2, 15], pct_col=[6, 7, 8, 9, 10, 11, 12, 13, 14])
            row_flag += df.shape[0]
            row_flag += 2
        except:
            continue

def tree_model_auto_file(file_nm, model, train_x, test_x, vldt_x, train_y,
                         test_y, vldt_y, slc_col, samp_col):
    import pickle
    #模型部署所需文档保存
    if os.path.exists('result/%s' % file_nm):
        pass
    else:
        os.mkdir('result/%s' % file_nm)
    #原始入模变量，分箱后入模变量，WOE后入模变量，prob,score
    train_x[samp_col] = '01.train'
    test_x[samp_col] = '02.test'
    vldt_x[samp_col] = '03.vldt'
    init_x = pd.concat([train_x, test_x, vldt_x]).reset_index()
    init_y = pd.concat([train_y, test_y, vldt_y]).reset_index()
    prob = model.predict_proba(init_x[slc_col])[:, 1]
    result = pd.concat(
        [init_x[slc_col + [samp_col]], init_y,
         pd.Series(prob, name='prob')],
        axis=1)
    result.to_csv(r'result/%s/oral_model_vars.csv' % file_nm,
                  encoding='utf8',
                  index=False)
    model_vars = pd.DataFrame({
        'col_nm': slc_col,
        'importance': list(model.feature_importances_)
    })
    model_vars.to_csv(r'result/%s/model_vars.csv' % file_nm,
                      encoding='utf8',
                      index=False)
    with open(r'result/%s/model.pkl' % file_nm, 'wb') as f:
        pickle.dump(model, f)



def model_vars_csv2(woe,slc_cols,model_var,model_y):
    ks_res=pd.DataFrame()
    for col in slc_cols:
        print('col:',col)
        temp = dict(zip(woe[col].values(),woe[col].keys()))
        ks_temp_res = toad.KS_bucket(model_var[col],model_y,bucket=sorted(set(model_var[col])))
        ks_temp_res['WOE'] = np.log(ks_temp_res['bad_prop']/ks_temp_res['good_prop'])
        ks_temp_res['bin'] = ks_temp_res['min'].apply(lambda x: temp[x])
        ks_temp_res['var'] = col
        ks_temp_res = ks_temp_res[['var','bin','bads','goods','total','bad_rate','good_rate','odds','bad_prop','good_prop','total_prop','cum_bads_prop','cum_goods_prop','cum_total_prop','WOE','ks']]
        ks_temp_res.sort_values(by='bin')
        ks_res=pd.concat([ks_res,ks_temp_res])
    ks_res=ks_res[['var','bin','total','total_prop','bads','goods','bad_rate','WOE','ks']]
    ks_res.columns=["变量名","分箱","总样本量","组内样本量占比","坏样本量","好样本量","坏样本率","分组WOE","单变量ks"]
    return ks_res

def model_vars_sheet2(wb,woe,slc_cols,train_model_var,test_model_var,vldt_model_var,train_y,test_y,vldt_y,outputfile,outputsheet):
    train_ks_res=model_vars_csv2(woe,slc_cols,train_model_var,train_y)
    train_ks_res['样本类型']='训练集'
    test_ks_res=model_vars_csv2(woe,slc_cols,test_model_var,test_y)
    test_ks_res['样本类型']='测试集'
    vldt_ks_res=model_vars_csv2(woe,slc_cols,vldt_model_var,vldt_y)
    vldt_ks_res['样本类型']='验证集'
    ks_res = pd.merge(train_ks_res,test_ks_res,how='left',on=["变量名","分箱"],suffixes=('_train','_test'))
    ks_res = pd.merge(ks_res, vldt_ks_res, how='left', on=["变量名", "分箱"], suffixes=('_', '_oot'))
    # ks_res=pd.concat([train_ks_res,test_ks_res,vldt_ks_res])
    ks_res.to_csv(outputfile,sep=",",index=False,encoding="utf8")
    wb_sheet = wb.add_worksheet(outputsheet)
    row_flag=0
    write_table(wb_sheet,ks_res,start_row=row_flag,start_col=0,ch_col=[0],str_col=[0,1],int_col=[2,3,4],decimal_col=[13],pct_col=[5,6,7,8,9,10,11,12])

def bin_plt_export2(wb, train_model_var, df_slc2_woe_xy, output_sheet, model_var_stat, samp_col, tgt_col):
    for col in train_model_var.columns.tolist():
        bin_plt(df_slc2_woe_xy, col, samp_col, tgt_col, save_fig=True)

    ws6 = wb.add_worksheet(output_sheet)

    # 图片标准长宽
    std_width = 640
    std_high = 480
    # 缩放比例
    scale = 0.75

    # 表格标准长宽
    ws6.set_column(0, 50, 10)  # 设置列宽
    ws6.set_row(0, 20)  # 设置行高
    ws6.hide_gridlines(2)

    # 位置标识初始位置
    row_flag = 0
    col_flag1 = 1
    col_flag2 = 10

    # model_var_nm=pd.DataFrame({'var':train_model_var.columns,'name':pd.Series(train_model_var.columns)})
    string_pre = "变量释义："

    for i in range(len(model_var_stat)):
        var_nm = model_var_stat["变量名"][i]
        string = string_pre + var_nm
        ws6.write(row_flag, col_flag1, string, title_format)
        row_flag += 1
        ws6.write(row_flag, col_flag1, "训练集", title_format)
        im = var_nm + "_01.train_bins.jpg"
        row_flag += 1
        add_image(ws6, im, std_width, std_high, scale, row_flag, col_flag1)

        ws6.write(row_flag - 1, col_flag2, "测试集", title_format)
        im = var_nm + "_02.test_bins.jpg"
        add_image(ws6, im, std_width, std_high, scale, row_flag, col_flag2)

        ####调整图片位置
        row_flag += 19
        ws6.write(row_flag, col_flag1, "外推集", title_format)
        im = var_nm + "_03.vldt_bins.jpg"
        row_flag += 1
        add_image(ws6, im, std_width, std_high, scale, row_flag, col_flag1)

        ws6.write(row_flag - 1, col_flag2, "不同数据集上分箱表现", title_format)
        im = var_nm + "_badrate.jpg"
        add_image(ws6, im, std_width, std_high, scale, row_flag, col_flag2)

        row_flag += 20

def model_result_all(wb,model,file_nm,yearmonth,tgt_col,
                     train_x,train_y,
                     test_x,test_y,
                     vldt_x,vldt_y):
    model_var = model.feature_name_
    plot_all(model, train_x[model_var], train_y, test_x[model_var], test_y, vldt_x[model_var], vldt_y, save_fig=True)
    model_verify(model, train_x[model_var], train_y, test_x[model_var], test_y, vldt_x[model_var], vldt_y, output_flag=True,bucket_num=10)
    train_ks, test_ks, vldt_ks, train_auc, test_auc, vldt_auc, train_test_psi, train_vldt_psi, traintest_vldt_psi = rst_print(model, train_x[model_var], train_y, test_x[model_var], test_y, vldt_x[model_var], vldt_y)
    model_result(wb, train_ks, test_ks, vldt_ks, train_auc, test_auc, vldt_auc, train_test_psi, train_vldt_psi,traintest_vldt_psi, train_x[model_var], model, '%s_模型效果' % file_nm)
    model_stability(wb, model,
                    train_x[model_var+[yearmonth]],
                    test_x[model_var+[yearmonth]],
                    vldt_x[model_var+[yearmonth]],
                    pd.concat([train_y, train_x[[yearmonth]]], axis=1),
                    pd.concat([test_y, test_x[[yearmonth]]], axis=1),
                    pd.concat([vldt_y, vldt_x[[yearmonth]]], axis=1),
                    model_var, '%s_模型稳定性' % file_nm, yearmonth, tgt_col)
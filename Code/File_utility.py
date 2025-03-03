"""
Author: Zhang Lu
Version: 2.0.0
Date:2025-02-24
Description: Monday Blur
"""

import importlib
import xlsxwriter
import config
importlib.reload(config)
from config import *

#####新增一个workbook
wb = xlsxwriter.Workbook(report_file2,{'nan_inf_to_errors': True})

#####小标题格式
# 11号微软雅黑，加粗无边框，无自动换行，左对齐，黄色底
title_format = wb.add_format({
    "font_size": 11,  # 字体大小
    "bold": True,  # 是否粗体
    "font_name": "微软雅黑",  # 字体名称
    "font_color": "#000000",  # 字体颜色
    "bg_color": "#FFFF00",  # 背景颜色
    "align": "left",  # 居中对齐
    "valign": "vcenter",  # 垂直居中对齐
    "text_wrap": 0  # 自动换行
})

#####文本格式
# 11号微软雅黑，黑色加粗无边框，无自动换行，左对齐，无背景填充
text_title_format = wb.add_format({
    "font_size": 11,  # 字体大小
    "bold": True,  # 是否粗体
    "font_name": "微软雅黑",  # 字体名称
    "font_color": "#000000",  # 字体颜色
    # "bg_color":"#5B9BD5",        #背景颜色，默认是透明，RGB颜色编码和16进制颜色编码之间的转化有网页版可用
    "align": "left",  # 左对齐
    "valign": "vcenter",  # 垂直居中对齐
    "top": 0,  # 上边框
    "left": 0,  # 左边框
    "right": 0,  # 右边框
    "bottom": 0,  # 底边框
    "text_wrap": 0  # 自动换行
})

#####文本正文格式
# 11号微软雅黑，黑色无边框，无自动换行，左对齐，无背景填充
text_content_format = wb.add_format({
    "font_size": 11,  # 字体大小
    "bold": False,  # 是否粗体
    "font_name": "微软雅黑",  # 字体名称
    "font_color": "#000000",  # 字体颜色
    # "bg_color":"#5B9BD5",        #背景颜色，默认是透明，RGB颜色编码和16进制颜色编码之间的转化有网页版可用
    "align": "left",  # 左对齐
    "valign": "vcenter",  # 垂直居中对齐
    "top": 0,  # 上边框
    "left": 0,  # 左边框
    "right": 0,  # 右边框
    "bottom": 0,  # 底边框
    "text_wrap": 0  # 自动换行
})

#####表格标题格式
# 11号微软雅黑，蓝底加粗带边框，自动换行，居中，蓝色背景填充
table_title_format = wb.add_format({
    "font_size": 11,  # 字体大小
    "bold": True,  # 是否粗体
    "font_name": "微软雅黑",  # 字体名称
    "font_color": "#000000",  # 字体颜色
    "bg_color": "#5B9BD5",  # 背景颜色
    "align": "center",  # 居中对齐
    "valign": "vcenter",  # 垂直居中对齐
    "top": 1,  # 上边框
    "left": 1,  # 左边框
    "right": 1,  # 右边框
    "bottom": 1,  # 底边框
    "text_wrap": 1  # 自动换行
})

#####表格字符格式
# 11号微软雅黑，带边框，无自动换行，居中，无背景填充
table_str_format = wb.add_format({
    "font_size": 11,  # 字体大小
    "bold": False,  # 是否粗体
    "font_name": "微软雅黑",  # 字体名称
    "font_color": "#000000",  # 字体颜色
    # "bg_color":"#5B9BD5",        #背景颜色
    "align": "center",  # 居中对齐
    "valign": "vcenter",  # 垂直居中对齐
    "top": 1,  # 上边框
    "left": 1,  # 左边框
    "right": 1,  # 右边框
    "bottom": 1,  # 底边框
    "text_wrap": 0  # 自动换行
})

#####表格整数格式
# 11号微软雅黑，带边框，居中，无背景填充，无自动换行
table_int_format = wb.add_format({
    "font_size": 11,  # 字体大小
    "bold": False,  # 是否粗体
    "font_name": "微软雅黑",  # 字体名称
    "font_color": "#000000",  # 字体颜色
    # "bg_color":"#5B9BD5",        #背景颜色
    "align": "center",  # 居中对齐
    "valign": "vcenter",  # 垂直居中对齐
    "top": 1,  # 上边框
    "left": 1,  # 左边框
    "right": 1,  # 右边框
    "bottom": 1,  # 底边框
    "num_format": "0",  # 字符格式
    "text_wrap": 0  # 自动换行
})
# table_int_format.set_num_format("0")

#####表格小数格式
# 11号微软雅黑，带边框，居中，无背景填充，无自动换行
table_decimal_format = wb.add_format({
    "font_size": 11,  # 字体大小
    "bold": False,  # 是否粗体
    "font_name": "微软雅黑",  # 字体名称
    "font_color": "#000000",  # 字体颜色
    # "bg_color":"#5B9BD5",        #背景颜色
    "align": "center",  # 居中对齐
    "valign": "vcenter",  # 垂直居中对齐
    "top": 1,  # 上边框
    "left": 1,  # 左边框
    "right": 1,  # 右边框
    "bottom": 1,  # 底边框
    "num_format": "0.00",  # 字符格式
    "text_wrap": 0  # 自动换行
})
# table_decimal_format.set_num_format("0.00")

#####表格百分数格式
# 11号微软雅黑，带边框，居中，无背景填充，无自动换行
table_pct_format = wb.add_format({
    "font_size": 11,  # 字体大小
    "bold": False,  # 是否粗体
    "font_name": "微软雅黑",  # 字体名称
    "font_color": "#000000",  # 字体颜色
    # "bg_color":"#5B9BD5",        #背景颜色
    "align": "center",  # 居中对齐
    "valign": "vcenter",  # 垂直居中对齐
    "top": 1,  # 上边框
    "left": 1,  # 左边框
    "right": 1,  # 右边框
    "bottom": 1,  # 底边框
    "num_format": "0.00%",  # 字符格式
    "text_wrap": 0  # 自动换行train_model_var
})


def write_table(ws,
                df_input,
                start_row=0,
                start_col=0,
                ch_col=[],
                str_col=[],
                int_col=[],
                decimal_col=[],
                pct_col=[],
                title_format=table_title_format,
                str_format=table_str_format,
                int_format=table_int_format,
                decimal_format=table_decimal_format,
                pct_format=table_pct_format,
                auto_adj=True):
    # ws --- 写入的目标sheet
    # df --- 写入的df
    # start_row --- 表格起始行编号
    # start_col --- 表格起始列编号
    # ch_col --- 中文字段
    # str_col --- 字符型字段
    # int_col --- 整数型字段
    # decimal_col --- 小数型字段
    # pct_col --- 百分数型字段
    # title_format --- 表格抬头字段格式
    # str_format --- 字符型字段格式
    # int_format --- 整数型字段格式
    # decimal_format --- 小数型字段格式
    # pct_format --- 百分数型字段格式
    df = df_input.copy()
    col_list = df.columns.tolist()
    # 列名写入
    for i in range(len(col_list)):
        ws.write(start_row, start_col + i, col_list[i], title_format)

    # 表格内容写入
    start_row += 1
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            content = df.iloc[row][col]
            if pct_col and col in pct_col:
                ws.write(row + start_row, start_col + col, content,
                         pct_format)
            elif int_col and col in int_col:
                ws.write(row + start_row, start_col + col, content,
                         int_format)
            elif decimal_col and col in decimal_col:
                ws.write(row + start_row, start_col + col, content,
                         decimal_format)
            else:
                ws.write(row + start_row, start_col + col, content,
                         str_format)

    if auto_adj:
        col_list2 = [x + x for x in col_list]
        df.loc["new"] = col_list2
        for idx, col in enumerate(df):
            series = df[col]
            if ch_col and idx in ch_col:
                max_len = 2 * max((series.astype(str).apply(len).max(), len(col))) + 3
            else:
                max_len = max((series.astype(str).apply(len).max(), len(col))) + 3
            ws.set_column(start_col + idx, start_col + idx, max_len)
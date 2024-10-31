import pandas as pd
import numpy as np
from dateutil.parser import parse


def is_date(string, fuzzy=False):
    if isinstance(string, str):
        try:
            parse(string, fuzzy=fuzzy)
            return True
        except ValueError:
            return False
    return False


def detect_column_types(df):
    column_types = {}
    for column in df.columns:
        unique_values = df[column].dropna().unique()
        sample_values = unique_values[:min(len(unique_values), 100)]  # 取样本值来判断类型

        if all(is_date(val) for val in sample_values):
            column_types[column] = 'datetime'
        elif np.issubdtype(df[column].dtype, np.number):
            column_types[column] = 'numeric'
        elif df[column].nunique() / len(df[column]) < 0.05:  # 判断类别类型
            column_types[column] = 'categorical'
        elif len(unique_values) < 20:  # 判断有序类型（这是一个简单的启发式方法）
            column_types[column] = 'ordinal'
        else:
            column_types[column] = 'text'
    return column_types

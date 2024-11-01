# column_type_detector.py

import pandas as pd

def detect_column_types(df):
    """
    Detects whether each column in the DataFrame is a dimension or a measure.
    A dimension is non-numeric data (categorical), while a measure is numeric data.
    """
    column_types = {}
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            column_types[column] = '度量'
        else:
            column_types[column] = '维度'
    return column_types

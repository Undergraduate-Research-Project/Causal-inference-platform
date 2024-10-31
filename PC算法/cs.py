import pandas as pd
import chardet

# 自动检测文件编码
with open('result2.csv', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']
    print(f"Detected encoding: {encoding}")

# 使用检测到的编码读取文件
try:
    df = pd.read_csv('result2.csv', encoding=encoding)
    print("File content:")
    print(df.head())  # 打印文件前几行
except Exception as e:
    print(f"Error reading file: {e}")

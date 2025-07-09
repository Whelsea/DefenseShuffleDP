import pandas as pd

# 读取 CSV
df = pd.read_csv('./ontario-public-sector-salary-2016.csv')

# 去掉 "$" 和 ","，转为 float
df['Salary Paid'] = (
    df['Salary Paid']
    .replace('[\$,]', '', regex=True)
    .astype(float)
)

# 可选：保存新文件
df.to_csv('./data.csv', index=False)

print("✅ Salary Paid 列已成功转换为数值")

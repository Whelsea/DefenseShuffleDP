import pandas as pd

df = pd.read_csv('./ontario-public-sector-salary-2016.csv')

df['Salary Paid'] = (
    df['Salary Paid']
    .replace('[\$,]', '', regex=True)
    .astype(float)
)

df.to_csv('./data.csv', index=False)


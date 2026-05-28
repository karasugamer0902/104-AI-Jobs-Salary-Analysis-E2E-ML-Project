import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("ai_jobs_cleaned.csv")

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x='職缺所需學歷', y='salary_min', data=df, palette='Set3')
plt.title('AI 工程師：學歷與起薪 ( Min Salary ) 分佈')
plt.xlabel('學歷要求')
plt.ylabel('月薪 ( NTD )')

plt.subplot(1, 2, 2)
sns.histplot(df['salary_min'], bins=10, kde=True, color='skyblue')
plt.title('AI 工程師：起薪頻率分佈')
plt.xlabel('月薪 ( NTD )')
plt.ylabel('職缺數量')

plt.tight_layout()
plt.show()
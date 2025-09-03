import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

df = pd.read_csv('application_train.csv')
df.head()

df.info()
df.describe()

missing_ratio = df.isnull().sum() / len(df)
missing_ratio[missing_ratio > 0].sort_values(ascending=False)

sns.countplot(x='TARGET', data=df)
plt.title('Class Distribution of TARGET')
plt.show()

sns.histplot(df['AMT_INCOME_TOTAL'], bins=50, kde=True)
plt.show()

sns.countplot(x='TARGET', hue='CODE_GENDER', data=df)
plt.show()

sns.countplot(x='OCCUPATION_TYPE', hue='TARGET', data=df)
plt.xticks(rotation=90)
plt.show()

numeric_cols = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(12,10))
sns.heatmap(numeric_cols.corr(), cmap='coolwarm')
plt.show()

missing_df = df.isnull().sum() / len(df)
missing_df[missing_df > 0].sort_values(ascending=False).head(20).plot(kind='bar')
plt.show()

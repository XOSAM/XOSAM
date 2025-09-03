# 1. Import Libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import missingno as msno


# 2. Load Dataset

df = pd.read_csv("train.csv")

df.head()


# 3. Dataset Information

print("Dataset shape:", df.shape)

df.info()

df.describe()


# 4. Missing Values Analysis

msno.bar(df)

plt.show()



missing = df.isnull().sum()

missing = missing[missing > 0].sort_values(ascending=False)

missing_ratio = missing / len(df)

missing_df = pd.DataFrame({"Total": missing, "Missing_Ratio": missing_ratio})

missing_df


# Drop columns with >=5 missing values

df_clean = df.dropna(axis=1, thresh=len(df)-5)



# Drop rows with missing values

df_clean = df_clean.dropna()



print("Cleaned dataset shape:", df_clean.shape)


# 5. Target Variable Distribution

sns.histplot(df['SalePrice'], kde=True)

plt.title("Distribution of SalePrice")

plt.show()



print("Skewness:", df['SalePrice'].skew())

print("Kurtosis:", df['SalePrice'].kurt())


# Log Transformation

df['SalePrice_log'] = np.log1p(df['SalePrice'])



sns.histplot(df['SalePrice_log'], kde=True)

plt.title("Log-Transformed SalePrice")

plt.show()



print("Skewness (log):", df['SalePrice_log'].skew())

print("Kurtosis (log):", df['SalePrice_log'].kurt())


# 6. Correlation Analysis

corr = df.corr()

plt.figure(figsize=(16,10))

sns.heatmap(corr, cmap="coolwarm", center=0)

plt.title("Correlation Heatmap")

plt.show()


# Top 10 correlated features with SalePrice

top_corr = corr['SalePrice'].sort_values(ascending=False).head(11)

print(top_corr)



top_features = top_corr.index

plt.figure(figsize=(10,8))

sns.heatmap(df[top_features].corr(), annot=True, cmap="coolwarm", center=0)

plt.title("Top 10 Features Correlated with SalePrice")

plt.show()


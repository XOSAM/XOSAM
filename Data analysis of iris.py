import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import os

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='Species')
df = pd.concat([X, y], axis=1)

print(df.iloc[3])
print(df['Species'].value_counts())
print(df.isnull().sum())
print(df.describe())

col1 = df['sepal width (cm)']
col2 = df.iloc[:,1]
subset_rows = df.iloc[50:100]
subset_petal_length = df.iloc[50:100]['petal length (cm)']
petal_width_02 = df[df['petal width (cm)'] == 0.2]

output_dir = "iris_plots"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(5,5))
plt.pie(df['Species'].value_counts(), labels=['setosa','versicolor','virginica'], autopct='%1.1f%%')
plt.title('Species Distribution')
plt.savefig(f"{output_dir}/species_pie.png")
plt.close()

for feature in X.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Species', y=feature, data=df)
    plt.title(f'Boxplot of {feature}')
    plt.savefig(f"{output_dir}/boxplot_{feature.replace(' ', '_')}.png")
    plt.close()

for feature in X.columns:
    plt.figure(figsize=(6,4))
    sns.violinplot(x='Species', y=feature, data=df)
    plt.title(f'Violin plot of {feature}')
    plt.savefig(f"{output_dir}/violin_{feature.replace(' ', '_')}.png")
    plt.close()

features = X.columns
for i in range(len(features)):
    for j in range(i+1, len(features)):
        plt.figure(figsize=(5,4))
        sns.scatterplot(x=features[i], y=features[j], hue='Species', data=df, palette='Set1')
        plt.title(f'{features[i]} vs {features[j]}')
        plt.savefig(f"{output_dir}/scatter_{features[i].replace(' ', '_')}_vs_{features[j].replace(' ', '_')}.png")
        plt.close()

sns.pairplot(df, hue='Species', palette='Set1')
plt.savefig(f"{output_dir}/pairplot.png")
plt.close()

corr = df.iloc[:,:4].corr()
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.savefig(f"{output_dir}/correlation_heatmap.png")
plt.close()

print(corr)
      

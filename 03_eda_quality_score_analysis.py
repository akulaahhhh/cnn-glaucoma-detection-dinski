#EDA - Quality Score Analysis
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

df = pd.read_csv("Labels.csv")

#Check average quality score
print("\nAverage quality score")
print(df["Quality Score"].mean())

#Check average quality score by label
print("\nAverage quality score by label")
print(df.groupby("Label")["Quality Score"].mean())

#Distribution plot of quality scores for GON+ and GON-
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Quality Score', hue='Label', kde=True, bins=30)
plt.title('Distribution of Image Quality Scores by Label')
plt.xlabel('Quality Score')
plt.ylabel('Count of Images')
plt.show()

#EDA - Data Quality
import os
import pandas as pd

df = pd.read_csv("Labels.csv")

print("\n--- DATA QUALITY CHECKS ---")

#Check for missing values
print("\nMissing values per column")
print(df.isnull().sum())

#Check for duplicates
print("\nDuplicate rows")
print(df.duplicated().sum())

#Check data types
print("\nData types")
print(df.dtypes)

#Check unique values in Label column
print("\nUnique values in Label column")
print(df["Label"].unique())

#Check value counts in Label column
print("\nValue counts in Label column")
print(df["Label"].value_counts())


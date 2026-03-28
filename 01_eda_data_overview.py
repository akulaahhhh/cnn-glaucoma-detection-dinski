#EDA - Data Overview
import os
import pandas as pd

df = pd.read_csv("Labels.csv")

print("--- BASIC STRUCTURE ---")

#View first 5 rows
print("First 5 rows")
print(df.head())

#View dataset info
print("\nDataset Info")
print(df.info())

#View statistical summary
print("\nStatistical Summary")
print(df.describe())
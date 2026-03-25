import os
import pandas as pd

df = pd.read_csv("Labels.csv")

print("---DATA CLEANING---")

#Remove unnamed:4 column
print("Remove Unnamed: 4 column")
if "Unnamed: 4" in df.columns:
    df = df.drop(columns=["Unnamed: 4"])
print(df.head())

#Encode Labels column
print("Encode Labels column")
label_map = {
    "GON+" : 1,
    "GON-" : 0
}
df["label_numeric"] = df["Label"].map(label_map)
print(df.head())

#Remove rows with quality score less than 3.8
print("Remove rows with quality score less than 3.8")
#Total rows removed by label
dfRemoved = df[df["Quality Score"] < 3.8]
print("Total rows removed by label")
print(dfRemoved["Label"].value_counts())
df = df[df["Quality Score"] >= 3.8]
print(df.head())

#Save cleaned data
df.to_csv("Labels_cleaned.csv", index=False)
print("Cleaned data saved to Labels_cleaned.csv")
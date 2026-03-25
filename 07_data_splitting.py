from pandas.core.common import random_state
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("Labels_cleaned.csv")

#Get unique patients id
patients = df["Patient"].unique()

#Split patient into train and test set
train_patients, test_patients = train_test_split(
    patients,
    test_size = 0.2,
    random_state = 42
)

#Select images from the patient list
train_df = df[df["Patient"].isin(train_patients)]
test_df = df[df["Patient"].isin(test_patients)]
print("Training images:", len(train_df))
print("Testing images:", len(test_df))
print("Training patients:", train_df["Patient"].nunique())
print("Testing patients:", test_df["Patient"].nunique())

train_df.to_csv("train_dataset.csv", index=False)
test_df.to_csv("test_dataset.csv", index=False)
print("Dataset Splitting Completed!")
import os 
import pandas as pd 

train_df = pd.read_csv("train_dataset.csv")
test_df = pd.read_csv("test_dataset.csv")

print("TRAINING DATASET")
print("Training images:", len(train_df))
print("Training GON-:", train_df[train_df["label_numeric"] == 0].shape[0])
print("Training GON+:", train_df[train_df["label_numeric"] == 1].shape[0])
print("GON+ percentage:", train_df[train_df["label_numeric"] == 1].shape[0] / len(train_df) * 100)

print("\nTESTING DATASET")
print("Testing images:", len(test_df))
print("Testing GON-:", test_df[test_df["label_numeric"] == 0].shape[0])
print("Testing GON+:", test_df[test_df["label_numeric"] == 1].shape[0])
print("GON+ percentage:", test_df[test_df["label_numeric"] == 1].shape[0] / len(test_df) * 100)

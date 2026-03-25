#EDA - Image Analysis
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
# import seaborn as sns

df = pd.read_csv("Labels.csv")

print("\nImage Analysis")

#create image path
image_folder = "Images"
df["image_path"] = df["Image Name"].apply(lambda x: os.path.join(image_folder,x))

# Check if image exists
df["file_exists"] = df["image_path"].apply(os.path.exists)#check the path
print("Image file check summary:\n")
print(df["file_exists"].value_counts())

# Check image size for 20 random images
image_sizes = []
for img_path in df["image_path"].sample(20):
    if os.path.exists(img_path):
        img = Image.open(img_path)
        image_sizes.append(img.size)
    else:
        image_sizes.append("Missing")
print("\nImage size for 20 random images")
print(image_sizes)



from tensorflow.keras import models
import os
import pandas as pd
model = models.load_model("glaucoma_model_dinski.h5")
# test_df = pd.read_csv("test_dataset.csv")
# image_folder = "Images_Processed"
print(model.summary())
# img_path = os.path.join(image_folder, test_df["Image Name"])
# print(img_path)
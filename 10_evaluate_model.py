import pandas as pd 
import numpy as np 
import os 
from PIL import Image 
from tensorflow.keras import models
import seaborn as sb 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, classification_report

#Load trained model
model = models.load_model("glaucoma_model_dinski.h5")

#Load test dataset
test_df = pd.read_csv("test_dataset.csv")
image_folder = "Images_Processed"

#Prepare test dataset
x_test, y_test = [], []
for i, row in test_df.iterrows():
    img_path = os.path.join(image_folder, row["Image Name"])
    img = Image.open(img_path)
    img_array = np.array(img) / 255.0
    x_test.append(img_array)
    y_test.append(row["label_numeric"])
x_test = np.array(x_test)
y_test = np.array(y_test)

#Generate Predictions
predictions = model.predict(x_test)
prediction_labels = (predictions>0.5).astype(int)

#Confusion Matrix
cm = confusion_matrix(y_test,prediction_labels, labels=[0,1])
print("Confusion Matrix")
print(cm)

#Classification Report
cr = classification_report(y_test, prediction_labels, labels=[0,1], target_names=["GON-", "GON+"])
print("Classification Report")
print(cr)

#Visualize Confusion Matrix
plt.figure(figsize=(8,6))
sb.heatmap(
    cm,
    annot = True,
    cmap="Blues",
    fmt="d"
)
plt.title("Glaucoma Detection Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.show()
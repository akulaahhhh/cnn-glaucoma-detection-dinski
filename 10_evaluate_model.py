import pandas as pd 
import numpy as np 
import os 
from PIL import Image 
from tensorflow.keras import models
# import tensorflow as tf
# import keras
import seaborn as sb 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, classification_report

#Load trained model
model = models.load_model("glaucoma_model_dinski.h5")

#Load test dataset
test_df = pd.read_csv("test_dataset.csv")
image_folder = "Images_Processed"

#-------------GRAD-CAM ALGORITHM----------------
model_builder = keras.applications.xception.Xception(weights="imagenet", include_top=False)
preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions
last_conv_layer_name = "conv2d_2"
img_size = (224,224)
#the local path to get our 10 target image
# img_paths = []
# for i in range(10):
#     img_paths.append(os.path.join(image_folder, test_df["Image Name"][i]))
img_path = os.path.join(image_folder, test_df["Image Name"][0])

def get_img_array(img_path,size=(224,224)):
    img = keras.utils.load_img(img_path, target_size = size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model,last_conv_layer_name, pred_index =None):
    #Create a model that maps the input images to the activations of the last conv layer
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    #Then we compute the gradient of the top predicted class for out input image with respective to our last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

#TEST
# Prepare image
img_array = preprocess_input(get_img_array(img_path, size=(224,224)))

# Make model
model = model_builder(weights="imagenet")

# Remove last layer's softmax
model.layers[-1].activation = None

# Print what the top predicted class is
preds = model.predict(img_array)
print("Predicted:", decode_predictions(preds, top=1)[0])

# Generate class activation heatmap
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

# Display heatmap
plt.matshow(heatmap)
plt.show()

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
cr = classification_report(y_test, prediction_labels, labels=[0,1], target_names=["GON-", "GON+"], output_dict=True)
print("Classification Report")
print(cr)
#Visualize Classification Report in Heatmap
report_df = pd.DataFrame(cr).transpose()
report_plot = report_df.iloc[:-3,:-1]
plt.figure(figsize=(8, 6))
sb.heatmap(report_plot, annot=True, cmap='Blues', fmt=".2f", linewidths=.5)
plt.title('Classification Report Heatmap')
plt.tight_layout()
plt.show()

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
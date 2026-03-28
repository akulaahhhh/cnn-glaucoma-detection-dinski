import os
import pandas as pd 
import keras
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
import matplotlib.cm as mpl 


#Load test dataset
test_df = pd.read_csv("test_dataset.csv")
image_folder = "Images_Processed"

#-------------GRAD-CAM ALGORITHM----------------
model = models.load_model("glaucoma_model_dinski.h5")
last_conv_layer_name = "conv2d_2"

img_path = os.path.join(image_folder, test_df["Image Name"].sample(1).iloc[0])

def get_img_array(img_path,size=(224,224)):
    img = keras.utils.load_img(img_path, target_size = size)
    array = keras.utils.img_to_array(img)
    array = array / 255.0
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model,last_conv_layer_name, pred_index =None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_model = keras.models.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations to the final predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    layer_names = [layer.name for layer in model.layers]
    start_idx = layer_names.index(last_conv_layer_name) + 1
    for layer in model.layers[start_idx:]:
        x = layer(x)
    classifier_model = keras.models.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        # Handle binary classification
        if preds.shape[-1] == 1:
            class_channel = preds[:, 0]
        else:
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

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # 8. Display Grad CAM
    plt.imshow(superimposed_img)
    plt.axis('off') # Hides the axes for a cleaner look
    plt.title("Grad-CAM: Glaucoma Detection")
    plt.show()




#TEST
# Prepare image
img_array = get_img_array(img_path)

# Remove last layer's softmax
# model.layers[-1].activation = None

# Print what the top predicted class is
preds = model.predict(img_array)
print("Predicted Score:", preds[0])

# Generate class activation heatmap
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
print(img_path)
# Display heatmap
plt.matshow(heatmap)
plt.show()


save_and_display_gradcam(img_path, heatmap)
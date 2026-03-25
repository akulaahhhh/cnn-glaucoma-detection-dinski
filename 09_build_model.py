import pandas as pd 
import os 
import numpy as np 
from tensorflow.keras import layers, models
# from sklearn.model_selection import StratifiedGroupKFold
from keras.callbacks import EarlyStopping
from PIL import Image

# Load datasets
train_df = pd.read_csv("train_dataset.csv")
test_df = pd.read_csv("test_dataset.csv")
image_folder = "Images_Processed"

#Prepare training data
x_train, y_train = [], []
for i, row in train_df.iterrows():
    img_path = os.path.join(image_folder, row["Image Name"])
    img = Image.open(img_path)
    img = np.array(img) / 255.0
    x_train.append(img)
    y_train.append(row["label_numeric"])

x_test, y_test = [], []
for i, row in test_df.iterrows():
    img_path = os.path.join(image_folder, row["Image Name"])
    img = Image.open(img_path)
    img = np.array(img) / 255.0
    x_test.append(img)
    y_test.append(row["label_numeric"])

# Convert lists to numpy arrays for faster feeding into Keras
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Generate data augmentation
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    # I commented out Rescaling here because you already rescaled by 
    # doing `np.array(img) / 255.0` in the loops above! Doing both would break it :)
    # layers.Rescaling(scale=1./255) 
])

# 1. Create model structure
def build_model():
    model = models.Sequential([
        data_augmentation,
        layers.Conv2D(32,(3,3), activation="relu", input_shape = (224,224,3)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

# Compile Model
model = build_model()
# You forgot to compile the model! This connects your optimizer and loss function.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Implement Early Stopping
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    verbose=1,
    mode = "min",
    restore_best_weights = True
)

# Train Model
history = model.fit(
    x_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_accuracy)
 
# Save model
model.save("glaucoma_model_dinski.h5")
print("Model saved successfully.")
   

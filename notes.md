# Glaucoma Detection Progress

## 1. Data Cleaning & Preparation
- [x] Remove `Unnamed: 4` column
- [x] Encode Labels column (`GON+` -> `1`, `GON-` -> `0`)
- [x] Clean Data: Remove images with a quality score below `3.8`

## 2. Image Processing
- [x] Enhance images (fix lighting and contrast using CLAHE)
- [x] Resize images to `224x224`
- [x] Save processed images to `Images_Processed` folder
- [x] Convert images to numpy arrays
- [x] Normalize image pixels to `0-1` range

## 3. Modeling & Evaluation
- [x] Split data into training and testing sets
- [x] Build a CNN model
- [x] Train the model
- [x] Evaluate the model
- [x] Save the trained model

## 4. Pending Tasks / Next Steps
- [ ] Handle Class Imbalance
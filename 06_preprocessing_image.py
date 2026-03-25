import os
import pandas as pd 
from PIL import Image
import cv2 as cv


df = pd.read_csv("Labels_cleaned.csv")

image_folder = "Images"
output_folder = "Images_Processed"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("Start Image Processing~")


for i, row in df.iterrows():
    img_name = row["Image Name"]
    image_path = os.path.join(image_folder,img_name)
    save_path = os.path.join(output_folder, img_name)

    #Read Image
    img = cv.imread(image_path)
    #Resize Image
    img = cv.resize(img, (224,224))
    

    #Image Enhancement using CLAHE to improve contrast
    #Convert image to LAB color space
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    #Split LAB image into L, A, B channels
    l_channel, a, b = cv.split(lab)
    #Apply CLAHE to L channel
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl_l = clahe.apply(l_channel)
    #Merge CLAHE enhanced L channel with A and B channels
    cl_lab = cv.merge((cl_l, a, b))
    #Convert LAB image back to BGR color space
    cl_image = cv.cvtColor(cl_lab, cv.COLOR_LAB2BGR)
    #Save processed image
    cv.imwrite(save_path, cl_image)
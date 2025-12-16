import os
import cv2
import numpy as np

def preprocess_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for subfolder in os.listdir(input_dir):
        subfolder_path = os.path.join(input_dir, subfolder)
        output_subfolder = os.path.join(output_dir, subfolder)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        for filename in os.listdir(subfolder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(subfolder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (224, 224))
                output_path = os.path.join(output_subfolder, filename)
                cv2.imwrite(output_path, img)

# Process all dataset folders
dataset_folders = [
    'images_E_S_SB_69x69_a_03_train',
    'images_E_S_SB_69x69_a_03_test',
    'images_E_S_SB_227x227_a_03_train',
    'images_E_S_SB_227x227_a_03_test',
    'images_E_S_SB_299x299_a_03_train',
    'images_E_S_SB_299x299_a_03_test'
]

for folder in dataset_folders:
    preprocess_images(os.path.join('data', folder), os.path.join('data', folder + '_processed'))
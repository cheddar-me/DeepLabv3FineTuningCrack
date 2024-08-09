import cv2
import numpy as np
import os
from pathlib import Path

# Function for Contrast Enhancement
def apply_contrast_enhancement(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)

# Function for Noise Reduction
def apply_noise_reduction(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

# Function for Edge Enhancement
def apply_edge_enhancement(image):
    return cv2.Canny(image, 100, 200)

# Function for Color Space Transformation
def apply_color_space_transformation(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

# Function to process images
def process_images(src_folder, dest_folder):
    Path(dest_folder).mkdir(parents=True, exist_ok=True)
    
    for folder in ['Train', 'Test']:
        src_image_path = Path(src_folder) / folder / 'Image'
        dest_folder_path = Path(dest_folder) / folder / 'Image'
        
        Path(dest_folder_path).mkdir(parents=True, exist_ok=True)
        
        for img_name in os.listdir(src_image_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = src_image_path / img_name
                image = cv2.imread(str(img_path))

                # Apply all processing steps
                contrast_img = apply_contrast_enhancement(image)
                noise_reduced_img = apply_noise_reduction(contrast_img)
                edge_enhanced_img = apply_edge_enhancement(noise_reduced_img)
                final_img = apply_color_space_transformation(edge_enhanced_img)

                # Save only the final image
                cv2.imwrite(str(dest_folder_path / img_name), final_img)

if __name__ == "__main__":
    src_folder = 'sample_dataset_ids'
    dest_folder = 'sample_dataset_ids_new'

    process_images(src_folder, dest_folder)
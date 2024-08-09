import cv2
import numpy as np
import os
from pathlib import Path

def apply_contrast_enhancement(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)

def apply_noise_reduction(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def apply_edge_enhancement(image):
    return cv2.Canny(image, 100, 200)

def apply_color_space_transformation(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

def process_images(src_folder, dest_folder):
    Path(dest_folder).mkdir(parents=True, exist_ok=True)
    for folder in ['Train', 'Test']:
        src_path = Path(src_folder) / folder / 'Image'
        dest_path = Path(dest_folder) / folder / 'Image'
        Path(dest_path).mkdir(parents=True, exist_ok=True)
        for img_name in os.listdir(src_path):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = src_path / img_name
                image = cv2.imread(str(img_path))


                # Contrast Enhancement
                contrast_img = apply_contrast_enhancement(image)

                # Contrast Enhancement + Noise Reduction
                noise_reduced_img = apply_noise_reduction(contrast_img)

                # Contrast Enhancement + Noise Reduction + Edge Enhancement
                edge_enhanced_img = apply_edge_enhancement(noise_reduced_img)

                cv2.imwrite(str(dest_path / f'{img_name}'), edge_enhanced_img )

src_folder = 'sample_dataset_ids'
dest_folder = 'sample_dataset_ids_new'

process_images(src_folder, dest_folder)
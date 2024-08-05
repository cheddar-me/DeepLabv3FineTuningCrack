import os
import shutil
import numpy as np
from PIL import Image

# Define paths
original_dataset_path = 'sample_dataset_ids'
new_dataset_path = 'sample_dataset_passports'

# Create new directories
def create_directory_structure(base_path):
    os.makedirs(os.path.join(base_path, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'train', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'val', 'labels'), exist_ok=True)

create_directory_structure(new_dataset_path)

# Process images and labels
def process_dataset(src_path, dest_path, dataset_type):
    images_path = os.path.join(src_path, dataset_type, 'images')
    labels_path = os.path.join(src_path, dataset_type, 'labels')

    dest_images_path = os.path.join(dest_path, dataset_type, 'images')
    dest_labels_path = os.path.join(dest_path, dataset_type, 'labels')

    for filename in os.listdir(labels_path):
            # Load label image
            label_path = os.path.join(labels_path, filename)
            label_img = Image.open(label_path)
            label_array = np.array(label_img)

            # Get unique values in the label image
            unique_values = np.unique(label_array)
            print(filename)
            print(unique_values)

            if set(unique_values) == {0, 2}:
                # Rename label 2 to 1
                label_array[label_array == 2] = 255
                new_label_img = Image.fromarray(label_array.astype(np.uint8))

                # Save modified label image
                new_label_path = os.path.join(dest_labels_path, filename)
                new_label_img.save(new_label_path)

                # Copy the corresponding original image 
                image_path = os.path.join(images_path, filename.replace(".png",".jpg"))
                if os.path.exists(image_path):
                    shutil.copy(image_path, os.path.join(dest_images_path, filename.replace(".png",".jpg")))
                    print(image_path )

            elif set(unique_values) == {0, 1}:
                # Skip images and labels with only 0 and 1
                continue

# Process both train and val datasets
for dataset_type in ['train', 'val']:
    process_dataset(original_dataset_path, new_dataset_path, dataset_type)

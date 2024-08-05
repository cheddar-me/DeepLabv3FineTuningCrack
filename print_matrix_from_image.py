from PIL import Image
import numpy as np

# Load the image
#image_path = 'sample_dataset_passports/Train/Mask/006.png'
image_path = 'CrackForest/Masks/001_label.PNG'

img = Image.open(image_path)

# Convert the image to a NumPy array
img_array = np.array(img)

# Get and print the unique values in the array
unique_values = np.unique(img_array)
print(unique_values)
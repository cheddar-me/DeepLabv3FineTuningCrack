
import matplotlib.pyplot as plt
import cv2
import imutils
from imutils.perspective import four_point_transform
import numpy as np
import os

# --------- Segmentation with old fashion techniques -------------
def draw_contour_with_opencv(the_image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(the_image, cv2.COLOR_BGR2GRAY)
    # Blur the image
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    # apply adaptive thresholding and then invert the threshold map
    thresh = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    # Use morphological operations to clean the thresholded image (new)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Create a rectangular kernel (new)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)  # Close operation (new)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)  # Open operation (new)
    # find contours in the thresholded image 
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	    cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Initialize variables to keep track of the largest rectangle
    largest_rect = None
    largest_area = 0

    # Loop over the contours
    for c in cnts:
        #cv2.drawContours(the_image, [c], -1, (255, 0, 0), 1)  # Blue contour with thickness 2
        # Approximate the contour to a polygon
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
        # Get the bounding rectangle of the approximated polygon
        if len(approx) >= 4:  # Ensure the approximated contour has enough points
            x, y, w, h = cv2.boundingRect(approx)
            rect_area = w * h
        
            # Update the largest rectangle if the current one is larger
            if rect_area > largest_area:
                largest_area = rect_area
                largest_rect = (x, y, w, h)

    # Draw the largest rectangle if found
    if largest_rect is not None:
        x, y, w, h = largest_rect
        color = (0, 0, 255)  # Blue color for the rectangle

    cv2.rectangle(the_image, (x, y), (x + w, y + h), color, 3)


# Define paths
source_folder = './sample_dataset_ids/Test/Image/'
target_folder = './sample_dataset_ids/OpenCVs/'

# Process each image in the source folder
for file_name in os.listdir(source_folder):
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Read image
        img_original = cv2.imread(os.path.join(source_folder, file_name))
        
        # Apply contour drawing
        prediction = img_original.copy()
        draw_contour_with_opencv(prediction)
        
        # Save processed image
        save_path = os.path.join(target_folder, file_name)
        cv2.imwrite(save_path, prediction)
        print(f"Processed and saved: {save_path}")

print("Processing completed.")

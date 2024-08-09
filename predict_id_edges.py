import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse

# Detect if we have a GPU available
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Function to find the closest non-background pixel to a given corner
def find_closest_non_background(mask, corner):
    non_bg_points = np.argwhere(mask != 0)
    if len(non_bg_points) == 0:
        return None  # Return None if no points are found

    distances = np.sqrt((non_bg_points[:, 0] - corner[0]) ** 2 + (non_bg_points[:, 1] - corner[1]) ** 2)
    closest_point = non_bg_points[np.argmin(distances)]
    print(closest_point)
    return tuple(closest_point)

# Function to process and visualize the segmentation
def visualize_segmentation(ino):
    # Read a sample image and mask from the dataset
    img_original = cv2.imread(f'./sample_dataset_ids_new/Test/Image/{ino}.jpg')
    img = img_original.transpose(2, 0, 1).reshape(1, 3, *img_original.shape[:2])

    # Load the mask
    mask = cv2.imread(f'./sample_dataset_ids_new/Test/Mask/{ino}.png', cv2.IMREAD_GRAYSCALE)

    # Define the colors for each class (0, 1, 2)
    colors = {
        0: [0, 0, 0],       # Black for class 0 (background)
        1: [0, 255, 0],     # Green for class 1
        2: [0, 0, 255]      # Blue for class 2
    }

    # Create a color mask for the ground truth
    mask_color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i in range(3):
        mask_color[mask == i] = colors[i]

    # Get the model prediction
    with torch.no_grad():
        prediction = model(torch.from_numpy(img).float().to(device) / 255)

    # Convert the prediction to a numpy array and get the class with the highest score
    prediction_np = prediction['out'].cpu().detach().numpy()
    prediction_class = np.argmax(prediction_np[0], axis=0)

    # Create a color mask for the predicted output
    pred_color = np.zeros((*prediction_class.shape, 3), dtype=np.uint8)
    for i in range(3):
        pred_color[prediction_class == i] = colors[i]

    # Create an overlay of the original image with the prediction
    overlay = cv2.addWeighted(img_original, 1.0, pred_color, 0.5, 0)

    # Find the closest non-background points to the four corners
    height, width = prediction_class.shape
    corners = {
        'bottom_left': (height-1, 0),
        'bottom_right': (height-1, width-1),
        'top_left': (0, 0),
        'top_right': (0, width-1)
    }
    
    closest_points = {k: find_closest_non_background(prediction_class, v) for k, v in corners.items()}

    print(closest_points)
    # Draw yellow circles at the closest points on the overlay
    for point in closest_points.values():
        if point is not None:  # Ensure we found a valid point
            cv2.circle(overlay, (point[1], point[0]), radius=10, color=(255, 255, 0), thickness=2)  # RGB format for yellow

    # Plot the input image, ground truth, predicted output, and overlay
    plt.figure(figsize=(20, 8))

    plt.subplot(141)
    plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(142)
    plt.imshow(cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
    plt.title('Ground Truth')
    plt.axis('off')

    plt.subplot(143)
    plt.imshow(cv2.cvtColor(pred_color, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
    plt.title('Segmentation Output')
    plt.axis('off')

    plt.subplot(144)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
    plt.title('Overlay')
    plt.axis('off')

    plt.savefig(f'./Passport_Exp/SegmentationOutput-predict-edges-pic-{ino}.png', bbox_inches='tight')
    plt.show()

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Segmentation')
    parser.add_argument('--ino', type=str, required=True, help='Image number to process')

    args = parser.parse_args()
    ino = args.ino

    # Load the trained model 
    model = torch.load('./Passport_Exp/weights-edges-30-epochs.pt')
    # Set the model to evaluate mode
    model.eval()
    print("hola")
    # Call the visualization function
    visualize_segmentation(ino)
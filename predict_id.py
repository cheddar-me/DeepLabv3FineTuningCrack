import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Detect if we have a GPU available
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load the trained model 
model = torch.load('./Passport_Exp/weights.pt')
# Set the model to evaluate mode
model.eval()

ino = "000"
# Read  a sample image and mask from the data-set
# Resize the image
new_size = (480, 320)  # (width, height) for OpenCV
img_original = cv2.imread(f'./sample_dataset_ids/Test/Image/{ino}.jpg')
img_resized = cv2.resize(img_original, new_size)
img = img_resized.transpose(2,0,1).reshape(1,3,320,480)
mask = cv2.imread(f'./sample_dataset_ids/Test/Mask/{ino}.png', cv2.IMREAD_GRAYSCALE)
mask =  cv2.resize(mask, new_size)
#mask_np = np.zeros((mask.shape[0], mask.shape[1], 1), dtype=np.uint8)
#print(np.unique(mask_np))
#mask_np[mask_np == 1] = 200  
#mask_np[mask_np == 2] = 100 

with torch.no_grad():
    #a = model(torch.from_numpy(img).type(torch.cuda.FloatTensor)/255)
    a = model(torch.from_numpy(img).float().to("mps") / 255)

# Plot the input image, ground truth and the predicted output
plt.figure(figsize=(10,10));
plt.subplot(131);
plt.imshow(img[0,...].transpose(1,2,0));
plt.title('Image')
plt.axis('off');
plt.subplot(132);
plt.imshow(mask);
plt.title('Ground Truth')
plt.axis('off');
plt.subplot(133);
plt.imshow(a['out'].cpu().detach().numpy()[0][0]>0.2);
plt.title('Segmentation Output')
plt.axis('off');
plt.savefig(f'./Passport_Exp/SegmentationOutput-predict-pic-{ino}-epochs-50.png',bbox_inches='tight')
plt.show()
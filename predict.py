import torch
import matplotlib.pyplot as plt
import cv2

# Detect if we have a GPU available
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Load the trained model 
model = torch.load('./CFExp/weights.pt')
# Set the model to evaluate mode
model.eval()

ino = 2
# Read  a sample image and mask from the data-set
img = cv2.imread(f'./CrackForest/Images/{ino:03d}.jpg').transpose(2,0,1).reshape(1,3,320,480)
mask = cv2.imread(f'./CrackForest/Masks/{ino:03d}_label.PNG')
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
plt.savefig('./CFExp/SegmentationOutput-predict.png',bbox_inches='tight')
plt.show()
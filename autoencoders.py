import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader


# Define the transform to resize and convert the images to tensors
transform = transforms.Compose([
    transforms.Resize((640, 480)),  # Resize all images to 640x480
    transforms.ToTensor(),          # Convert image to tensor
])

# Specify your data directory
data_dir = 'cars'  # Assuming the data is organized correctly in subfolders

# Load the dataset with the transform
dataset = ImageFolder(data_dir, transform=transform)

# Create a DataLoader
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, shuffle=True)

# Try loading one batch
for img, cls in data_loader:
    print(img.shape)  # This should work now, with all images having the same size
    break



class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Modify the first convolutional layer to accept 3 channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Change from 1 to 3 channels
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output 3 channels
            nn.Sigmoid()  # Assuming output is in the range [0, 1]
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



model = Autoencoder()

model = torch.load('model.pth')

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((640, 480)),  # Resize all images to 640x480
    transforms.ToTensor(),          # Convert image to tensor
])

# Open the image
img = Image.open('image.jpg')

# Apply the transformation
img_t = transform(img)  # Now img_t is a tensor with the correct transformations

# Add a batch dimension (since the model expects a batch of images)
img_t = img_t.unsqueeze(0)  # Shape: [1, 3, 640, 480]

# Inference with no gradients
with torch.no_grad():
    recon = model(img_t)  # Forward pass through the model



#show image
recon = recon.squeeze(0)  # Shape: [3, 640, 480] removing batch size

# Convert from a PyTorch tensor to a NumPy array
recon_np = recon.permute(1, 2, 0).cpu().numpy()  # Shape: [640, 480, 3]

# Visualize the image using matplotlib
#plt.imshow(recon_np)
plt.imsave('output.png', recon_np)  # Specify the file format (e.g., png, jpg)
plt.show()
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
#from model import HazeNet, INet
from dataset import HazeDataset
from pytorch_msssim import ssim
from Statistical_Transmission.bounding_fun import bounding_function
from Gamma_Estimation.cnn_beta_estimator2 import BetaCNN
from utils import DarkChannel, AtmLight  # Import utility functions
from INet.models.dehazeformer import DehazeFormer

# Compute transmission function
def compute_transmission(hazy_img, device):
    """Compute transmission for a batch of images by processing one image at a time."""
    batch_size = hazy_img.shape[0]  # Get batch size
    transmission_list = []

    for i in range(batch_size):
        img = hazy_img[i].permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)
        img = (img * 255).astype(np.uint8)  # Convert to uint8
        _, transmission, _ = bounding_function(img, zeta=1)
        transmission_list.append(transmission)

    # Convert list to tensor and move to device
    transmission_tensor = torch.tensor(np.stack(transmission_list), dtype=torch.float32, device=device)
    return transmission_tensor.unsqueeze(1)  # Shape: (B, 1, H, W)

# Atmospheric light estimation function
def estimate_atmospheric_light(hazy_img):
    """Estimate atmospheric light for a batch."""
    hazy_np = (hazy_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    batch_A = []
    for img in hazy_np:
        dark = DarkChannel(img)
        A = AtmLight(img, dark)
        batch_A.append(A)
    A_tensor = torch.tensor(batch_A, dtype=torch.float32, device=hazy_img.device).unsqueeze(2).unsqueeze(3)
    return A_tensor  # Shape: (B, 3, 1, 1)

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('GPU: ', device)
learning_rate = 0.2
batch_size = 2
epochs = 100

# Data preparation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
dataset = HazeDataset(folder_path="../Gamma_Estimation/data/simu/", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize Haze-Net (Gamma Estimation)
haze_net = BetaCNN().to(device)
haze_net.load_state_dict(torch.load("../Gamma_Estimation/beta_cnn.pth"))
haze_net.eval()

# Initialize DehazeFormer 
i_net = DehazeFormer().to(device)
checkpoint_path = "/home/student1/Desktop/Zero_Shot/zero-shot-SID/INet/models/dehazeformer-t.pth"
checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
if "state_dict" in checkpoint:
    i_net.load_state_dict(checkpoint["state_dict"], strict=False)
else:
    i_net.load_state_dict(checkpoint, strict=False)
i_net.train()

# Define an optimizer for i_net
optimizer = torch.optim.Adam(i_net.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    epoch_loss = 0
    print('epoch: ', epoch)

    for hazy_img in dataloader:
        hazy_img = hazy_img.to(device)
        optimizer.zero_grad()  # Zero the gradients

        with torch.no_grad():
            gamma = haze_net(hazy_img)
        
        transmission = compute_transmission(hazy_img, device)
        t_power_gamma = torch.pow(transmission, gamma.view(-1, 1, 1, 1))
        A = estimate_atmospheric_light(hazy_img)
        A = A.squeeze().view(-1, 3, 1, 1)
        J_haze_free = i_net(hazy_img)  # Forward pass through i_net

        # Compute reconstructed hazy image and loss
        reconstructed_hazy = A * (1 - t_power_gamma) + t_power_gamma * J_haze_free
        loss = 1 - ssim(reconstructed_hazy, hazy_img, data_range=1.0, size_average=True)
        
        loss.backward()       # Backward pass to compute gradients
        optimizer.step()      # Update model parameters

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

# Save the trained model
model_path = "dehazeformer_trained.pth"
torch.save(i_net.state_dict(), model_path)
print(f"Model saved to {model_path}")
print("Training complete!")

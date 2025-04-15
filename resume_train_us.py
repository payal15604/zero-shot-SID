import torch
import os
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
# def compute_transmission(hazy_img):
#     """Compute transmission for a batch of images."""
#     print('Inside Compute Transmission')
#     hazy_np = (hazy_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
#     zeta = 1

#     batch_transmission = []
#     for img in hazy_np:
#         _, transmission, _ = bounding_function(img, zeta, device)
#         batch_transmission.append(transmission)

#     transmission_tensor = torch.tensor(batch_transmission, dtype=torch.float32, device=hazy_img.device)
#     return transmission_tensor.unsqueeze(1)  # Shape: (B, 1, H, W)
'''
def compute_transmission(hazy_img, device):
    """Compute transmission for a batch of images by processing one image at a time."""
    print('Inside Compute Transmission')
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
    
def compute_transmission(hazy_img, device):
    """Compute transmission for a batch of images by processing one image at a time."""
    print('Inside Compute Transmission')
    batch_size = hazy_img.shape[0]  # Get batch size
    transmission_list = []

    for i in range(batch_size):
        img = hazy_img[i].cpu().numpy().transpose(1, 2, 0) * 255  # Convert to (H, W, C) and scale
        img = np.uint8(img)
        _, transmission, _ = bounding_function(img, zeta=1)
        transmission_list.append(transmission)

    transmission_tensor = torch.tensor(np.stack(transmission_list), dtype=torch.float32, device=device)
    return transmission_tensor.unsqueeze(1)  # Shape: (B, 1, H, W)

# Atmospheric light estimation function
def estimate_atmospheric_light(hazy_img):
    """Estimate atmospheric light for a batch."""
    print('Inside Estimate Atmospheric Light')
    hazy_np = (hazy_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    batch_A = []
    for img in hazy_np:
        dark = DarkChannel(img)
        A = AtmLight(img, dark)
        batch_A.append(A)

    A_tensor = torch.tensor(batch_A, dtype=torch.float32, device=hazy_img.device).unsqueeze(2).unsqueeze(3)
    return A_tensor  # Shape: (B, 3, 1, 1)
'''

def compute_transmission(hazy_img, device):
    """Compute transmission for a batch of images."""
    #print('Inside Compute Transmission')
    batch_size = hazy_img.shape[0]
    transmission_list = []

    for i in range(batch_size):
        img = hazy_img[i].cpu().numpy().transpose(1, 2, 0) * 255  # Convert to (H, W, C) and scale
        img = np.uint8(img)
        _, transmission, _ = bounding_function(img, zeta=0.95)
        transmission_tensor = torch.tensor(transmission, dtype=torch.float32, device=device)

        # Check for NaN or Inf values
        if torch.isnan(transmission_tensor).any() or torch.isinf(transmission_tensor).any():
            print(f"NaN or Inf found in transmission for image {i}")

        transmission_list.append(transmission_tensor)

    transmission_tensor = torch.stack(transmission_list)
    return transmission_tensor.unsqueeze(1)  # Shape: (B, 1, H, W)

def estimate_atmospheric_light(hazy_img):
    """Estimate atmospheric light for a batch."""
    #print('Inside Estimate Atmospheric Light')
    hazy_np = (hazy_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    batch_A = []
    for img in hazy_np:
        dark = DarkChannel(img)
        A = AtmLight(img, dark)

        # Convert the numpy.ndarray to a PyTorch tensor
        A_tensor = torch.tensor(A, dtype=torch.float32, device=hazy_img.device)

        # Check for NaN or Inf in atmospheric light
        if torch.isnan(A_tensor).any() or torch.isinf(A_tensor).any():
            print(f"NaN or Inf found in atmospheric light for image.")
            
        batch_A.append(A_tensor)

    A_tensor = torch.stack(batch_A, dim=0).float()
    return A_tensor.unsqueeze(2).unsqueeze(3)  # Shape: (B, 3, 1, 1)

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('GPU: ', device)
lr = 1e-3
batch_size = 4
# Temporarily
epochs = 2

# Then run the script
# Data preparation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
print('transform function loaded')

dataset = HazeDataset(folder_path="../Gamma_Estimation/data/simu/", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Better to shuffle for training
print('Data Loader Loaded')

# Initialize Haze-Net (Gamma Estimation)
haze_net = BetaCNN().to(device)
haze_net.load_state_dict(torch.load("../Gamma_Estimation/beta_cnn.pth"))
haze_net.eval()
print('HazeNet loaded')

# Initialize DehazeFormer and optimizer
i_net = DehazeFormer().to(device)
optimizer = torch.optim.SGD(i_net.parameters(), lr)
start_epoch = 0

# Check for existing checkpoint to resume training
#TEMPORARY
checkpoint_path = "/home/student1/Desktop/Zero_Shot/zero-shot-SID/dehazeformer_trained_epoch_3.pth" # Path to latest checkpoint
if os.path.exists(checkpoint_path):
    print("Loading checkpoint to resume training...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    i_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}")
else:
    # Load initial pretrained weights if no checkpoint found
    initial_checkpoint_path = "/home/student1/Desktop/Zero_Shot/zero-shot-SID/INet/models/dehazeformer-t.pth"
    checkpoint = torch.load(initial_checkpoint_path, map_location=device)
    if "state_dict" in checkpoint:
        i_net.load_state_dict(checkpoint["state_dict"], strict=False)
    else:
        i_net.load_state_dict(checkpoint, strict=False)
    print("Initial model weights loaded")
  

i_net.train()
criterion = torch.nn.MSELoss()

for epoch in range(start_epoch, epochs):
    epoch_loss = 0
    print(f"Epoch: {epoch + 1}")

    for idx, hazy_img in enumerate(dataloader):
        hazy_img = hazy_img.to(device)

        with torch.no_grad():
            gamma = haze_net(hazy_img)
            transmission = compute_transmission(hazy_img, device)
            t_power_gamma = torch.pow(transmission, gamma.view(-1, 1, 1, 1))
            A = estimate_atmospheric_light(hazy_img).squeeze().view(-1, 3, 1, 1) / 255

        J_haze_free = i_net(hazy_img)
        reconstructed_hazy = A * (1 - t_power_gamma) + t_power_gamma * J_haze_free
        reconstructed_hazy = torch.clamp(reconstructed_hazy, 0, 1)

        # Calculate losses
        mse_loss = criterion(reconstructed_hazy, hazy_img)
        ssim_loss = 1 - ssim(reconstructed_hazy, hazy_img, data_range=1.0, size_average=True)
        loss = mse_loss + ssim_loss
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(i_net.parameters(), max_norm=1.0)
        optimizer.step()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # Save checkpoint every 100 epochs and at the end
    if (epoch + 1) % 100 == 0 or (epoch + 1) == epochs:
        model_path = f"dehazeformer_trained_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': i_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, model_path)
        print(f"Checkpoint saved to {model_path}")

# Final save
final_model_path = "/home/student1/Desktop/Zero_Shot/zero-shot-SID/dehazeformer_trained_epoch_3.pth"
torch.save({
    'epoch': epochs - 1,
    'model_state_dict': i_net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
}, final_model_path)
print(f"Final model saved to {final_model_path}")

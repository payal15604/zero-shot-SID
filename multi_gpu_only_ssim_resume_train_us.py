import torch
import os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from model import HazeNet, INet
from dataset import HazeDataset
from pytorch_msssim import ssim
from Statistical_Transmission.bounding_fun import bounding_function
from Gamma_Estimation.cnn_beta_estimator2 import BetaCNN
from utils import DarkChannel, AtmLight  # Import utility functions
from INet.models.dehazeformer import DehazeFormer

def compute_transmission(hazy_img, device):
    """Compute transmission for a batch of images."""
    batch_size = hazy_img.shape[0]
    transmission_list = []

    for i in range(batch_size):
        img = hazy_img[i].cpu().numpy().transpose(1, 2, 0) * 255  # Convert to (H, W, C) and scale
        img = np.uint8(img)
        _, transmission, _ = bounding_function(img, zeta=0.95)
        transmission_tensor = torch.tensor(transmission, dtype=torch.float32, device=device)

        if torch.isnan(transmission_tensor).any() or torch.isinf(transmission_tensor).any():
            print(f"NaN or Inf found in transmission for image {i}")

        transmission_list.append(transmission_tensor)

    transmission_tensor = torch.stack(transmission_list)
    return transmission_tensor.unsqueeze(1)  # Shape: (B, 1, H, W)

def estimate_atmospheric_light(hazy_img):
    """Estimate atmospheric light for a batch."""
    hazy_np = (hazy_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    batch_A = []
    for img in hazy_np:
        dark = DarkChannel(img)
        A = AtmLight(img, dark)
        A_tensor = torch.tensor(A, dtype=torch.float32, device=hazy_img.device)

        if torch.isnan(A_tensor).any() or torch.isinf(A_tensor).any():
            print(f"NaN or Inf found in atmospheric light for image.")

        batch_A.append(A_tensor)

    A_tensor = torch.stack(batch_A, dim=0).float()
    return A_tensor.unsqueeze(2).unsqueeze(3)  # Shape: (B, 3, 1, 1)

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f'Using {num_gpus} GPUs')
lr = 1e-3
batch_size = 8 * num_gpus  # Scale batch size with number of GPUs
epochs = 200

# Data preparation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
print('transform function loaded')

dataset = HazeDataset(folder_path="../Datasets/Combined_Dataset_Train/", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print('Data Loader Loaded')

# Initialize Haze-Net (Gamma Estimation) with DataParallel
haze_net = BetaCNN()
if num_gpus > 1:
    haze_net = nn.DataParallel(haze_net)
haze_net.to(device)

# Load HazeNet state_dict and adjust for DataParallel
state_dict = torch.load("../Gamma_Estimation/beta_cnn.pth", map_location=device)
if num_gpus > 1:
    new_state_dict = {f'module.{k}': v for k, v in state_dict.items()}
else:
    new_state_dict = state_dict
haze_net.load_state_dict(new_state_dict)
haze_net.eval()
print('HazeNet loaded')

# Initialize DehazeFormer with DataParallel
i_net = DehazeFormer()
if num_gpus > 1:
    i_net = nn.DataParallel(i_net)
i_net.to(device)
optimizer = torch.optim.Adam(i_net.parameters(), lr)
start_epoch = 0

# Check for existing checkpoint to resume training
checkpoint_path = "/home/student1/Desktop/Zero_Shot/zero-shot-SID/Saved_Models/combined_dataset_epoch_0.pth"
if os.path.exists(checkpoint_path):
    print("Loading checkpoint to resume training...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Adjust checkpoint state_dict for DataParallel
    state_dict = checkpoint['model_state_dict']
    if num_gpus > 1 and not any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    elif num_gpus == 1 and any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    i_net.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}")
else:
    # Load initial pretrained weights
    initial_checkpoint_path = "/home/student1/Desktop/Zero_Shot/zero-shot-SID/INet/models/dehazeformer-t.pth"
    checkpoint = torch.load(initial_checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    
    # Adjust state_dict for DataParallel
    if num_gpus > 1 and not any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    elif num_gpus == 1 and any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    i_net.load_state_dict(state_dict, strict=False)
    print("Initial model weights loaded")

i_net.train()
criterion = lambda img1, img2: 1 - ssim(img1, img2, data_range=1.0, size_average=True)

for epoch in range(start_epoch, epochs):
    epoch_loss = 0
    total_images = len(dataset)
    print(f"\nEpoch: {epoch + 1} of {epochs}")

    with tqdm(total=total_images, desc=f"Epoch {epoch+1}", unit="img") as pbar:
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

            loss = criterion(reconstructed_hazy, hazy_img)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(i_net.parameters(), max_norm=1.0)
            optimizer.step()

            processed_images = (idx + 1) * batch_size
            processed_images = min(processed_images, total_images)
            pbar.update(hazy_img.size(0))
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Processed': f'{processed_images}/{total_images}'})

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss (SSIM): {avg_loss:.4f}")

    if (epoch + 1) % 100 == 0 or (epoch + 1) == epochs:
        # Save the underlying model state_dict without DataParallel wrapper
        model_state_dict = i_net.module.state_dict() if num_gpus > 1 else i_net.state_dict()
        model_path = f"/home/student1/Desktop/Zero_Shot/zero-shot-SID/Saved_Models/combined_dataset_model_epoch_{epoch + 1}_ssim.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, model_path)
        print(f"Checkpoint saved to {model_path}")

# Final save
model_state_dict = i_net.module.state_dict() if num_gpus > 1 else i_net.state_dict()
final_model_path = "/home/student1/Desktop/Zero_Shot/zero-shot-SID/Saved_Models/combined_trained_ssim_200_1e-3.pth"
torch.save({
    'epoch': epochs - 1,
    'model_state_dict': model_state_dict,
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
}, final_model_path)
print(f"Final model saved to {final_model_path}")

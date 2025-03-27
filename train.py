import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from model import HazeNet, INet  # Assuming your models are defined in model.py
from dataset import HazeDataset  # Assuming you have a custom dataset
from pytorch_msssim import ssim  # SSIM loss from pytorch_msssim

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-4
batch_size = 16
epochs = 50
lambda_ssim = 0.85
lambda_l1 = 0.15

# Data preparation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = HazeDataset(root="data/", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
haze_net = HazeNet().to(device)
i_net = INet().to(device)

# Optimizer and loss function
optimizer = optim.Adam(list(haze_net.parameters()) + list(i_net.parameters()), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    haze_net.train()
    i_net.train()
    
    epoch_loss = 0
    for hazy_img, gt_img in dataloader:
        hazy_img, gt_img = hazy_img.to(device), gt_img.to(device)
        
        # Forward pass
        gamma = haze_net(hazy_img)  # Haze-Net estimates gamma
        pred_img = i_net(hazy_img, gamma)  # I-Net estimates haze-free image
        
        # Compute loss
        loss_ssim = 1 - ssim(pred_img, gt_img, data_range=1.0, size_average=True)
        loss_l1 = nn.L1Loss()(pred_img, gt_img)
        loss = lambda_ssim * loss_ssim + lambda_l1 * loss_l1
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

print("Training complete!")

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

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('GPU: ', device)
learning_rate = 1e-4
batch_size = 16
epochs = 50

# Data preparation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
print('transform function loaded')

dataset = HazeDataset(folder_path="../Gamma_Estimation/data/simu/", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print('Data Loader Loaded')

# Initialize Haze-Net (Gamma Estimation)
haze_net = BetaCNN().to(device)
haze_net.load_state_dict(torch.load("../Gamma_Estimation/beta_cnn.pth"))
haze_net.eval()
print('HazeNet loaded')

#Initialize DehazeFormer 
i_net = DehazeFormer().to(device)
# Load the checkpoint
checkpoint_path = "/home/student1/Desktop/Zero_Shot/zero-shot-SID/INet/models/dehazeformer-t.pth"
checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
if "state_dict" in checkpoint:
    i_net.load_state_dict(checkpoint["state_dict"], strict=False)
else:
    i_net.load_state_dict(checkpoint, strict=False)

#checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
import pickle
# Load the data.pkl file
# with open('INet/models/dehazeformer-t.pth/archive/data.pkl', 'rb') as f:
#     data = pickle.load(f)
# # Check the contents of 'data' to understand its structure
# print(data)

# If 'state_dict' is in the loaded data, you can load it into your model
# if 'state_dict' in data:
#     i_net.load_state_dict(data['state_dict'])
# else:
#     print("No state_dict found in the data.")
# Load the data.pkl file
# with open('INet/models/dehazeformer-t.pth/archive/data.pkl', 'rb') as f:
#     data = pickle.load(f)
# if 'state_dict' in data:
#     i_net.load_state_dict(data['state_dict'])
# else:
#     print("No state_dict found in the data.")



# Load the model weights
#i_net.load_state_dict(checkpoint)
i_net.train()

optimizer = torch.optim.Adam(i_net.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()  # Example loss function

# Training loop
for epoch in range(epochs):
    epoch_loss = 0

    print('epoch: ', epoch)

    for hazy_img in dataloader:
        hazy_img = hazy_img.to(device)
        print('Image of Hazy Image: ', np.shape(hazy_img))

        with torch.no_grad():
            gamma = haze_net(hazy_img)
            print('Gamma: ', gamma)
        transmission = compute_transmission(hazy_img, device)
        # t_power_gamma = torch.pow(transmission, gamma)
        t_power_gamma = torch.pow(transmission, gamma.view(-1, 1, 1, 1))
        A = estimate_atmospheric_light(hazy_img)
        print('Atmospheric Light: ', A)
        J_haze_free = i_net(hazy_img)  # Pass through INet

        # Compute reconstructed hazy image
        reconstructed_hazy = A * (1 - t_power_gamma) + t_power_gamma * J_haze_free

        # Compute loss
        loss = 1 - ssim(reconstructed_hazy, hazy_img, data_range=1.0, size_average=True)
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

print("Training complete!")

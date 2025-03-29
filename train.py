import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from model import HazeNet, INet
from dataset import HazeDataset
from pytorch_msssim import ssim
from Statistical_Transmission.bounding_fun import bounding_function
from Gamma_Estimation.cnn_beta_estimator2 import BetaCNN
from utils import DarkChannel, AtmLight  # Import utility functions

# Compute transmission function
def compute_transmission(hazy_img):
    """Compute transmission for a batch of images."""
    hazy_np = (hazy_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    zeta = 1

    batch_transmission = []
    for img in hazy_np:
        _, transmission, _ = bounding_function(img, zeta)
        batch_transmission.append(transmission)

    transmission_tensor = torch.tensor(batch_transmission, dtype=torch.float32, device=hazy_img.device)
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
learning_rate = 1e-4
batch_size = 16
epochs = 50

# Data preparation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

dataset = HazeDataset(root="data/", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize Haze-Net (Gamma Estimation)
haze_net = BetaCNN().to(device)
haze_net.load_state_dict(torch.load("Gamma_Estimation/beta_cnn.pth"))
haze_net.eval()

# Training loop
for epoch in range(epochs):
    epoch_loss = 0

    for hazy_img in dataloader:
        hazy_img = hazy_img.to(device)

        with torch.no_grad():
            gamma = haze_net(hazy_img)

        transmission = compute_transmission(hazy_img)
        t_power_gamma = torch.pow(transmission, gamma)
        A = estimate_atmospheric_light(hazy_img)

        J_haze_free = INet()(hazy_img)  # Pass through INet

        # Compute reconstructed hazy image
        reconstructed_hazy = A * (1 - t_power_gamma) + t_power_gamma * J_haze_free

        # Compute loss
        loss = 1 - ssim(reconstructed_hazy, hazy_img, data_range=1.0, size_average=True)
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

print("Training complete!")

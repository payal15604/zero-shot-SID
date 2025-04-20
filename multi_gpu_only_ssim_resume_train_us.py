import torch
import os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

from dataset import HazeDataset
from pytorch_msssim import ssim
from Statistical_Transmission.bounding_fun import bounding_function
from Gamma_Estimation.cnn_beta_estimator2 import BetaCNN
from utils import DarkChannel, AtmLight
from INet.models.dehazeformer import DehazeFormer

# ---------- Utility Functions ----------

def compute_transmission(hazy_img, device):
    transmission_list = []
    for i in range(hazy_img.size(0)):
        img = hazy_img[i].cpu().numpy().transpose(1, 2, 0) * 255
        img = np.uint8(img)
        _, transmission, _ = bounding_function(img, zeta=0.95)
        transmission_tensor = torch.tensor(transmission, dtype=torch.float32, device=device)
        transmission_list.append(transmission_tensor)
    transmission_tensor = torch.stack(transmission_list)
    return transmission_tensor.unsqueeze(1)

def estimate_atmospheric_light(hazy_img):
    hazy_np = (hazy_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    batch_A = []
    for img in hazy_np:
        dark = DarkChannel(img)
        A = AtmLight(img, dark)
        A_tensor = torch.tensor(A, dtype=torch.float32, device=hazy_img.device)
        batch_A.append(A_tensor)
    A_tensor = torch.stack(batch_A, dim=0).float()
    return A_tensor.unsqueeze(2).unsqueeze(3)

# ---------- Setup ----------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('GPU:', device)

lr = 1e-3
batch_size = 8
epochs = 200

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
print("Transform loaded")

dataset = HazeDataset(folder_path="../Datasets/Combined_Dataset_Train/", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("DataLoader ready")

# ---------- Load BetaCNN (Gamma Estimator) ----------

haze_net = BetaCNN().to(device)
beta_cnn_ckpt = torch.load("../Gamma_Estimation/beta_cnn.pth", map_location=device)

# Handle DataParallel mismatch
new_state_dict = {}
for k, v in beta_cnn_ckpt.items():
    new_k = f"module.{k}" if not k.startswith("module.") else k
    new_state_dict[new_k] = v

# Wrap with DataParallel
haze_net = torch.nn.DataParallel(haze_net)
haze_net.load_state_dict(new_state_dict)
haze_net.eval()
print("BetaCNN (Gamma Estimator) loaded")

# ---------- Load DehazeFormer ----------

i_net = DehazeFormer().to(device)
i_net = torch.nn.DataParallel(i_net)
optimizer = optim.Adam(i_net.parameters(), lr)
start_epoch = 0

checkpoint_path = "/home/student1/Desktop/Zero_Shot/zero-shot-SID/Saved_Models/combined_dataset_epoch_0.pth"
if os.path.exists(checkpoint_path):
    print("Loading checkpoint to resume training...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    i_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming from epoch {start_epoch}")
else:
    print("Loading initial pretrained weights...")
    initial_weights = torch.load("/home/student1/Desktop/Zero_Shot/zero-shot-SID/INet/models/dehazeformer-t.pth", map_location=device)
    if "state_dict" in initial_weights:
        i_net.load_state_dict(initial_weights["state_dict"], strict=False)
    else:
        i_net.load_state_dict(initial_weights, strict=False)
    print("Initial model weights loaded")

# ---------- Training ----------

criterion = lambda x, y: 1 - ssim(x, y, data_range=1.0, size_average=True)

i_net.train()
for epoch in range(start_epoch, epochs):
    epoch_loss = 0
    total_images = len(dataset)
    print(f"\nEpoch: {epoch + 1}/{epochs}")

    with tqdm(total=total_images, desc=f"Epoch {epoch+1}", unit="img") as pbar:
        for idx, hazy_img in enumerate(dataloader):
            hazy_img = hazy_img.to(device)

            with torch.no_grad():
                gamma = haze_net(hazy_img)
                transmission = compute_transmission(hazy_img, device)
                t_power_gamma = torch.pow(transmission, gamma.view(-1, 1, 1, 1))
                A = estimate_atmospheric_light(hazy_img).squeeze().view(-1, 3, 1, 1) / 255

            J_haze_free = i_net(hazy_img)
            reconstructed = A * (1 - t_power_gamma) + t_power_gamma * J_haze_free
            reconstructed = torch.clamp(reconstructed, 0, 1)

            loss = criterion(reconstructed, hazy_img)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(i_net.parameters(), 1.0)
            optimizer.step()

            processed = min((idx + 1) * batch_size, total_images)
            pbar.update(hazy_img.size(0))
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Processed': f'{processed}/{total_images}'})

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}], Avg SSIM Loss: {avg_loss:.4f}")

    if (epoch + 1) % 100 == 0 or (epoch + 1) == epochs:
        save_path = f"/home/student1/Desktop/Zero_Shot/zero-shot-SID/Saved_Models/combined_dataset_model_epoch_{epoch+1}_ssim.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': i_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss
        }, save_path)
        print(f"Checkpoint saved at {save_path}")

# Final Save
final_path = "/home/student1/Desktop/Zero_Shot/zero-shot-SID/Saved_Models/combined_trained_ssim_200_1e-3.pth"
torch.save({
    'epoch': epochs - 1,
    'model_state_dict': i_net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': avg_loss,
}, final_path)
print(f"Final model saved at {final_path}")

import torch
import os
from tqdm import tqdm
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

def compute_transmission(hazy_img, device):
    """Compute transmission for a batch of images."""
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
    
# Function to reduce LR manually
def adjust_learning_rate(optimizer, epoch, lr_schedule_epoch=20, lr_decay=0.1):
    """Reduce learning rate by a factor every `lr_schedule_epoch` epochs."""
    if (epoch + 1) % lr_schedule_epoch == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        print(f"Learning rate reduced to {optimizer.param_groups[0]['lr']:.1e}")


# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('GPU: ', device)
lr = 1e-3
batch_size = 1
epochs = 200

# Data preparation
transform = transforms.Compose([
    transforms.Resize((256, 256)), #512 * 512
    transforms.ToTensor()
])
print('transform function loaded')

dataset = HazeDataset(folder_path="../Datasets/OneImage/", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # Better to shuffle for training
print('Data Loader Loaded')

# Initialize Haze-Net (Gamma Estimation)
haze_net = BetaCNN().to(device)
haze_net.load_state_dict(torch.load("../Gamma_Estimation/beta_cnn.pth"))
haze_net.train()
print('HazeNet loaded')


# Initialize DehazeFormer and optimizer
i_net = DehazeFormer().to(device)
optimizer_i_net = torch.optim.Adam(i_net.parameters(), lr)
optimizer_haze_net = torch.optim.Adam(haze_net.parameters(), 0.1)
start_epoch = 0

# Check for existing checkpoint to resume training
checkpoint_path = "/home/student1/Desktop/Zero_Shot/zero-shot-SID/Saved_Models/combined_dataset_model21Aprilmorning_gamma.pth" # Path to latest checkpoint
if os.path.exists(checkpoint_path):
    print("Loading checkpoint to resume training...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    i_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer_i_net.load_state_dict(checkpoint['optimizer_state_dict'])
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

import cv2

import cv2
import numpy as np

def display_image_opencv(image_tensor, title="Image", pause_time=5000, target_size=(800, 800)):
    """Helper function to display an image tensor using OpenCV, resize it, and pause for a given time."""
    # Check the shape of the image tensor
    print(f"Image tensor shape: {image_tensor.shape}")

    # If the tensor is a batch (e.g., B, C, H, W), select the first image in the batch
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]  # Select the first image from the batch (B, C, H, W) -> (C, H, W)
    
    # Convert the tensor to numpy array and move it to CPU if it's on GPU
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)  # Convert (C, H, W) to (H, W, C)

    # Convert the image to BGR for OpenCV (OpenCV uses BGR format by default)
    # If the image is in RGB, convert it to BGR
    image_bgr = image[..., ::-1]  # Reverse the color channels (RGB -> BGR)

    # Resize the image to the target size (default is 800x800)
    image_resized = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_LINEAR)

    # Display the resized image using OpenCV
    cv2.imshow(title, image_resized)

    # Wait for a key press or pause for a given time (in milliseconds)
    cv2.waitKey(pause_time)  # pause_time is in milliseconds (e.g., 1000ms = 1 second)
    cv2.destroyAllWindows()


i_net.train()
# Criterion is now just the SSIM loss (1 - SSIM)
# Corrected version of the criterion with MSELoss properly instantiated
criterion_mse = lambda img1, img2:nn.MSELoss()(img1, img2)
criterion_ssim = lambda img1, img2: 1 - ssim(img1, img2, data_range=1.0, size_average=True)
max_gamma = 0
for epoch in range(start_epoch, epochs):
    epoch_loss = 0
    total_images = len(dataset)
    print(f"\nEpoch: {epoch + 1} of {epochs}")

    with tqdm(total=total_images, desc=f"Epoch {epoch+1}", unit="img") as pbar:
        for idx, hazy_img in enumerate(dataloader):
            hazy_img = hazy_img.to(device)

            with torch.no_grad():
                if epoch == 0:
                    # Initial gamma value at epoch 0 (0.1)
                    gamma = torch.tensor([1.0], dtype=torch.float32, requires_grad=True, device=device)  # (B,)
                else:
                    # For subsequent epochs, compute gamma and clamp it between 0 and 2
                    #gamma = haze_net(reconstructed_hazy)  # Assume output of haze_net is in a reasonable range
                    gamma = torch.tensor([1.0], dtype=torch.float32, requires_grad=True, device=device)  # (B,)
                    #gamma = torch.clamp(gamma, 0, 2)  # Clamping gamma to the range [0, 2]
                # max_gamma = torch.max(max_gamma, gamma.max())  # Keep it as a tensor
  # Get the scalar value of the max
                #max_gamma = max(gamma.max().item(), max_gamma)
                #print(f"Max Gamma at epoch {epoch + 1}: {gamma.max()}")
                #gamma = gamma / max_gamma
                
                print(f"Gamma at epoch {epoch + 1}: {gamma.item():.6f}")
                transmission = compute_transmission(hazy_img, device)
                
                print(f"Before gamma application: {transmission.shape}")
                t_power_gamma = torch.pow(transmission, gamma.view(1, 1, 1, 1))
                print(f"After gamma application: {t_power_gamma.shape}")

                A = estimate_atmospheric_light(hazy_img).squeeze().view(-1, 3, 1, 1) / 255
                print(f"gamma shape: {gamma.shape}")
                print(f"transmission shape: {transmission.shape}")


            J_haze_free = i_net(hazy_img)
            J_haze_free = torch.clamp(J_haze_free, 0, 1)
            #print(J_haze_free)
                  
            display_image_opencv(J_haze_free, title=f"Dehazed Image Epoch {epoch+1}, Batch {idx+1}", target_size=(256, 256))

     
            
            reconstructed_hazy = A * (1 - t_power_gamma) + t_power_gamma * J_haze_free
            reconstructed_hazy = torch.clamp(reconstructed_hazy, 0, 1)

            loss_mse = criterion_mse(reconstructed_hazy, hazy_img)
            
            # SSIM loss for BetaCNN (update only BetaCNN)
            loss_ssim = criterion_ssim(reconstructed_hazy, hazy_img)
            combine_loss=(loss_mse+loss_ssim)
         
            epoch_loss += combine_loss.item()

            optimizer_i_net.zero_grad()
            optimizer_haze_net.zero_grad()
            combine_loss.backward()
            torch.nn.utils.clip_grad_norm_(i_net.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(haze_net.parameters(), max_norm=1.0)
            optimizer_i_net.step()
            optimizer_haze_net.step()

            # Update progress bar and print count
            processed_images = (idx + 1) * batch_size
            processed_images = min(processed_images, total_images)
            pbar.update(hazy_img.size(0))
            pbar.set_postfix({'Loss': f'{combine_loss.item():.4f}', 'Processed': f'{processed_images}/{total_images}'})

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss (SSIM): {avg_loss:.4f}")
    
    adjust_learning_rate(optimizer_i_net, epoch)
    #adjust_learning_rate(optimizer_haze_net, epoch, 20, 0.1)

    # Save checkpoint every 100 epochs and at the end
    if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
        model_path = f"/home/student1/Desktop/Zero_Shot/zero-shot-SID/Saved_Models/combined_dataset_model21Aprilmorning_epoch_{epoch + 1}_ssim.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': i_net.state_dict(),
            'optimizer_state_dict': optimizer_i_net.state_dict(),
            'loss': avg_loss,
        }, model_path)
        print(f"Checkpoint saved to {model_path}")

# Final save
final_model_path = "/home/student1/Desktop/Zero_Shot/zero-shot-SID/Saved_Models/combined_dataset_model21Aprilmorning_gamma.pth"
torch.save({
    'epoch': epochs - 1,
    'model_state_dict': i_net.state_dict(),
    'optimizer_state_dict': optimizer_i_net.state_dict(),
    'loss': avg_loss,
}, final_model_path)
print(f"Final model saved to {final_model_path}")

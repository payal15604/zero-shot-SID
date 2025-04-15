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
epochs = 3000


# Data preparation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
print('transform function loaded')

dataset = HazeDataset(folder_path="../Gamma_Estimation/data/simu/", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
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
   print("STATE-DICT NOT FOUND---------------------------------------------------------------------------------------------")
   i_net.load_state_dict(checkpoint, strict=False)


i_net.train()

optimizer = torch.optim.SGD(i_net.parameters(), lr)
criterion = torch.nn.MSELoss()  # Example loss function

# Training loop
'''
for epoch in range(epochs):
    epoch_loss = 0

    print('epoch: ', epoch)

    for hazy_img in dataloader:
        print('epoch: ', epoch)
        hazy_img = hazy_img.to(device)
        print('Image of Hazy Image: ', np.shape(hazy_img))

        with torch.no_grad():
            gamma = haze_net(hazy_img)
            print('Gamma: ', gamma)
        transmission = compute_transmission(hazy_img, device)
        # t_power_gamma = torch.pow(transmission, gamma)
        t_power_gamma = torch.pow(transmission, gamma.view(-1, 1, 1, 1))
        A = estimate_atmospheric_light(hazy_img)
        print('Atmospheric Light Old: ', A)
        A = A.squeeze()  # Remove extra dimensions
        A = A.view(-1, 3, 1, 1)  # Ensure correct shape (batch_size, 3, 1, 1)
        print('Atmospheric Light New: ', A)
        print(np.shape(A))
	
        J_haze_free = i_net(hazy_img)  # Pass through INet

        # Compute reconstructed hazy image
        reconstructed_hazy = A * (1 - t_power_gamma) + t_power_gamma * J_haze_free
	
        mse_loss = criterion(reconstructed_hazy, hazy_img)
        ssim_loss = 1 - ssim(reconstructed_hazy, hazy_img, data_range=1.0, size_average=True)
        print(f"SSIM value: {ssim_loss.item()}")
        loss = mse_loss + ssim_loss  # Combine both losses

        # Compute loss
        #loss = 1 - ssim(reconstructed_hazy, hazy_img, data_range=1.0, size_average=True)
        epoch_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

for epoch in range(epochs):
    epoch_loss = 0
    print(f"Epoch: {epoch + 1}")

    for idx, hazy_img in enumerate(dataloader):
        hazy_img = hazy_img.to(device)
        
        # Forward pass
        with torch.no_grad():
            gamma = haze_net(hazy_img)
            #print(f"Gamma: {gamma}")
            if torch.isnan(gamma).any() or torch.isinf(gamma).any():
                print(f"NaN or Inf found in gamma!")
        
        transmission = compute_transmission(hazy_img, device)
        t_power_gamma = torch.pow(transmission, gamma.view(-1, 1, 1, 1))
        A = estimate_atmospheric_light(hazy_img)
        A = A.squeeze().view(-1, 3, 1, 1)

        # Dehaze network output
        J_haze_free = i_net(hazy_img)  
        
        # Ensure all images are within [0, 1] range
        hazy_img = torch.clamp(hazy_img, 0, 1)
        reconstructed_hazy = A * (1 - t_power_gamma) + t_power_gamma * J_haze_free

        # Ensure clamped images for SSIM calculation
        reconstructed_hazy = torch.clamp(reconstructed_hazy, 0, 1)

        # Compute SSIM for each image in the batch
        for i in range(hazy_img.size(0)):  # Iterate through batch size
            ssim_value = 1 - ssim(reconstructed_hazy[i:i+1], hazy_img[i:i+1], data_range=1.0, size_average=True)
            #print(f"SSIM for image {i + 1} in batch {idx + 1}: {ssim_value.item()}")

        # Compute the total loss (MSE + SSIM)
        mse_loss = criterion(reconstructed_hazy, hazy_img)
        ssim_loss = 1 - ssim(reconstructed_hazy, hazy_img, data_range=1.0, size_average=True)
        loss = mse_loss + ssim_loss
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(i_net.parameters(), max_norm=1.0)
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

print("Training complete!")
'''
for epoch in range(epochs):
    epoch_loss = 0
    print(f"Epoch: {epoch + 1}")

    # Loop through the dataset
    for idx, hazy_img in enumerate(dataloader):
    	
        hazy_img = hazy_img.to(device)
        #print(np.mean(hazy_img.cpu().numpy()))
        
        # Forward pass
        with torch.no_grad():
            gamma = haze_net(hazy_img)
            #print(f"Gamma: {gamma}")
            #if torch.isnan(gamma).any() or torch.isinf(gamma).any():
                #print(f"NaN or Inf found in gamma!")
        
        # Compute transmission
        transmission = compute_transmission(hazy_img, device)
        t_power_gamma = torch.pow(transmission, gamma.view(-1, 1, 1, 1))
        
        # Estimate atmospheric light
        A = estimate_atmospheric_light(hazy_img)
        A = A.squeeze().view(-1, 3, 1, 1) / 255
        #print('Atmnospheric Light: ', A)
        # Dehaze network output
        J_haze_free = i_net(hazy_img)  
        
        # Ensure all images are within [0, 1] range
        hazy_img = torch.clamp(hazy_img, 0, 1)
        reconstructed_hazy = A * (1 - t_power_gamma) + t_power_gamma * J_haze_free

        # Ensure clamped images for SSIM calculation
        reconstructed_hazy = torch.clamp(reconstructed_hazy, 0, 1)

        # Compute SSIM for each image
        for i in range(hazy_img.size(0)):  # Here it's always 1 image since batch_size = 1
            ssim_value = 1 - ssim(reconstructed_hazy[i:i+1], hazy_img[i:i+1], data_range=1.0, size_average=True)
            #print(f"SSIM for image {i + 1}: {ssim_value.item()}")

        # Compute the total loss (MSE + SSIM)
        mse_loss = criterion(reconstructed_hazy, hazy_img)
        #print(mse_loss)
        ssim_loss = 1 - ssim(reconstructed_hazy, hazy_img, data_range=1.0, size_average=True)
        loss = mse_loss + ssim_loss
        epoch_loss += loss.item()

        # Zero gradients, backpropagation, and optimizer step
        optimizer.zero_grad()
        loss.backward()

	# Print the gradients for each parameter
        #for name, param in i_net.named_parameters():
            #if param.grad is not None:  # Check if the parameter has a gradient
                      #print(f'Gradient for {name}: {param.grad}')

	# Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(i_net.parameters(), max_norm=1.0)
        optimizer.step()

        # Print loss for every image in the batch (which is just 1 image here)
        #print(f"Loss for image {idx + 1}: {loss.item()}")
        

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
    if (epoch % 100 == 0):
        model_path = f"dehazeformer_trained_epoch_{epoch + 1}.pth"
        torch.save(i_net.state_dict(), model_path)
        print(f"Model saved to {model_path}")

# Save the trained model
model_path = "dehazeformer_trained_1000_1e-3.pth"
torch.save(i_net.state_dict(), model_path)
print(f"Model saved to {model_path}")


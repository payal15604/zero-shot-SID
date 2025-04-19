import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from dataset import HazeDataset
from INet.models.dehazeformer import DehazeFormer

# Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 1e-6
batch_size = 8

# Load the dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load your training dataset (use the correct path)
dataset = HazeDataset(folder_path="../Gamma_Estimation/data/simu/", transform=transform)
train_dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # No shuffle for testing

# Load the saved model
checkpoint_path = "//home/student1/Desktop/Zero_Shot/zero-shot-SID/dehazeformer_trained_epoch_2100.pth"  # Path to your final model checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
model = DehazeFormer().to(device)  # Initialize the model

# Load the model state dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set the model to evaluation mode

# Define loss and accuracy calculation
psnr_sum = 0
ssim_sum = 0
n = len(train_dl)  # Total number of batches

# Compute loss and accuracy for training data
with torch.no_grad():
    for hazy, clean in train_dl:
        hazy = hazy.to(device)
        clean = clean.to(device)

        # Forward pass
        output = model(hazy)

        # Convert output and clean to numpy arrays for evaluation
        output_np = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
        clean_np = clean.squeeze(0).cpu().permute(1, 2, 0).numpy()

        # PSNR
        psnr = compare_psnr(clean_np, output_np, data_range=1.0)
        # SSIM (multichannel=True for color)
        ssim = compare_ssim(clean_np, output_np, data_range=1.0, multichannel=True)

        mse_sum += mse
        psnr_sum += psnr
        ssim_sum += ssim

# Calculate average MSE, PSNR, and SSIM
avg_psnr = psnr_sum / n
avg_ssim = ssim_sum / n

print(f"Avg PSNR: {avg_psnr:.2f} dB")
print(f"Avg SSIM: {avg_ssim:.4f}")

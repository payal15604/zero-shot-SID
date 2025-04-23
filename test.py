import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pytorch_msssim import ssim
from PIL import Image
#from model import HazeNet, INet
from dataset import HazeDataset
from pytorch_msssim import ssim
from Statistical_Transmission.bounding_fun import bounding_function
from Gamma_Estimation.cnn_beta_estimator2 import BetaCNN
from utils import DarkChannel, AtmLight  # Import utility functions
from INet.models.dehazeformer import DehazeFormer
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Custom dataset class for loading paired hazy and clear images
class HazePairDataset(Dataset):
    def __init__(self, hazy_folder, clear_folder, transform=None):
        self.hazy_images = sorted(os.listdir(hazy_folder))  # Assuming sorted order of images
        self.clear_images = sorted(os.listdir(clear_folder))  # Assuming sorted order of images
        self.hazy_folder = hazy_folder
        self.clear_folder = clear_folder
        self.transform = transform

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_img_path = os.path.join(self.hazy_folder, self.hazy_images[idx])
        clear_img_path = os.path.join(self.clear_folder, self.clear_images[idx])

        hazy_img = Image.open(hazy_img_path).convert('RGB')
        clear_img = Image.open(clear_img_path).convert('RGB')

        if self.transform:
            hazy_img = self.transform(hazy_img)
            clear_img = self.transform(clear_img)

        return hazy_img, clear_img

def display_image_opencv(image_tensor, title="Image", pause_time=10000):
    """Helper function to display an image tensor using OpenCV and pause for a given time."""
    # Convert the tensor to numpy array and move it to CPU if it's on GPU
    image = image_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
    image = np.clip(image, 0, 1)  # Ensure the values are in the [0, 1] range

    # Convert the image to BGR for OpenCV (OpenCV uses BGR format by default)
    image_bgr = (image * 255).astype(np.uint8)  # Convert the range [0, 1] to [0, 255]

    # Display the image using OpenCV
    cv2.imshow(title, image_bgr)

    # Wait for a key press or pause for a given time (in milliseconds)
    cv2.waitKey(pause_time)  # pause_time is in milliseconds (e.g., 1000ms = 1 second)
    cv2.destroyAllWindows() 
    
# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    total_ssim = 0.0
    total_psnr = 0.0
    total_images = 0

    with torch.no_grad():
        for hazy_img, clear_img in tqdm(dataloader, desc="Evaluating"):
            hazy_img = hazy_img.to(device)
            clear_img = clear_img.to(device)

            # Get the dehazed output from the model
            dehazed_img = model(hazy_img)
            dehazed_img = torch.clamp(dehazed_img, 0, 1)  # Ensure the output is in the [0, 1] range

            # Calculate SSIM (Structural Similarity Index)
            ssim_value = ssim(dehazed_img, clear_img, data_range=1.0, size_average=True)
            total_ssim += ssim_value.item()

            # Calculate PSNR (Peak Signal-to-Noise Ratio)
            mse = nn.functional.mse_loss(dehazed_img, clear_img)
            psnr_value = 10 * torch.log10(1 / mse)  # PSNR formula
            total_psnr += psnr_value.item()

            total_images += hazy_img.size(0)
            display_image_opencv(dehazed_img[0], title=f"Dehazed Image", pause_time=20000)

    # Calculate average SSIM and PSNR
    avg_ssim = total_ssim / total_images
    avg_psnr = total_psnr / total_images

    print(f"Evaluation Results - SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}")

# Main function
def main():
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('GPU: ', device)

    # Load test dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Same as training
        transforms.ToTensor()
    ])
    print('Transform function loaded')

    hazy_folder = "../Datasets/Test/hazy"  # Replace with your hazy folder path
    clear_folder = "../Datasets/Test/clear"  # Replace with your clear folder path
    test_dataset = HazePairDataset(hazy_folder=hazy_folder, clear_folder=clear_folder, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    print('Data Loader Loaded')

    # Load pre-trained model
    model_path = "/home/student1/Desktop/Zero_Shot/zero-shot-SID/Saved_Models/combined_dataset_model21Aprilmorning_epoch_100_ssim.pth"  # Replace with the path to your saved model
    print(f"Loading model from {model_path}")
    
    i_net = DehazeFormer().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    i_net.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {model_path}")

    # Evaluate the model
    evaluate(i_net, test_dataloader, device)

if __name__ == "__main__":
    main()


import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np

from pytorch_msssim import ssim
from dataset import HazeDataset
from Statistical_Transmission.bounding_fun import bounding_function
from utils import DarkChannel, AtmLight
from INet.models.dehazeformer import DehazeFormer
from Gamma_Estimation.cnn_beta_estimator2 import BetaCNN

def compute_transmission(hazy_img, device, zeta=0.95):
    batch_trans = []
    for i in range(hazy_img.size(0)):
        img = (hazy_img[i].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
        _, t, _ = bounding_function(img, zeta=zeta)
        batch_trans.append(torch.tensor(t, dtype=torch.float32, device=device))
    return torch.stack(batch_trans).unsqueeze(1)

def estimate_atmospheric_light(hazy_img, device):
    batch_A = []
    np_imgs = (hazy_img.permute(0,2,3,1).cpu().numpy()*255).astype(np.uint8)
    for img in np_imgs:
        d = DarkChannel(img)
        A = AtmLight(img, d)
        batch_A.append(torch.tensor(A, dtype=torch.float32, device=device))
    return torch.stack(batch_A).unsqueeze(2).unsqueeze(3) / 255.0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Dataset & Dataloader ---
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])
    dataset = HazeDataset(folder_path="../Gamma_Estimation/data/simu/", transform=transform)
    loader  = DataLoader(dataset, batch_size=8, shuffle=False)

    # --- Models ---
    haze_net = BetaCNN().to(device)
    haze_net.load_state_dict(torch.load("../Gamma_Estimation/beta_cnn.pth", map_location=device))
    haze_net.eval()

    dehaze = DehazeFormer().to(device)
    ckpt = torch.load("dehazeformer_trained_epoch_2100.pth", map_location=device)
    dehaze.load_state_dict(ckpt['model_state_dict'])
    dehaze.eval()

    ssim_losses = []

    with torch.no_grad():
        for hazy in loader:
            hazy = hazy.to(device)

            # estimate gamma, transmission, A
            gamma     = haze_net(hazy)
            trans     = compute_transmission(hazy, device)
            t_gamma   = trans ** gamma.view(-1,1,1,1)
            A         = estimate_atmospheric_light(hazy, device)

            # predicted clean image
            J = dehaze(hazy)

            # reconstruct hazy image
            rec = A * (1 - t_gamma) + t_gamma * J
            rec = rec.clamp(0,1)

            # SSIM loss
            loss_ssim = 1 - ssim(rec, hazy, data_range=1.0, size_average=True)
            ssim_losses.append(loss_ssim.item())

    avg_ssim = sum(ssim_losses) / len(ssim_losses)
    print(f"Average SSIM loss over dataset: {avg_ssim:.4f}")

if __name__ == "__main__":
    main()

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from dataset import HazeDataset         # must return (hazy, clean)
from INet.models.dehazeformer import DehazeFormer

# 1. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Data transforms & loader
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Make sure your dataset __getitem__ returns (hazy_tensor, clean_tensor)
test_ds = HazeDataset(folder_path="../Gamma_Estimation/data/simu/", transform=transform)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False)

# 3. Load your trained model
checkpoint_path = "/home/student1/Desktop/Zero_Shot/zero-shot-SID/dehazeformer_trained_epoch_2100.pth"
ckpt = torch.load(checkpoint_path, map_location=device)

model = DehazeFormer().to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# 4. Accumulators
psnr_total = 0.0
ssim_total = 0.0
count = 0

# 5. Loop & compute
with torch.no_grad():
    for hazy, clean in test_dl:
        hazy = hazy.to(device)
        clean = clean.to(device)

        # forward
        output = model(hazy).clamp(0, 1)

        # to numpy HWC
        out_np   = output[0].cpu().permute(1, 2, 0).numpy()
        clean_np = clean[0].cpu().permute(1, 2, 0).numpy()

        psnr = compare_psnr(clean_np, out_np, data_range=1.0)
        ssim = compare_ssim(clean_np, out_np, data_range=1.0, multichannel=True)

        psnr_total += psnr
        ssim_total += ssim
        count += 1

# 6. Report
avg_psnr = psnr_total / count
avg_ssim = ssim_total / count

print(f"Average PSNR: {avg_psnr:.2f} dB")
print(f"Average SSIM: {avg_ssim:.4f}")

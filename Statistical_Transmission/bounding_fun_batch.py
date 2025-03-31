import torch
import torch.nn.functional as F
import numpy as np
from skimage.morphology import closing, footprint_rectangle
from .airlight import airlight
from .cal_transmission import cal_trans
from .defog import defog

def bounding_function(I_batch, zeta, device):
    """
    Processes a batch of images in parallel on GPU.
    Args:
        I_batch (torch.Tensor): Batch of images (B, C, H, W) on GPU
        zeta (float): Processing parameter
        device (str): 'cuda:0' or 'cuda:1' to distribute processing
    Returns:
        r_batch (torch.Tensor): Dehazed images (B, C, H, W)
        trans_batch (torch.Tensor): Transmission maps (B, 1, H, W)
        A_batch (torch.Tensor): Atmospheric light per image (B, 1, 1, 1)
    """
    print("Initial I_batch shape:", I_batch.shape)  

    # Convert NumPy array to PyTorch tensor if needed
    if isinstance(I_batch, np.ndarray):
        I_batch = torch.tensor(I_batch, dtype=torch.float32, device=device) 

    # Ensure correct shape (B, C, H, W)
    if len(I_batch.shape) == 3:  
        I_batch = I_batch.unsqueeze(0)  # Add batch dimension (1, H, W, C)
    
    if I_batch.shape[-1] == 3:  
        I_batch = I_batch.permute(0, 3, 1, 2)  # Change (B, H, W, C) → (B, C, H, W)
    
    print("Fixed I_batch shape:", I_batch.shape)  
    B, C, H, W = I_batch.shape  

    min_I = torch.min(I_batch, dim=1, keepdim=True)[0]  
    MAX = torch.max(min_I)

    # Compute airlight per image (Loop to handle batch processing)
    A_list = []
    for i in range(B):
        img_np = I_batch[i].permute(1, 2, 0).cpu().numpy()  # Convert (C, H, W) → (H, W, C)
        A_list.append(airlight(img_np, 3))

    A1 = torch.tensor(A_list, device=device, dtype=torch.float32)  # Convert back to tensor
    print("A1 shape:", A1.shape)  

    # Fix shape for broadcasting
    A = A1.view(B, 1, 1, 1)  

    delta = zeta / (min_I.sqrt() + 1e-6)  # Prevent division by zero
    est_tr_proposed = 1 / (1 + (MAX * 10 ** (-0.05 * delta)) / (A - min_I + 1e-6))

    tr1 = (min_I >= A).float()
    tr2 = (min_I < A).float() * est_tr_proposed
    tr4 = tr1 * est_tr_proposed

    # Ensure tr3_max is never zero
    tr3_max = torch.max(tr4, dim=[1, 2, 3], keepdim=True)[0]
    tr3_max = torch.clamp(tr3_max, min=1e-6)  # Prevent division by zero
    tr3 = tr4 / tr3_max
    est_tr_proposed = tr2 + tr3

    # Apply morphological closing (Ensure proper dtype)
    est_tr_np = est_tr_proposed.cpu().numpy()
    est_tr_np = (est_tr_np * 255).astype(np.uint8)  # Convert to uint8 for morphology
    est_tr_proposed = closing(est_tr_np, footprint_rectangle(3, 3))  
    est_tr_proposed = torch.tensor(est_tr_proposed, device=device, dtype=torch.float32) / 255.0  # Normalize back

    est_tr_proposed = cal_trans(I_batch, est_tr_proposed, 1, 0.5)  
    r = defog(I_batch, est_tr_proposed, A, 0.9)  

    return r, est_tr_proposed.unsqueeze(1), A

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
    print("I_batch shape:", I_batch.shape)
    B, C, H, W = I_batch.shape
    
    min_I = torch.min(I_batch, dim=1, keepdim=True)[0]  # Get min across channels
    MAX = torch.max(min_I)

    A1 = airlight(I_batch, 3)  # Estimate airlight
    A = torch.max(A1, dim=1, keepdim=True)[0]  # Max per image

    delta = zeta / (min_I.sqrt() + 1e-6)  # Prevent division by zero
    est_tr_proposed = 1 / (1 + (MAX * 10 ** (-0.05 * delta)) / (A - min_I + 1e-6))

    tr1 = (min_I >= A).float()
    tr2 = (min_I < A).float() * est_tr_proposed
    tr4 = tr1 * est_tr_proposed
    tr3_max = torch.max(tr4, dim=[1, 2, 3], keepdim=True)[0]
    tr3_max[tr3_max == 0] = 1  # Prevent division by zero
    tr3 = tr4 / tr3_max
    est_tr_proposed = tr2 + tr3

    # Apply morphological closing
    est_tr_proposed = closing(est_tr_proposed.cpu().numpy(), footprint_rectangle((3, 3)))
    est_tr_proposed = torch.tensor(est_tr_proposed, device=device)

    est_tr_proposed = cal_trans(I_batch, est_tr_proposed, 1, 0.5)  # Compute refined transmission
    r = defog(I_batch, est_tr_proposed, A1, 0.9)  # Dehaze images

    return r, est_tr_proposed.unsqueeze(1), A.unsqueeze(1)

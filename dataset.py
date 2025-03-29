import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class HazyImageDataset(Dataset):
    def __init__(self, folder_path, img_size=(256, 256)):
        """Initialize dataset, loading image paths only (lazy loading)."""
        self.folder_path = folder_path
        self.img_size = img_size
        self.image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Loads a single image and preprocesses it."""
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, self.img_size)  # Resize
        img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
        
        # Convert to PyTorch tensor: (H, W, C) → (C, H, W)
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        return img_tensor

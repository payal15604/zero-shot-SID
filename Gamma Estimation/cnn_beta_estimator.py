import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import cv2
import numpy as np
import os

# CNN Model for Beta Estimation
class BetaCNN(nn.Module):
    def __init__(self):
        super(BetaCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 64 * 64, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Output β
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Dataset Class
class BetaDataset(Dataset):
    def __init__(self, hazy_folder, csv_file, transform=None):
        self.hazy_folder = hazy_folder
        self.beta_data = pd.read_csv(csv_file, header=None, names=["image", "beta"])
        self.transform = transform

    def __len__(self):
        return len(self.beta_data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.hazy_folder, self.beta_data.iloc[idx, 0])
        image = cv2.imread(img_name, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.float32) / 255.0
        beta = np.array([self.beta_data.iloc[idx, 1]], dtype=np.float32)

        if self.transform:
            image = self.transform(image)

        return image, beta

# Training Function
def train_model(hazy_folder, csv_file, epochs=10, batch_size=16, lr=1e-3):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128))  # Resize images for CNN
    ])

    dataset = BetaDataset(hazy_folder, csv_file, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = BetaCNN().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for images, betas in dataloader:
            images, betas = images.cuda(), betas.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, betas)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")

    torch.save(model.state_dict(), "beta_cnn.pth")

# Example usage
if __name__ == "__main__":
    train_model("data/hazy", "beta_values.csv", epochs=20)

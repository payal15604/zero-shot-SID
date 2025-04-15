import os
import torch
from train_2 import i_net, optimizer, dataloader, haze_net, compute_transmission, estimate_atmospheric_light
from pytorch_msssim import ssim

# Configuration for resuming
resume_checkpoint_path = "dehazeCOPY.pth"
start_epoch = 1000
total_epochs = 1500
new_learning_rate = 0.01  # ⬅️ Your custom LR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Resuming on device:", device)

# Resume model and optimizer states if checkpoint exists
if os.path.exists(resume_checkpoint_path):
    print(f"Resuming training from checkpoint: {resume_checkpoint_path}")
    checkpoint = torch.load(resume_checkpoint_path, map_location=device)
    i_net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1

    # Change learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_learning_rate
    print(f"Learning rate changed to {new_learning_rate}")
else:
    raise FileNotFoundError("No checkpoint found to resume from.")

# Training loop
i_net.train()
for epoch in range(start_epoch, total_epochs):
    epoch_loss = 0.0
    print(f"Epoch: {epoch}")

    for hazy_img in dataloader:
        hazy_img = hazy_img.to(device)

        with torch.no_grad():
            gamma = haze_net(hazy_img)

        transmission = compute_transmission(hazy_img, device)
        t_power_gamma = torch.pow(transmission, gamma.view(-1, 1, 1, 1))
        A = estimate_atmospheric_light(hazy_img)
        A = A.squeeze().view(-1, 3, 1, 1)

        J_haze_free = i_net(hazy_img)
        reconstructed_hazy = A * (1 - t_power_gamma) + t_power_gamma * J_haze_free

        loss = 1 - ssim(reconstructed_hazy, hazy_img, data_range=1.0, size_average=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch}/{end_epoch}], Loss: {epoch_loss/len(dataloader):.4f}")
    # Save main resume checkpoint
    torch.save({
        "epoch": epoch,
        "model_state_dict": i_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, resume_checkpoint_path)

    # Save a uniquely named checkpoint every 100 epochs
    if (epoch + 1) % 100 == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": i_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, f"checkpoint_epoch_{epoch + 1}.pth")
        print(f"Saved full checkpoint at epoch {epoch + 1}")

print("\nResumed training complete.")

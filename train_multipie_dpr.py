# train_multipie_dpr.py
# Name: Yunpei Gu (Team: Yunpei Gu)
# Class: CS 7180 Advanced Perception
# Date: 2025-10-15

'''
Trains the DPR relighting model using paired Multi-PIE images.
'''

import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from model.defineHourglass_512_gray_skip import HourglassNet
import datetime
import matplotlib.pyplot as plt
import time

# ===============================
# 1. Dataset Definition
# ===============================
import json

class MultiPiePairDataset(Dataset):
    """
    Dataset for Multi-PIE paired images with source and target lighting conditions.
    Loads image pairs and their corresponding spherical harmonic lighting coefficients.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.src_dir = os.path.join(data_dir, "source")
        self.tgt_dir = os.path.join(data_dir, "target")
        self.src_light = np.load(os.path.join(data_dir, "source_light.npy"))
        self.tgt_light = np.load(os.path.join(data_dir, "target_light.npy"))

        mapping_file = os.path.join(data_dir, "pairs_mapping.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                self.mapping = json.load(f)
        else:
            # fallback: assume identical filenames
            self.mapping = [{"source": f, "target": f} for f in sorted(os.listdir(self.src_dir))]

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        pair = self.mapping[idx]
        src_img_path = os.path.join(self.src_dir, pair["source"])
        tgt_img_path = os.path.join(self.tgt_dir, pair["target"])

        src_img = Image.open(src_img_path).convert("RGB")
        tgt_img = Image.open(tgt_img_path).convert("RGB")

        src_tensor = self.transform(src_img)
        tgt_tensor = self.transform(tgt_img)

        src_light = torch.tensor(self.src_light[idx], dtype=torch.float32).view(9, 1, 1)
        tgt_light = torch.tensor(self.tgt_light[idx], dtype=torch.float32).view(9, 1, 1)


        return src_tensor, tgt_tensor, src_light, tgt_light

# ===============================
# 2. Train Function (without GAN)
# ===============================
def gradient_loss(pred, target):
    """
    Computes spatial gradient loss to preserve image structure.
    Measures L1 difference between horizontal and vertical gradients.
    """
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return torch.mean(torch.abs(pred_dx - tgt_dx)) + torch.mean(torch.abs(pred_dy - tgt_dy))

def train_dpr(data_dir, epochs=20, batch_size=2, lr=1e-4, save_dir="trained_model"):
    """
    Trains the DPR model using L1, gradient, and feature losses (no adversarial loss).
    
    Args:
        data_dir: Path to Multi-PIE paired dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        save_dir: Directory to save checkpoints and logs
    """
    # === Device Setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"GPU Count: {torch.cuda.device_count()}", flush=True)
        print(f"Current GPU: {torch.cuda.current_device()}", flush=True)
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB", flush=True)
        print(f"CUDA Version: {torch.version.cuda}", flush=True)
    else:
        print("⚠️ No GPU available, using CPU", flush=True)
    print(f"{'-'*50}\n", flush=True)

    # === Training Setup ===
    # Record start time
    start_time = time.time()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # store loss history
    loss_history = []

    # === Data Loading ===
    # Dataset and Loader
    print("✅ Step 1: Creating dataset...", flush=True)
    dataset = MultiPiePairDataset(data_dir)
    print(f"✅ Step 2: Dataset loaded, total {len(dataset)} samples", flush=True)

    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print("✅ Step 3: DataLoader ready!", flush=True)

    # === Model Initialization ===
    # Model
    model = HourglassNet(baseFilter=16, gray=True).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    os.makedirs(save_dir, exist_ok=True)

    # === Sanity Check ===
    src, tgt, src_light, tgt_light = next(iter(dataloader))
    print(f"✅ Loaded one batch: src={src.shape}, tgt={tgt.shape}", flush=True)
    with torch.no_grad():
        pred, _ = model(src.to(device), tgt_light.to(device), skip_count=0)
        print(f"✅ Forward check ok: pred={pred.shape}", flush=True)
    print("✅ Step 6: Passed sanity check, starting training...\n", flush=True)

    # === Training Loop ===
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        skip_count = max(0, min(epoch - 5, 4))
        λ = 0.5  # weight for feature loss

        # for src, tgt, src_light, tgt_light in dataloader:
        for i, (src, tgt, src_light, tgt_light) in enumerate(dataloader):
            src, tgt = src.to(device), tgt.to(device)
            src_light, tgt_light = src_light.to(device), tgt_light.to(device)

            # optimizer
            optimizer.zero_grad()

            # Forwarding (skip)
            # pred, _ = model(src, tgt_light, skip_count=skip_count)

            # # Forward (skip, featureloss) : relight from source → target lighting ; self-lighting (same light)
            pred, zf_tgt = model(src, tgt_light, skip_count=skip_count)
            _, zf_src = model(src, src_light, skip_count=skip_count)


            # Compute loss (L1 only)
            # loss = criterion(pred, tgt)

            # Compute loss (L1 + 0.1×Gradient loss)
            # l1_loss = criterion(pred, tgt)
            # grad_loss = gradient_loss(pred, tgt)
            # loss = l1_loss + 0.1 * grad_loss

            # Compute loss (L1 + 0.1×Gradient loss + feature loss)
            l1_loss = criterion(pred, tgt)
            grad_loss = gradient_loss(pred, tgt)
            feature_loss = torch.mean(torch.abs(zf_tgt - zf_src))
            loss = l1_loss + 0.1 * grad_loss + λ * feature_loss


            # Backwarding
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * src.size(0)

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{i+1}/{len(dataloader)}] | Loss: {loss.item():.6f}", flush=True)
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{i+1}/{len(dataloader)}] "
                f"| Total: {loss.item():.4f}", flush=True)


        avg_loss = total_loss / len(dataset)
        loss_history.append(avg_loss)

        print(f"📉 Epoch [{epoch+1}/{epochs}] | Avg Loss: {avg_loss:.6f}", flush=True)

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"trained_multipie_epoch{epoch+1}_{timestamp}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved to {save_path}\n", flush=True)


    # ===== Save and Plot Loss Curve =====
    loss_array = np.array(loss_history)
    np.save(os.path.join(save_dir, "training_loss.npy"), loss_array)
    print(f"💾 Saved loss history -> {os.path.join(save_dir, 'training_loss.npy')}")

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_loss_curve.png"))
    print(f"📊 Saved training loss curve -> training_loss_curve.png")
    
    end_time = time.time()
    total_seconds = int(end_time - start_time)
    hrs = total_seconds // 3600
    mins = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    print(f"⏱️ Total training time: {hrs}h {mins}m {secs}s")
    print(f"✅ Final average loss after {epochs} epochs: {loss_history[-1]:.6f}")


# ===============================
# 3. Train Function (with GAN)
# ===============================
def train_dpr_gan(data_dir, epochs=20, batch_size=2, lr=1e-4, save_dir="trained_model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"GPU Count: {torch.cuda.device_count()}", flush=True)
        print(f"Current GPU: {torch.cuda.current_device()}", flush=True)
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB", flush=True)
        print(f"CUDA Version: {torch.version.cuda}", flush=True)
    else:
        print("⚠️ No GPU available, using CPU", flush=True)
    print(f"{'-'*50}\n", flush=True)

    # Record start time
    start_time = time.time()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # === Dataset ===
    print("✅ Step 1: Creating dataset...", flush=True)
    dataset = MultiPiePairDataset(data_dir)
    print(f"✅ Step 2: Dataset loaded, total {len(dataset)} samples", flush=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print("✅ Step 3: DataLoader ready!", flush=True)

    # === Model & Discriminator ===
    model = HourglassNet(baseFilter=16, gray=True).to(device)
    criterion = nn.L1Loss()
    optimizer_G = optim.Adam(model.parameters(), lr=lr)

    # PatchGAN discriminator
    class PatchDiscriminator(nn.Module):
        def __init__(self, in_channels=1):
            super(PatchDiscriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, 64, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 1, 4, 1, 1)
            )

        def forward(self, x):
            return self.model(x)

    D = PatchDiscriminator(in_channels=1).to(device)
    optimizer_D = optim.Adam(D.parameters(), lr=lr)

    os.makedirs(save_dir, exist_ok=True)
    λ = 0.5     # feature loss weight
    gan_weight = 0.01    # GAN loss weight

    # === Sanity check ===
    src, tgt, src_light, tgt_light = next(iter(dataloader))
    with torch.no_grad():
        pred, _ = model(src.to(device), tgt_light.to(device), skip_count=0)
    print(f"✅ Forward check ok: pred={pred.shape}", flush=True)

    # === Training loop ===
    loss_history = []
    for epoch in range(epochs):
        model.train()
        total_loss_G, total_loss_D = 0.0, 0.0
        skip_count = max(0, min(epoch - 5, 4))
        print(f"🧩 Epoch {epoch+1}: using {skip_count} skip connections", flush=True)

        for i, (src, tgt, src_light, tgt_light) in enumerate(dataloader):
            src, tgt = src.to(device), tgt.to(device)
            src_light, tgt_light = src_light.to(device), tgt_light.to(device)

            # ============================================
            # 1️⃣ Train Discriminator (LSGAN)
            # ============================================
            with torch.no_grad():
                fake_pred, _ = model(src, tgt_light, skip_count=skip_count)
            real_out = D(tgt)
            fake_out = D(fake_pred.detach())

            loss_D_real = torch.mean((real_out - 1) ** 2)
            loss_D_fake = torch.mean(fake_out ** 2)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # ============================================
            # 2️⃣ Train Generator (HourglassNet)
            # ============================================
            pred, zf_tgt = model(src, tgt_light, skip_count=skip_count)
            _, zf_src = model(src, src_light, skip_count=skip_count)

            pred_out = D(pred)
            loss_GAN = torch.mean((pred_out - 1) ** 2)
            l1_loss = criterion(pred, tgt)
            grad_loss = gradient_loss(pred, tgt)
            feature_loss = torch.mean(torch.abs(zf_tgt - zf_src))

            # total generator loss
            loss_G = l1_loss + 0.1 * grad_loss + λ * feature_loss + gan_weight * loss_GAN

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            total_loss_G += loss_G.item() * src.size(0)
            total_loss_D += loss_D.item() * src.size(0)

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{i+1}/{len(dataloader)}] "
                      f"| L1={l1_loss.item():.4f} | Grad={grad_loss.item():.4f} "
                      f"| Feat={feature_loss.item():.4f} | GAN={loss_GAN.item():.4f} "
                      f"| D={loss_D.item():.4f} | Total_G={loss_G.item():.4f}", flush=True)

        avg_loss_G = total_loss_G / len(dataset)
        avg_loss_D = total_loss_D / len(dataset)
        loss_history.append(avg_loss_G)

        print(f"📉 Epoch [{epoch+1}/{epochs}] | Avg G Loss: {avg_loss_G:.6f} | Avg D Loss: {avg_loss_D:.6f}", flush=True)

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}_{timestamp}.pth")
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': model.state_dict(),
                'discriminator_state_dict': D.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'loss_history': loss_history
            }, save_path)
            print(f"✅ Model saved to {save_path}\n", flush=True)

    # === Save training loss curve ===
    loss_array = np.array(loss_history)
    np.save(os.path.join(save_dir, "training_loss.npy"), loss_array)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Generator Loss")
    plt.title("Training Loss Curve (with GAN)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_loss_curve.png"))
    print(f"📊 Saved training loss curve -> training_loss_curve.png")

    end_time = time.time()
    total_seconds = int(end_time - start_time)
    hrs = total_seconds // 3600
    mins = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    print(f"⏱️ Total training time: {hrs}h {mins}m {secs}s")
    print(f"✅ Final average generator loss after {epochs} epochs: {loss_history[-1]:.6f}")

# ===============================
# 4. Main
# ===============================
if __name__ == "__main__":
    data_dir = "data/Multi_Pie/pairs"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = f"trained_model_{timestamp}_L1GradientFeatureGAN_skip"

    # train_dpr(data_dir=data_dir, epochs=100, batch_size=2, lr=1e-4, save_dir=save_dir)
    train_dpr_gan(data_dir=data_dir, epochs=100, batch_size=2, lr=1e-4, save_dir=save_dir)

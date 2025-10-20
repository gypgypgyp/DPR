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
import json
import torch.nn.functional as F

# ===============================
# 1. Dataset Definition
# ===============================
class MultiPiePairDataset(Dataset):
    """
    Dataset for Multi-PIE paired images with source and target lighting conditions.
    Loads image pairs and their corresponding spherical harmonic lighting coefficients.
    """
    def __init__(self, data_dir, train=True):
        self.data_dir = data_dir
        self.src_dir = os.path.join(data_dir, "source")
        self.tgt_dir = os.path.join(data_dir, "target")
        self.src_light = np.load(os.path.join(data_dir, "source_light.npy"))
        self.tgt_light = np.load(os.path.join(data_dir, "target_light.npy"))

        # load training data
        if train:
            mapping_file = os.path.join(data_dir, "pairs_mapping_train.json")
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r') as f:
                    self.mapping = json.load(f)
            else:
                # fallback: assume identical filenames
                self.mapping = [{"source": f, "target": f} for f in sorted(os.listdir(self.src_dir))]
        # load validation data
        else:
            mapping_file = os.path.join(data_dir, "pairs_mapping_val.json")
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
# 2. Train Function
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


def compute_si_mse(pred, target):
    """Scale-Invariant MSE  (Eq. 7 of paper)."""
    # flatten
    pred_f = pred.view(pred.size(0), -1)
    target_f = target.view(target.size(0), -1)

    # best scale α = (target·pred) / (pred·pred)
    alpha = (target_f * pred_f).sum(dim=1, keepdim=True) / (pred_f * pred_f).sum(dim=1, keepdim=True)
    # alpha = alpha.unsqueeze(-1)  # (B,1,1)
    alpha = alpha[:, :, None, None]


    # compute SI-MSE
    mse = ((target - alpha * pred) ** 2).mean(dim=[1,2,3])
    return mse.mean().item()


def compute_si_l2(pred_light, gt_light):
    """Scale-Invariant L2 between predicted and target SH lighting."""
    # pred_light, gt_light shape: [B, 9, 1, 1] or [B, 9]
    pred_light = pred_light.view(pred_light.size(0), -1)
    gt_light = gt_light.view(gt_light.size(0), -1)

    eps = 1e-8  # prevent denominator = 0
    beta = (gt_light * pred_light).sum(dim=1, keepdim=True) / (
        (pred_light * pred_light).sum(dim=1, keepdim=True) + eps
    )
    l2 = ((gt_light - beta * pred_light) ** 2).mean(dim=1)
    return l2.mean().item()

# ===============================
# Train Function (no GAN)
# ===============================
def train_dpr(
    data_dir, epochs=20, batch_size=2, lr=1e-4, save_dir="trained_model",
    loss_mode="L1" ):
    """
    Trains the DPR model with adversarial loss using PatchGAN discriminator.
    Combines L1, gradient, feature, and GAN losses for improved realism.
    
    Args:
        data_dir: Path to Multi-PIE paired dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate for both generator and discriminator
        save_dir: Directory to save checkpoints and logs
        loss_mode: "L1", "L1+Grad", "L1+Grad+Feat"
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

    # Record start time
    start_time = time.time()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # === Dataset ===
    print("✅ Step 1: Creating dataset...", flush=True)
    dataset = MultiPiePairDataset(data_dir)
    dataset_val = MultiPiePairDataset(data_dir, train=False)
    print(len(dataset), len(dataset_val))
    print(f"✅ Step 2: Dataset loaded, total {len(dataset)} samples", flush=True)

    # split data to training data and testing data
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=0)
    print("✅ Step 3: DataLoader ready!", flush=True)

    # === Model & Discriminator ===
    model = HourglassNet(baseFilter=16, gray=True).to(device)
    criterion = nn.L1Loss()
    optimizer_G = optim.Adam(model.parameters(), lr=lr)


    os.makedirs(save_dir, exist_ok=True)
    λ = 0.5     # feature loss weight

    # === Training loop ===
    loss_history = []
    val_si_mse_list = []
    val_si_l2_list = []
    for epoch in range(epochs):
        # === Training phase ===
        model.train()
        total_loss_G, total_loss_D = 0.0, 0.0
        skip_count = max(0, min(epoch - 5, 4))
        print(f"🧩 Epoch {epoch+1}: using {skip_count} skip connections", flush=True)

        for i, (src, tgt, src_light, tgt_light) in enumerate(dataloader):
            src, tgt = src.to(device), tgt.to(device)
            src_light, tgt_light = src_light.to(device), tgt_light.to(device)


            #  === Train Generator (HourglassNet) === 
            pred, feat_tgt, _ = model(src, tgt_light, skip_count=skip_count)
            _, feat_src, _ = model(src, src_light, skip_count=skip_count)

            l1_loss = criterion(pred, tgt)

            if loss_mode == "L1":
                loss_G = l1_loss
            elif loss_mode == "L1+Grad":
                grad_loss = gradient_loss(pred, tgt)
                loss_G = l1_loss + grad_loss
            elif loss_mode == "L1+Grad+Feat":
                grad_loss = gradient_loss(pred, tgt)
                feature_loss = torch.mean((feat_tgt - feat_src) ** 2)
                loss_G = l1_loss + grad_loss + λ * feature_loss
            else:
                raise ValueError(f"Unknown loss_mode: {loss_mode}")

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
            total_loss_G += loss_G.item() * src.size(0)

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{i+1}/{len(dataloader)} | Total_G={loss_G.item():.4f}", flush=True)

        avg_loss_G = total_loss_G / len(dataset)
        loss_history.append(avg_loss_G)

        # === Validation phase ===
        model.eval()
        val_loss = 0.0
        si_mse_total, si_l2_total = 0.0, 0.0
        with torch.no_grad():
            for src, tgt, src_light, tgt_light in dataloader_val:
                
                src, tgt = src.to(device), tgt.to(device)
                src_light, tgt_light = src_light.to(device), tgt_light.to(device)
                pred, _, pred_light = model(src, tgt_light, skip_count=skip_count)

                # === compute Si-MSE ===
                si_mse_total += compute_si_mse(pred, tgt) * src.size(0)

                # === compute Si-L2 ===
                si_l2_total += compute_si_l2(pred_light, tgt_light) * src.size(0)

        avg_si_mse = si_mse_total / len(dataset_val)
        avg_si_l2 = si_l2_total / len(dataset_val)
        val_si_mse_list.append(avg_si_mse)
        val_si_l2_list.append(avg_si_l2)


        print(f"📉 Epoch [{epoch+1}/{epochs}] | Avg Loss: {avg_loss_G:.6f} | Si-MSE={avg_si_mse:.6f} |  Si-L2={avg_si_l2:.6f}", flush=True)


        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}_{timestamp}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved to {save_path}\n", flush=True)

    # === Save training loss curve ===
    # loss_array = np.array(loss_history)
    # np.save(os.path.join(save_dir, "training_loss.npy"), loss_array)
    # plt.figure(figsize=(6, 4))
    # plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linewidth=1.5)
    # plt.xlabel("Epoch")
    # plt.ylabel("Generator Loss")
    # plt.title("Training Loss Curve (with GAN)")
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "training_loss_curve.png"))
    # print(f"📊 Saved training loss curve -> training_loss_curve.png")

   # === Save training & validation curves ===
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "train_loss.npy"), np.array(loss_history))
    np.save(os.path.join(save_dir, "val_si_mse.npy"), np.array(val_si_mse_list))
    np.save(os.path.join(save_dir, "val_si_l2.npy"),  np.array(val_si_l2_list))  

    # plt.figure(figsize=(6,4))
    # plt.plot(range(1, len(loss_history)+1), loss_history, label="Train Loss", marker='o')
    # plt.plot(range(1, len(val_si_mse_list)+1), val_si_mse_list, label="Val SI-MSE", marker='s')
    # plt.plot(range(1, len(val_si_l2_list)+1),  val_si_l2_list,  label="Val SI-L2",  marker='^')   
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss / Metric")
    # plt.title("Training vs Validation (without GAN)")
    # plt.legend()
    # plt.grid(True, linestyle="--", alpha=0.6)
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "train_val_curve.png"))
    # plt.close()
    # print("📊 Saved training-validation curve with SI-MSE & SI-L2")


    # end_time = time.time()
    # total_seconds = int(end_time - start_time)
    # hrs = total_seconds // 3600
    # mins = (total_seconds % 3600) // 60
    # secs = total_seconds % 60
    # print(f"⏱️ Total training time: {hrs}h {mins}m {secs}s")
    # print(f"✅ Final average generator loss after {epochs} epochs: {loss_history[-1]:.6f}")

    epochs_range = range(1, len(loss_history) + 1)

    # -----------------------------
    # Train Loss
    # -----------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, loss_history, marker='o', label="Train Loss", color='tab:blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve (Generator)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "train_loss_curve.png"))
    plt.close()
    print("📊 Saved training loss curve -> train_loss_curve.png")

    # -----------------------------
    # SI-MSE & SI-L2
    # -----------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, val_si_mse_list, marker='s', label="Val SI-MSE", color='tab:orange')
    plt.plot(epochs_range, val_si_l2_list, marker='^', label="Val SI-L2", color='tab:green')
    plt.xlabel("Epoch")
    plt.ylabel("Validation Metric")
    plt.title("Validation Curves (SI-MSE & SI-L2)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_metrics_curve.png"))
    plt.close()
    print("📊 Saved validation metrics curve -> val_metrics_curve.png")

# ===============================
# Train Function (with GAN)
# ===============================
def train_dpr_gan(data_dir, epochs=20, batch_size=2, lr=1e-4, save_dir="trained_model"):
    """
    Trains the DPR model with adversarial loss using PatchGAN discriminator.
    Combines L1, gradient, feature, and GAN losses for improved realism.
    
    Args:
        data_dir: Path to Multi-PIE paired dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate for both generator and discriminator
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

    # Record start time
    start_time = time.time()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # === Dataset ===
    print("✅ Step 1: Creating dataset...", flush=True)
    dataset = MultiPiePairDataset(data_dir)
    dataset_val = MultiPiePairDataset(data_dir, train=False)
    print(len(dataset), len(dataset_val))
    print(f"✅ Step 2: Dataset loaded, total {len(dataset)} samples", flush=True)

    # split data to training data and testing data
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=0)
    print("✅ Step 3: DataLoader ready!", flush=True)

    # === Model & Discriminator ===
    model = HourglassNet(baseFilter=16, gray=True).to(device)
    criterion = nn.L1Loss()
    optimizer_G = optim.Adam(model.parameters(), lr=lr)

    # PatchGAN discriminator
    class PatchDiscriminator(nn.Module):
        """
        PatchGAN discriminator that classifies image patches as real/fake.
        Uses strided convolutions with instance normalization.
        """
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

    # # === Sanity check ===
    # src, tgt, src_light, tgt_light = next(iter(dataloader))
    # with torch.no_grad():
    #     pred, _ = model(src.to(device), tgt_light.to(device), skip_count=0)
    # print(f"✅ Forward check ok: pred={pred.shape}", flush=True)

    # === Training loop ===
    loss_history = []
    val_si_mse_list = []
    val_si_l2_list = []
    for epoch in range(epochs):
        # === Training phase ===
        model.train()
        total_loss_G, total_loss_D = 0.0, 0.0
        skip_count = max(0, min(epoch - 5, 4))
        print(f"🧩 Epoch {epoch+1}: using {skip_count} skip connections", flush=True)

        for i, (src, tgt, src_light, tgt_light) in enumerate(dataloader):
            src, tgt = src.to(device), tgt.to(device)
            src_light, tgt_light = src_light.to(device), tgt_light.to(device)

            #  === Train Discriminator  === 
            with torch.no_grad():
                fake_pred, _, _ = model(src, tgt_light, skip_count=skip_count)
            real_out = D(tgt)
            fake_out = D(fake_pred.detach())

            loss_D_real = torch.mean((real_out - 1) ** 2)
            loss_D_fake = torch.mean(fake_out ** 2)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            #  === Train Generator (HourglassNet) === 
            pred, feat_tgt, _ = model(src, tgt_light, skip_count=skip_count)
            _, feat_src, _ = model(src, src_light, skip_count=skip_count)

            pred_out = D(pred)
            loss_GAN = torch.mean((pred_out - 1) ** 2)
            l1_loss = criterion(pred, tgt)
            grad_loss = gradient_loss(pred, tgt)
            feature_loss = torch.mean((feat_tgt - feat_src) ** 2)

            # total generator loss
            loss_G = l1_loss + grad_loss + λ * feature_loss + loss_GAN

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

        # === Validation phase ===
        model.eval()
        val_loss = 0.0
        si_mse_total, si_l2_total = 0.0, 0.0
        with torch.no_grad():
            for src, tgt, src_light, tgt_light in dataloader_val:
                
                src, tgt = src.to(device), tgt.to(device)
                src_light, tgt_light = src_light.to(device), tgt_light.to(device)
                pred, _, pred_light = model(src, tgt_light, skip_count=skip_count)

                # === compute Si-MSE ===
                si_mse_total += compute_si_mse(pred, tgt) * src.size(0)

                # === compute Si-L2 ===
                si_l2_total += compute_si_l2(pred_light, tgt_light) * src.size(0)

        avg_si_mse = si_mse_total / len(dataset_val)
        avg_si_l2 = si_l2_total / len(dataset_val)
        val_si_mse_list.append(avg_si_mse)
        val_si_l2_list.append(avg_si_l2)


        print(f"📉 Epoch [{epoch+1}/{epochs}] | Avg G Loss: {avg_loss_G:.6f} | Avg D Loss: {avg_loss_D:.6f} | Si-MSE={avg_si_mse:.6f} |  Si-L2={avg_si_l2:.6f}", flush=True)


        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}_{timestamp}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved to {save_path}\n", flush=True)

    # === Save training loss curve ===
    # loss_array = np.array(loss_history)
    # np.save(os.path.join(save_dir, "training_loss.npy"), loss_array)
    # plt.figure(figsize=(6, 4))
    # plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o', linewidth=1.5)
    # plt.xlabel("Epoch")
    # plt.ylabel("Generator Loss")
    # plt.title("Training Loss Curve (with GAN)")
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "training_loss_curve.png"))
    # print(f"📊 Saved training loss curve -> training_loss_curve.png")


   # === Save training & validation curves ===
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "train_loss.npy"), np.array(loss_history))
    np.save(os.path.join(save_dir, "val_si_mse.npy"), np.array(val_si_mse_list))
    np.save(os.path.join(save_dir, "val_si_l2.npy"),  np.array(val_si_l2_list))  # ✅ 保存SI-L2

    # plt.figure(figsize=(6,4))
    # plt.plot(range(1, len(loss_history)+1), loss_history, label="Train Loss", marker='o')
    # plt.plot(range(1, len(val_si_mse_list)+1), val_si_mse_list, label="Val SI-MSE", marker='s')
    # plt.plot(range(1, len(val_si_l2_list)+1),  val_si_l2_list,  label="Val SI-L2",  marker='^')   # ✅ 新增这条线
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss / Metric")
    # plt.title("Training vs Validation (with GAN)")
    # plt.legend()
    # plt.grid(True, linestyle="--", alpha=0.6)
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "train_val_curve.png"))
    # plt.close()
    # print("📊 Saved training-validation curve with SI-MSE & SI-L2")

    epochs_range = range(1, len(loss_history) + 1)

    # -----------------------------
    # 图1：训练损失曲线（Train Loss）
    # -----------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, loss_history, marker='o', label="Train Loss", color='tab:blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve (Generator)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "train_loss_curve.png"))
    plt.close()
    print("📊 Saved training loss curve -> train_loss_curve.png")

    # -----------------------------
    # 图2：验证指标曲线（SI-MSE & SI-L2）
    # -----------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, val_si_mse_list, marker='s', label="Val SI-MSE", color='tab:orange')
    plt.plot(epochs_range, val_si_l2_list, marker='^', label="Val SI-L2", color='tab:green')
    plt.xlabel("Epoch")
    plt.ylabel("Validation Metric")
    plt.title("Validation Curves (SI-MSE & SI-L2)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_metrics_curve.png"))
    plt.close()
    print("📊 Saved validation metrics curve -> val_metrics_curve.png")



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

    # train_dpr(data_dir, epochs=100, batch_size=2, lr=1e-4,
    #         save_dir=f"trained_model/{timestamp}_L1_skip",
    #         loss_mode="L1")
    # train_dpr(data_dir, epochs=100, batch_size=2, lr=1e-4,
    #           save_dir=f"trained_model/{timestamp}_L1Grad_skip",
    #           loss_mode="L1+Grad")
    # train_dpr(data_dir, epochs=100, batch_size=2, lr=1e-4,
    #         save_dir=f"trained_model/{timestamp}_L1GradFeat_skip",
    #         loss_mode="L1+Grad+Feat")
    train_dpr_gan(data_dir=data_dir, epochs=100, batch_size=2, lr=1e-4, 
            save_dir=f"trained_model/{timestamp}_L1GradientFeatureGAN_skip")

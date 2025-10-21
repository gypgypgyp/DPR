# test_compare_models.py
# Author: Yunpei Gu
# CS7180 Advanced Perception
# Date: 2025-10-19
# Purpose: Compare relighting results across multiple trained models.

import os
import torch
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model.defineHourglass_512_gray_skip import HourglassNet
import json
import time
import datetime


# ===============================
# 1. Configuration
# ===============================

def test_multiple_model():

    data_dir = "data/Multi_Pie/pairs"

    # List of model directories to compare
    model_dirs = [
        "trained_model/20251021_0039_L1_skip",
        "trained_model/20251021_0040_L1GradFeat_skip",
        "trained_model/20251021_0041_L1GradientFeatureGAN_skip"
    ]

    gan_dir = "trained_model/20251021_0041_L1GradientFeatureGAN_skip"

    # Create a timestamped output folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"comparison_outputs/output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # ===============================
    # 2. Load one test pair
    # ===============================
    with open(os.path.join(data_dir, "pairs_mapping.json"), 'r') as f:
        pairs = json.load(f)

    test_idx = 0  # choose which pair to test
    pair = pairs[test_idx]
    src_path = os.path.join(data_dir, "source", pair["source"])
    tgt_path = os.path.join(data_dir, "target", pair["target"])

    src_img = Image.open(src_path).convert("RGB")
    tgt_img = Image.open(tgt_path).convert("RGB")

    # Convert both images to grayscale tensors
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    src_tensor = transform(src_img).unsqueeze(0).to(device)
    tgt_tensor = transform(tgt_img).unsqueeze(0).to(device)

    src_light = torch.tensor(np.load(os.path.join(data_dir, "source_light.npy"))[test_idx],
                            dtype=torch.float32).view(1, 9, 1, 1).to(device)
    tgt_light = torch.tensor(np.load(os.path.join(data_dir, "target_light.npy"))[test_idx],
                            dtype=torch.float32).view(1, 9, 1, 1).to(device)

    # ===============================
    # 3. Compare all models
    # ===============================
    results = []

    for model_dir in model_dirs:
        # Find model weight files
        model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
        if not model_files:
            print(f"⚠️ No .pth file found in {model_dir}")
            continue

        # Load the newest model checkpoint
        model_path = os.path.join(model_dir, sorted(model_files)[-1])
        print(f"📦 Loading model: {model_path}")

        model = HourglassNet(baseFilter=16, gray=True).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Inference
        with torch.no_grad():
            pred, _, _= model(src_tensor, tgt_light, skip_count=0)

        # Save predicted image
        model_name = os.path.basename(model_dir)
        pred_out_path = os.path.join(output_dir, f"pred_{model_name}.png")
        utils.save_image(pred, pred_out_path)
        print(f"✅ Saved result: {pred_out_path}")

        results.append((model_name, pred.cpu().squeeze().numpy()))

    # ===============================
    # 4. Visualization & Comparison
    # ===============================
    fig, axes = plt.subplots(1, len(results) + 2, figsize=(4*(len(results)+2), 4))

    # Convert source and target to grayscale numpy arrays
    src_gray = np.array(src_img.convert("L"))
    tgt_gray = np.array(tgt_img.convert("L"))

    axes[0].imshow(src_gray, cmap='gray')
    axes[0].set_title("Source Image (Grayscale)")
    axes[1].imshow(tgt_gray, cmap='gray')
    axes[1].set_title("Target Image (Grayscale)")

    # Plot predicted results
    for i, (name, img_data) in enumerate(results):
        # axes[i+2].imshow(np.transpose(img_data, (1, 2, 0)), cmap='gray')
        img_np = img_data
        if img_np.ndim == 2:
            # shape: (H, W)
            axes[i+2].imshow(img_np, cmap='gray')
        elif img_np.ndim == 3:
            # shape: (C, H, W) 或 (H, W, C)
            if img_np.shape[0] == 1:
                img_np = img_np.squeeze(0)  # (H, W)
                axes[i+2].imshow(img_np, cmap='gray')
            elif img_np.shape[0] in [3, 4]:
                img_np = np.transpose(img_np, (1, 2, 0))
                axes[i+2].imshow(img_np)
            else:
                axes[i+2].imshow(img_np[0], cmap='gray')
        else:
            print(f"⚠️ Unexpected shape: {img_np.shape}")

        axes[i+2].set_title(name.replace("trained_model_", ""))

    # Hide all axes
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    print(f"{model_name} -> pred shape: {pred.shape}")
    plt.show()

    print("🎨 Comparison figure saved -> model_comparison.png")

def test_model_multiple_epoch(model_dir):

    # ===============================
    # 1. Configuration
    # ===============================
    data_dir = "data/Multi_Pie/pairs"

    # choose one test sample
    test_idx = 0

    # output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"comparison_outputs/{os.path.basename(model_dir)}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # ===============================
    # 2. Load test pair
    # ===============================
    with open(os.path.join(data_dir, "pairs_mapping.json"), 'r') as f:
        pairs = json.load(f)

    pair = pairs[test_idx]
    src_path = os.path.join(data_dir, "source", pair["source"])
    tgt_path = os.path.join(data_dir, "target", pair["target"])

    src_img = Image.open(src_path).convert("RGB")
    tgt_img = Image.open(tgt_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    src_tensor = transform(src_img).unsqueeze(0).to(device)
    tgt_tensor = transform(tgt_img).unsqueeze(0).to(device)

    src_light = torch.tensor(np.load(os.path.join(data_dir, "source_light.npy"))[test_idx],
                            dtype=torch.float32).view(1, 9, 1, 1).to(device)
    tgt_light = torch.tensor(np.load(os.path.join(data_dir, "target_light.npy"))[test_idx],
                            dtype=torch.float32).view(1, 9, 1, 1).to(device)

    # ===============================
    # 3. Loop over all checkpoints
    # ===============================
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".pth")])
    if not model_files:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")

    results = []

    for ckpt in model_files:
        ckpt_path = os.path.join(model_dir, ckpt)
        print(f"📦 Loading checkpoint: {ckpt_path}")

        model = HourglassNet(baseFilter=16, gray=True).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        with torch.no_grad():
            pred, _, _ = model(src_tensor, tgt_light, skip_count=0)

        # save predicted image
        ckpt_name = ckpt.replace(".pth", "")
        pred_out_path = os.path.join(output_dir, f"pred_{ckpt_name}.png")
        utils.save_image(pred, pred_out_path)
        print(f"✅ Saved result: {pred_out_path}")

        results.append((ckpt_name, pred.cpu().squeeze().numpy()))

    # ===============================
    # 4. Visualization
    # ===============================
    cols = min(6, len(results) + 2)  # 每行最多6张
    rows = int(np.ceil((len(results) + 2) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    src_gray = np.array(src_img.convert("L"))
    tgt_gray = np.array(tgt_img.convert("L"))

    axes[0].imshow(src_gray, cmap='gray')
    axes[0].set_title("Source")
    axes[1].imshow(tgt_gray, cmap='gray')
    axes[1].set_title("Target")

    for i, (name, img_data) in enumerate(results):
        ax = axes[i + 2]
        if img_data.ndim == 2:
            ax.imshow(img_data, cmap='gray')
        elif img_data.ndim == 3 and img_data.shape[0] == 1:
            ax.imshow(img_data.squeeze(0), cmap='gray')
        elif img_data.ndim == 3 and img_data.shape[0] in [3, 4]:
            ax.imshow(np.transpose(img_data, (1, 2, 0)))
        ax.set_title(name.replace("checkpoint_", ""))
        ax.axis("off")

    for j in range(len(results) + 2, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "checkpoints_comparison.png"))
    plt.show()

    print(f"🎨 Comparison figure saved -> {output_dir}/checkpoints_comparison.png")



if __name__ == "__main__":
    test_model_multiple_epoch(model_dir = "trained_model/20251021_0041_L1GradientFeatureGAN_skip")
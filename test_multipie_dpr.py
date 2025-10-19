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

# ===============================
# 1. Configuration
# ===============================
data_dir = "data/Multi_Pie/pairs"

# 所有模型文件夹（可以根据你截图里路径直接复制）
model_dirs = [
    "trained_model_20251018_2351_L1Gradient_skip",
    "trained_model_20251018_2353_L1_skip",
    "trained_model_20251018_2354_L1GradientFeature_skip",
    "trained_model_20251018_2357_L1GradientFeatureGAN_skip"
]

# 对比输出路径
output_dir = "comparison_outputs"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ===============================
# 2. Load one test pair
# ===============================
with open(os.path.join(data_dir, "pairs_mapping.json"), 'r') as f:
    pairs = json.load(f)

test_idx = 0  # 改这个可以选不同测试样本
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
# 3. Compare all models
# ===============================
results = []

for model_dir in model_dirs:
    # 找到该目录下的模型文件
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    if not model_files:
        print(f"⚠️ No .pth file found in {model_dir}")
        continue

    # 默认加载最后一个（训练时间最新的）
    model_path = os.path.join(model_dir, sorted(model_files)[-1])
    print(f"📦 Loading model: {model_path}")

    # 加载模型
    model = HourglassNet(baseFilter=16, gray=True).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 推理
    with torch.no_grad():
        pred, _ = model(src_tensor, tgt_light, skip_count=0)

    # 保存预测图像
    model_name = os.path.basename(model_dir)
    pred_out_path = os.path.join(output_dir, f"pred_{model_name}.png")
    utils.save_image(pred, pred_out_path)
    print(f"✅ Saved result: {pred_out_path}")

    results.append((model_name, pred.cpu().squeeze().numpy()))

# ===============================
# 4. Optional: Show comparison figure
# ===============================
fig, axes = plt.subplots(1, len(results) + 2, figsize=(4*(len(results)+2), 4))
axes[0].imshow(src_img)
axes[0].set_title("Source Image")
axes[1].imshow(tgt_img)
axes[1].set_title("Target Image")

for i, (name, img_data) in enumerate(results):
    axes[i+2].imshow(np.transpose(img_data, (1, 2, 0)), cmap='gray')
    axes[i+2].set_title(name.replace("trained_model_", ""))

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "model_comparison.png"))
plt.show()

print("🎨 Comparison figure saved -> model_comparison.png")

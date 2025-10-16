# test_multipie_dpr.py
import os
import torch
from torchvision import transforms, utils
from PIL import Image
import numpy as np
from model.defineHourglass_512_gray_skip import HourglassNet

# ===============================
# 1. Configuration
# ===============================
data_dir = "data/Multi_Pie/pairs"
model_path = "trained_model/trained_multipie_epoch20.pth"
output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ===============================
# 2. Load Model
# ===============================
model = HourglassNet(baseFilter=16, gray=True).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"✅ Loaded model from {model_path}")

# ===============================
# 3. Load one test pair
# ===============================
import json
with open(os.path.join(data_dir, "pairs_mapping.json"), 'r') as f:
    pairs = json.load(f)

# pick first test pair (you can change the index)
test_idx = 0
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

# load lighting vectors
src_light = torch.tensor(np.load(os.path.join(data_dir, "source_light.npy"))[test_idx], dtype=torch.float32).view(1, 9, 1, 1).to(device)
tgt_light = torch.tensor(np.load(os.path.join(data_dir, "target_light.npy"))[test_idx], dtype=torch.float32).view(1, 9, 1, 1).to(device)

# ===============================
# 4. Inference
# ===============================
with torch.no_grad():
    pred, _ = model(src_tensor, tgt_light, skip_count=0)

# ===============================
# 5. Save output
# ===============================
src_out = os.path.join(output_dir, "src.png")
tgt_out = os.path.join(output_dir, "target.png")
pred_out = os.path.join(output_dir, "pred.png")

utils.save_image(src_tensor, src_out)
utils.save_image(tgt_tensor, tgt_out)
utils.save_image(pred, pred_out)

print("✅ Saved results:")
print(f"   Source: {src_out}")
print(f"   Target: {tgt_out}")
print(f"   Predicted Relighted: {pred_out}")

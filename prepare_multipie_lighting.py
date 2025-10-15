import os, json, numpy as np
from math import cos, sin, radians
from utils.utils_SH import SH_basis

# === 1️⃣ 读取灯位角度表 ===
meta_path = 'data/Multi_Pie/metadata/lighting_map.json'
with open(meta_path, 'r') as f:
    lighting_map = json.load(f)

# === 2️⃣ 角度 → 方向向量 ===
def angle_to_dir(azimuth, elevation):
    az, el = radians(azimuth), radians(elevation)
    x = cos(el) * cos(az)
    y = cos(el) * sin(az)
    z = sin(el)
    return np.array([x, y, z])

# === 3️⃣ 生成球谐参数 ===
all_sh = {}
for lid, (az, el) in lighting_map.items():
    light_dir = angle_to_dir(az, el)
    normal = np.tile(light_dir, (1, 1))        # 只取一个方向
    sh_basis = SH_basis(normal)                # 9维
    all_sh[lid] = sh_basis[0].tolist()

# === 4️⃣ 保存结果 ===
save_path = 'data/Multi_Pie/metadata/multipie_light_SH.npy'
np.save(save_path, all_sh)
print(f"✅ Saved {len(all_sh)} SH lighting vectors to {save_path}")

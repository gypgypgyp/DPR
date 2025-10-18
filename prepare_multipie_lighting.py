# prepare_multipie_lighting.py
# Name: Yunpei Gu (Team: Yunpei Gu)
# Class: CS 7180 Advanced Perception
# Date: 2025-10-15

'''
Computes and saves per-image lighting features for relighting training.
'''

import os, json, numpy as np
from math import cos, sin, radians
from utils.utils_SH import SH_basis

# === load the lighting map ===
meta_path = 'data/Multi_Pie/metadata/lighting_map.json'
with open(meta_path, 'r') as f:
    lighting_map = json.load(f)

# === angular to direction vector ===
def angle_to_dir(azimuth, elevation):
    az, el = radians(azimuth), radians(elevation)
    x = cos(el) * cos(az)
    y = cos(el) * sin(az)
    z = sin(el)
    return np.array([x, y, z])

# === create SH parameter ===
all_sh = {}
for lid, (az, el) in lighting_map.items():
    light_dir = angle_to_dir(az, el)
    normal = np.tile(light_dir, (1, 1))        # 只取一个方向
    sh_basis = SH_basis(normal)                # 9维
    all_sh[lid] = sh_basis[0].tolist()

# === save results ===
save_path = 'data/Multi_Pie/metadata/multipie_light_SH.npy'
np.save(save_path, all_sh)
print(f"✅ Saved {len(all_sh)} SH lighting vectors to {save_path}")

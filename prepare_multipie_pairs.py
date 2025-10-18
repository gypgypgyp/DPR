# prepare_multipie_pairs.py
# Name: Yunpei Gu (Team: Yunpei Gu)
# Class: CS 7180 Advanced Perception
# Date: 2025-10-15

'''
Creates paired data (source → target lighting) and mapping metadata for training.
'''

import os
import random
import json
import numpy as np
from glob import glob

# input path
data_dir = "data/Multi_Pie/HR_128"
metadata_dir = "data/Multi_Pie/metadata"
output_dir = "data/Multi_Pie/pairs"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "source"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "target"), exist_ok=True)

# load SH 
# light_SH = np.load(os.path.join(metadata_dir, "multipie_light_SH.npy"))
light_SH = np.load(os.path.join(metadata_dir, "multipie_light_SH.npy"), allow_pickle=True)

# filter out 00/19 (non-lighting) pics
image_files = sorted([f for f in glob(os.path.join(data_dir, "*.png")) 
                      if not f.endswith("_00_crop_128.png") and not f.endswith("_19_crop_128.png")])

# group pics (same person_id) 
grouped = {}
for path in image_files:
    fname = os.path.basename(path)
    person_id = "_".join(fname.split("_")[:4])
    grouped.setdefault(person_id, []).append(path)

pairs_info = []
source_light_list, target_light_list = [], []

# enumerate every person_id; Randomly get train pairs
for person_id, imgs in grouped.items():
    if len(imgs) < 2:
        continue

    for _ in range(3):  # generate 3 paris for every person
        src, tgt = random.sample(imgs, 2)
        src_name = os.path.basename(src)
        tgt_name = os.path.basename(tgt)

        # No. lighting
        # src_light_id = int(src_name.split("_")[-2])
        # tgt_light_id = int(tgt_name.split("_")[-2])
        src_light_id = int(src_name.split("_")[4])
        tgt_light_id = int(tgt_name.split("_")[4])


        # copy paste pictures
        os.system(f"cp '{src}' '{os.path.join(output_dir, 'source', src_name)}'")
        os.system(f"cp '{tgt}' '{os.path.join(output_dir, 'target', tgt_name)}'")

        # save lighting para
        # source_light_list.append(light_SH[src_light_id - 1])  # SH编号01-18 -> index 0-17
        # target_light_list.append(light_SH[tgt_light_id - 1])
        
        light_dict = light_SH.item()  

        source_light_list.append(np.array(light_dict[str(src_light_id).zfill(2)]))
        target_light_list.append(np.array(light_dict[str(tgt_light_id).zfill(2)]))


        # add the pairs information
        pairs_info.append({
            "source": src_name,
            "target": tgt_name,
            "source_light_id": src_light_id,
            "target_light_id": tgt_light_id
        })

# convert to numpy and save
np.save(os.path.join(output_dir, "source_light.npy"), np.array(source_light_list))
np.save(os.path.join(output_dir, "target_light.npy"), np.array(target_light_list))

# save JSON map 
with open(os.path.join(output_dir, "pairs_mapping.json"), "w") as f:
    json.dump(pairs_info, f, indent=4)

print(f"✅ successfully {len(pairs_info)} training pairs")
print(f"   output path: {output_dir}")
print(f"   mapping: pairs_mapping.json saved")


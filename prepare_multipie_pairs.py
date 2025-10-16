import os
import random
import json
import numpy as np
from glob import glob

# 输入路径
data_dir = "data/Multi_Pie/HR_128"
metadata_dir = "data/Multi_Pie/metadata"
output_dir = "data/Multi_Pie/pairs"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "source"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "target"), exist_ok=True)

# 加载光照 SH 系数
# light_SH = np.load(os.path.join(metadata_dir, "multipie_light_SH.npy"))
light_SH = np.load(os.path.join(metadata_dir, "multipie_light_SH.npy"), allow_pickle=True)

# 过滤掉 00/19（无闪光）的图片
image_files = sorted([f for f in glob(os.path.join(data_dir, "*.png")) 
                      if not f.endswith("_00_crop_128.png") and not f.endswith("_19_crop_128.png")])

# 分组：同一个人（person_id）下的所有图片
grouped = {}
for path in image_files:
    fname = os.path.basename(path)
    person_id = "_".join(fname.split("_")[:4])
    grouped.setdefault(person_id, []).append(path)

pairs_info = []
source_light_list, target_light_list = [], []

# 遍历每个人，随机生成训练样本对
for person_id, imgs in grouped.items():
    if len(imgs) < 2:
        continue

    for _ in range(3):  # 每人生成3对样本
        src, tgt = random.sample(imgs, 2)
        src_name = os.path.basename(src)
        tgt_name = os.path.basename(tgt)

        # 光照编号（倒数第二个字段）
        # src_light_id = int(src_name.split("_")[-2])
        # tgt_light_id = int(tgt_name.split("_")[-2])
        src_light_id = int(src_name.split("_")[4])
        tgt_light_id = int(tgt_name.split("_")[4])


        # 保存图片副本
        os.system(f"cp '{src}' '{os.path.join(output_dir, 'source', src_name)}'")
        os.system(f"cp '{tgt}' '{os.path.join(output_dir, 'target', tgt_name)}'")

        # 保存光照参数
        # source_light_list.append(light_SH[src_light_id - 1])  # SH编号01-18 -> index 0-17
        # target_light_list.append(light_SH[tgt_light_id - 1])
        
        light_dict = light_SH.item()  # 取出内部真正的 dict

        source_light_list.append(np.array(light_dict[str(src_light_id).zfill(2)]))
        target_light_list.append(np.array(light_dict[str(tgt_light_id).zfill(2)]))


        # 添加映射关系
        pairs_info.append({
            "source": src_name,
            "target": tgt_name,
            "source_light_id": src_light_id,
            "target_light_id": tgt_light_id
        })

# 转为 numpy 并保存
np.save(os.path.join(output_dir, "source_light.npy"), np.array(source_light_list))
np.save(os.path.join(output_dir, "target_light.npy"), np.array(target_light_list))

# 保存 JSON 映射
with open(os.path.join(output_dir, "pairs_mapping.json"), "w") as f:
    json.dump(pairs_info, f, indent=4)

print(f"✅ 成功生成 {len(pairs_info)} 个训练样本对")
print(f"   输出路径: {output_dir}")
print(f"   映射文件: pairs_mapping.json 已保存")


# import os, re, random, shutil, numpy as np

# # === 路径配置 ===
# root = "data/Multi_Pie/HR_128"                 # 原始 Multi-PIE 图片目录
# pair_root = "data/Multi_Pie/pairs"             # 输出目录
# meta_sh_path = "data/Multi_Pie/metadata/multipie_light_SH.npy"

# # === 读取光照映射表 ===
# if not os.path.exists(meta_sh_path):
#     raise FileNotFoundError("请先运行 prepare_multipie_lighting.py 生成 multipie_light_SH.npy")

# meta_sh = np.load(meta_sh_path, allow_pickle=True).item()

# os.makedirs(os.path.join(pair_root, "source"), exist_ok=True)
# os.makedirs(os.path.join(pair_root, "target"), exist_ok=True)

# # === 文件名解析： 001_01_01_010_05_crop_128.png ===
# # 提取 person_id 和 illumination_id
# pattern = re.compile(r"(\d+)_\d+_\d+_\d+_(\d+)_crop_128\.png")

# person_to_imgs = {}
# for f in os.listdir(root):
#     m = pattern.match(f)
#     if not m:
#         continue
#     pid, light_id = m.groups()
#     light_id = light_id.zfill(2)   # 补齐两位数 (e.g. 1 -> 01)

#     # ⚠️ 跳过无光照图像（00, 19）
#     if light_id in ["00", "19"]:
#         continue

#     # 跳过未知光照编号
#     if light_id not in meta_sh:
#         continue

#     person_to_imgs.setdefault(pid, []).append((f, light_id))

# src_light_list, tgt_light_list = [], []
# pair_count = 0

# # === 生成 source-target 训练对 ===
# for pid, imgs in person_to_imgs.items():
#     if len(imgs) < 2:
#         continue

#     # 每个ID生成若干随机光照配对
#     for _ in range(3):  # 每人3对
#         src, tgt = random.sample(imgs, 2)
#         src_img, src_lid = src
#         tgt_img, tgt_lid = tgt

#         shutil.copy(os.path.join(root, src_img), os.path.join(pair_root, "source", src_img))
#         shutil.copy(os.path.join(root, tgt_img), os.path.join(pair_root, "target", tgt_img))

#         src_light_list.append(meta_sh[src_lid])
#         tgt_light_list.append(meta_sh[tgt_lid])
#         pair_count += 1

# # === 保存光照参数 ===
# np.save(os.path.join(pair_root, "source_light.npy"), np.array(src_light_list))
# np.save(os.path.join(pair_root, "target_light.npy"), np.array(tgt_light_list))

# print(f"✅ generated {pair_count} training pairs")
# print(f"   output path: {pair_root}")
# print(f"   skip the pic without lighting (_00, _19)")

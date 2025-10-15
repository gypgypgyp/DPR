import os, re, random, shutil, numpy as np

# === 路径配置 ===
root = "data/Multi_Pie/HR_128"                 # 原始 Multi-PIE 图片目录
pair_root = "data/Multi_Pie/pairs"             # 输出目录
meta_sh_path = "data/Multi_Pie/metadata/multipie_light_SH.npy"

# === 读取光照映射表 ===
if not os.path.exists(meta_sh_path):
    raise FileNotFoundError("请先运行 prepare_multipie_lighting.py 生成 multipie_light_SH.npy")

meta_sh = np.load(meta_sh_path, allow_pickle=True).item()

os.makedirs(os.path.join(pair_root, "source"), exist_ok=True)
os.makedirs(os.path.join(pair_root, "target"), exist_ok=True)

# === 文件名解析： 001_01_01_010_05_crop_128.png ===
# 提取 person_id 和 illumination_id
pattern = re.compile(r"(\d+)_\d+_\d+_\d+_(\d+)_crop_128\.png")

person_to_imgs = {}
for f in os.listdir(root):
    m = pattern.match(f)
    if not m:
        continue
    pid, light_id = m.groups()
    light_id = light_id.zfill(2)   # 补齐两位数 (e.g. 1 -> 01)

    # ⚠️ 跳过无光照图像（00, 19）
    if light_id in ["00", "19"]:
        continue

    # 跳过未知光照编号
    if light_id not in meta_sh:
        continue

    person_to_imgs.setdefault(pid, []).append((f, light_id))

src_light_list, tgt_light_list = [], []
pair_count = 0

# === 生成 source-target 训练对 ===
for pid, imgs in person_to_imgs.items():
    if len(imgs) < 2:
        continue

    # 每个ID生成若干随机光照配对
    for _ in range(3):  # 每人3对
        src, tgt = random.sample(imgs, 2)
        src_img, src_lid = src
        tgt_img, tgt_lid = tgt

        shutil.copy(os.path.join(root, src_img), os.path.join(pair_root, "source", src_img))
        shutil.copy(os.path.join(root, tgt_img), os.path.join(pair_root, "target", tgt_img))

        src_light_list.append(meta_sh[src_lid])
        tgt_light_list.append(meta_sh[tgt_lid])
        pair_count += 1

# === 保存光照参数 ===
np.save(os.path.join(pair_root, "source_light.npy"), np.array(src_light_list))
np.save(os.path.join(pair_root, "target_light.npy"), np.array(tgt_light_list))

print(f"✅ generated {pair_count} training pairs")
print(f"   output path: {pair_root}")
print(f"   skip the pic without lighting (_00, _19)")

# DPR
Deep Single-Image Portrait Relighting


load training data set to /data/Multi_Pie/HR_128

```bash                 
python prepare_multipie_lighting.py
```

✅ Saved 18 SH lighting vectors to data/Multi_Pie/metadata/multipie_light_SH.npy

```bash  
python prepare_multipie_pairs.py
```
✅ 成功生成 6 个训练样本对
   输出路径: data/Multi_Pie/pairs
   跳过无闪光图像 (_00, _19)

```bash  
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
numpy==1.26.4 opencv-python==4.10.0.84 pillow==10.3.0 matplotlib==3.9.1 \
pyshtools==4.12.1 tqdm==4.66.4 tensorboard==2.17.0
```

```bash  
python check_data_prepare.py
```
```bash  
python train_multipie_dpr.py
```
```bash  
python test_multipie_dpr.py
```

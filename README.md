# DPR
Deep Single-Image Portrait Relighting

Name: Yunpei Gu (Team: Yunpei Gu)
Class: CS 7180 Advanced Perception
Date: 2025-10-15
Time Travel Days: I would like to use 4 travel days.

## Operating System
Rocky Linux 9.3 (x86_64) — NEU Explorer HPC Cluster

## Dependencies
Install the following dependencies before running:
```bash  
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
numpy==1.26.4 opencv-python==4.10.0.84 pillow==10.3.0 matplotlib==3.9.1 \
pyshtools==4.12.1 tqdm==4.66.4 tensorboard==2.17.0
```

## Compiling and Executing

### Data preparation

Download Multi-PIE dataset from Kaggle:
https://www.kaggle.com/datasets/aliates/multi-pie/data

Extract it under:
```bash 
data/Multi_Pie/HR_128/
```

Generate Lighting Metadata:
```bash                 
python prepare_multipie_lighting.py
```

you'll see:
```bash 
✅ Saved 18 SH lighting vectors to data/Multi_Pie/metadata/multipie_light_SH.npy
```

this will create folder containing multipie_light_SH.npy: 
```bash
./data/Multi_Pie/metadata
```

Generate Paired Training Samples:
```bash  
python prepare_multipie_pairs.py
```

Expected output:
```bash 
print(f"✅ successfully {len(pairs_info)} training pairs")
print(f"   output path: {output_dir}")
print(f"   mapping: pairs_mapping.json saved")
```

(If you modify your dataset, please delete data/Multi_Pie/pairs/ and re-run this script.)

Verify Data
```bash  
python check_data_prepare.py
```
This script will check if your paired dataset and lighting files are properly aligned.

### Training the Model
if you are using HPC, run:
```bash  
sbatch train_job.sh
```

If not, run:
```bash  
python train_multipie_dpr.py
```

The trained models and logs will be saved to:
```bash 
./trained_model
```

## Testing / Inference
To test the trained model:
```bash  
python test_multipie_dpr.py
```
Expected outputs:
Relit portrait images under various lighting conditions
Output folder: ./output/relit_results/
Visualization of side-by-side source → relit → target comparisons

### ✅ Summary of Included Files

| File | Description |
|------|--------------|
| `train_multipie_dpr.py` | Main training script with **L1**, **gradient**, **feature**, and **GAN loss** |
| `defineHourglass_512_gray_skip.py` | Hourglass relighting model with **skip training** |
| `prepare_multipie_lighting.py` | Extracts **spherical harmonics (SH)** lighting coefficients |
| `prepare_multipie_pairs.py` | Generates **paired training data** for supervised relighting |
| `check_data_prepare.py` | Validates dataset structure and sample consistency |
| `test_multipie_dpr.py` | Runs inference and saves relit image outputs |
| `README.md` | Project documentation (this file) |

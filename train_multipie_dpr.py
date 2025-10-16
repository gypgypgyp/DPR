import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from model.defineHourglass_512_gray_skip import HourglassNet
import datetime

# ===============================
# 1. Dataset Definition
# ===============================
import json

class MultiPiePairDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.src_dir = os.path.join(data_dir, "source")
        self.tgt_dir = os.path.join(data_dir, "target")
        self.src_light = np.load(os.path.join(data_dir, "source_light.npy"))
        self.tgt_light = np.load(os.path.join(data_dir, "target_light.npy"))

        mapping_file = os.path.join(data_dir, "pairs_mapping.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                self.mapping = json.load(f)
        else:
            # fallback: assume identical filenames
            self.mapping = [{"source": f, "target": f} for f in sorted(os.listdir(self.src_dir))]

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        pair = self.mapping[idx]
        src_img_path = os.path.join(self.src_dir, pair["source"])
        tgt_img_path = os.path.join(self.tgt_dir, pair["target"])

        src_img = Image.open(src_img_path).convert("RGB")
        tgt_img = Image.open(tgt_img_path).convert("RGB")

        src_tensor = self.transform(src_img)
        tgt_tensor = self.transform(tgt_img)

        # src_light = torch.tensor(self.src_light[idx], dtype=torch.float32).view(1, 9, 1, 1)
        # tgt_light = torch.tensor(self.tgt_light[idx], dtype=torch.float32).view(1, 9, 1, 1)
        src_light = torch.tensor(self.src_light[idx], dtype=torch.float32).view(9, 1, 1)
        tgt_light = torch.tensor(self.tgt_light[idx], dtype=torch.float32).view(9, 1, 1)


        return src_tensor, tgt_tensor, src_light, tgt_light



# ===============================
# 2. Train Function
# ===============================
def train_dpr(data_dir, epochs=20, batch_size=2, lr=1e-4, save_dir="trained_model"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    torch.autograd.set_detect_anomaly(False)  # 可改为 True 调试
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Dataset and Loader
    print("✅ Step 1: Creating dataset...", flush=True)
    dataset = MultiPiePairDataset(data_dir)
    print(f"✅ Step 2: Dataset loaded, total {len(dataset)} samples", flush=True)
    print("✅ Step 3: Creating DataLoader...", flush=True)

    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print("✅ Step 4: DataLoader ready!", flush=True)

    # Model
    model = HourglassNet(baseFilter=16, gray=True).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(save_dir, exist_ok=True)


    # print("✅ Step 5: Testing one batch...", flush=True)
    # for i, (src, tgt, src_light, tgt_light) in enumerate(dataloader):
    #     print(f"✅ Loaded batch {i}, src: {src.shape}, tgt: {tgt.shape}", flush=True)
    #     if i == 0:
    #         break
    # print("✅ Step 6: Passed sanity check, starting training...", flush=True)

    src, tgt, src_light, tgt_light = next(iter(dataloader))
    print(f"✅ Loaded one batch: src={src.shape}, tgt={tgt.shape}", flush=True)
    with torch.no_grad():
        pred, _ = model(src.to(device), tgt_light.to(device), skip_count=0)
        print(f"✅ Forward check ok: pred={pred.shape}", flush=True)
    print("✅ Step 6: Passed sanity check, starting training...\n", flush=True)



    for epoch in range(epochs):

        model.train()
        total_loss = 0.0

        # for src, tgt, src_light, tgt_light in dataloader:
        for i, (src, tgt, src_light, tgt_light) in enumerate(dataloader):

            src, tgt = src.to(device), tgt.to(device)
            src_light, tgt_light = src_light.to(device), tgt_light.to(device)

            optimizer.zero_grad()

            
            # print("➡️ Forwarding...", flush=True)
            pred, _ = model(src, tgt_light, skip_count=0)
            # print("✅ Forward done", flush=True)
            loss = criterion(pred, tgt)

            # torch.autograd.set_detect_anomaly(True) #debug

            # print("➡️ Backwarding...", flush=True)
            loss.backward()
            # print("✅ Backward done", flush=True)
            optimizer.step()
            # print("✅ Step done", flush=True)

            total_loss += loss.item() * src.size(0)

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{i+1}/{len(dataloader)}] | Loss: {loss.item():.6f}", flush=True)


        avg_loss = total_loss / len(dataset)
        print(f"📉 Epoch [{epoch+1}/{epochs}] | Avg Loss: {avg_loss:.6f}", flush=True)
        # print(f"Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"trained_multipie_epoch{epoch+1}_{timestamp}.pth")
            # save_path = os.path.join(save_dir, f"trained_multipie_epoch{epoch+1}.pth")
            # save_path = os.path.join(save_dir, f"trained_multipie_epoch{epoch+1}.pth", flush=True)
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved to {save_path}\n", flush=True)

    print("🎉 Training finished!")


# ===============================
# 3. Main
# ===============================
if __name__ == "__main__":
    data_dir = "data/Multi_Pie/pairs"
    train_dpr(data_dir=data_dir, epochs=20, batch_size=2, lr=1e-4, save_dir="trained_model")

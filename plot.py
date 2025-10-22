import numpy as np
import matplotlib.pyplot as plt
import os 

save_dir = "trained_model/20251021_0039_L1_skip"
save_dir = "trained_model/20251021_1454_L1Grad_skip"
# save_dir = "trained_model/20251021_0040_L1GradFeat_skip"
# save_dir = "trained_model/20251021_0041_L1GradientFeatureGAN_skip"



loss_history = np.load(os.path.join(save_dir, "train_loss.npy"))
# val_si_mse_list = np.load(os.path.join(save_dir, "val_si_mse.npy"))
# val_si_l2_list = np.load(os.path.join(save_dir, "val_si_l2.npy"))  



num_epochs_to_print = 100
for epoch, loss in enumerate(loss_history[:num_epochs_to_print], start=1):
    print(f"Epoch {epoch:02d}: Loss = {loss:.6f}")


# # 读取数据
# loss_history = np.load(os.path.join(save_dir, "train_loss.npy"))
# val_si_mse_list = np.load(os.path.join(save_dir, "val_si_mse.npy"))
# val_si_l2_list  = np.load(os.path.join(save_dir, "val_si_l2.npy"))

# # 只画前20个epoch
# num_epochs_to_plot = min(20, len(loss_history))
# epochs_range = range(1, num_epochs_to_plot + 1)

# loss_subset = loss_history[:num_epochs_to_plot]
# val_si_mse_subset = val_si_mse_list[:num_epochs_to_plot]
# val_si_l2_subset  = val_si_l2_list[:num_epochs_to_plot]

# # -----------------------------
# # 1. Train Loss (前20个epoch)
# # -----------------------------
# plt.figure(figsize=(6, 4))
# plt.plot(epochs_range, loss_subset, marker='o', label="Train Loss", color='tab:blue')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title(f"Training Loss Curve (First {num_epochs_to_plot} Epochs)")
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, f"train_loss_curve_first{num_epochs_to_plot}.png"))
# plt.close()
# print(f"📊 Saved first {num_epochs_to_plot} epoch training loss curve.")

# # -----------------------------
# # 2. Validation SI-MSE (前20个epoch)
# # -----------------------------
# plt.figure(figsize=(6, 4))
# plt.plot(epochs_range, val_si_mse_subset, marker='s', color='tab:orange', linewidth=1.8)
# plt.xlabel("Epoch")
# plt.ylabel("SI-MSE")
# plt.title(f"Validation SI-MSE (First {num_epochs_to_plot} Epochs)")
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, f"val_si_mse_first{num_epochs_to_plot}.png"))
# plt.close()
# print(f"📊 Saved validation SI-MSE curve (first {num_epochs_to_plot} epochs).")

# # -----------------------------
# # 3. Validation SI-L2 (前20个epoch)
# # -----------------------------
# plt.figure(figsize=(6, 4))
# plt.plot(epochs_range, val_si_l2_subset, marker='^', color='tab:green', linewidth=1.8)
# plt.xlabel("Epoch")
# plt.ylabel("SI-L2")
# plt.title(f"Validation SI-L2 (First {num_epochs_to_plot} Epochs)")
# plt.grid(True, linestyle="--", alpha=0.6)
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, f"val_si_l2_first{num_epochs_to_plot}.png"))
# plt.close()
# print(f"📊 Saved validation SI-L2 curve (first {num_epochs_to_plot} epochs).")
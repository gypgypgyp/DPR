import numpy as np
"""
check the results of prepare_multiplie
"""
src_light = np.load("data/Multi_Pie/pairs/source_light.npy")
tgt_light = np.load("data/Multi_Pie/pairs/target_light.npy")

print("Source lighting shape:", src_light.shape)
print("Target lighting shape:", tgt_light.shape)

print("Source lighting sample:\n", src_light[:2])
print("Target lighting sample:\n", tgt_light[:2])

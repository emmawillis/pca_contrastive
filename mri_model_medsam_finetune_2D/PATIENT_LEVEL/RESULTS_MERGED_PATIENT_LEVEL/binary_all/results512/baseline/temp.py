#!/usr/bin/env python3
# inspect_head_4.py
import torch

CKPT = "/home/ewillis/projects/aip-medilab/ewillis/pca_contrastive/mri_model_medsam_finetune_2D/PATIENT_LEVEL/RESULTS_MERGED_PATIENT_LEVEL/binary_all/results512/baseline/ckpt_best.pt"  # <-- EDIT THIS

obj = torch.load(CKPT, map_location="cpu")
sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
if not isinstance(sd, dict) or not all(hasattr(v, "shape") for v in sd.values()):
    for v in (obj.values() if isinstance(obj, dict) else []):
        if isinstance(v, dict) and all(hasattr(x, "shape") for x in v.values()):
            sd = v; break

w_key = next((k for k in sd.keys() if k.endswith("head.4.weight")), None)
b_key = w_key.replace("weight", "bias") if w_key else None

print("checkpoint:", CKPT)
print("weight key:", w_key)
print("bias key:  ", b_key if (b_key and b_key in sd) else None)

if w_key:
    w_shape = tuple(sd[w_key].shape)
    print("weight shape:", w_shape, "(classes, features_in)")
    if b_key and b_key in sd:
        print("bias shape:  ", tuple(sd[b_key].shape))
    print("→ inferred classes:", w_shape[0], "| features_in:", w_shape[1])
else:
    print("Could not find 'head.4.weight'."
          "\nHere are some head-ish keys:", [k for k in sd if k.endswith(".weight") and ".head." in k or k.startswith("head")][:10])

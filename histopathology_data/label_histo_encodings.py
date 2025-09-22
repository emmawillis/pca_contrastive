import os
import pandas as pd

# === Paths ===
h5_dir = "/Users/emma/Desktop/QUEENS/THESIS/contrastive/datasets/UNI2_panda_encodings_by_patch"
train_csv = "/Users/emma/Desktop/QUEENS/THESIS/contrastive/train.csv"

# === Load CSV ===
df = pd.read_csv(train_csv)
df = df.set_index("image_id")  # so we can look up by image_id fast

matched, missing = 0, 0

# === Loop over all h5 files ===
for fname in os.listdir(h5_dir):
    if not fname.endswith(".h5"):
        continue

    image_id = os.path.splitext(fname)[0]  # strip ".h5"

    if image_id in df.index:
        isup = df.loc[image_id, "isup_grade"]
        new_name = f"{image_id}_{isup}.h5"
        old_path = os.path.join(h5_dir, fname)
        new_path = os.path.join(h5_dir, new_name)

        os.rename(old_path, new_path)
        matched += 1
    else:
        missing += 1
        print(f"Missing in CSV: {image_id}")

print(f"\nMatched: {matched}")
print(f"Missing: {missing}")

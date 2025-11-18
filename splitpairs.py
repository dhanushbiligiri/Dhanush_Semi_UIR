# save this as split_pairs.py and run: python split_pairs.py
import os, shutil

raw_dir = "data/raw-890"
ref_dir = "data/reference-890"

labeled_input_dir = "data/labeled1/input"
labeled_gt_dir    = "data/labeled1/GT"
val_input_dir     = "data/val/input"
val_gt_dir        = "data/val/GT"

os.makedirs(labeled_input_dir, exist_ok=True)
os.makedirs(labeled_gt_dir, exist_ok=True)
os.makedirs(val_input_dir, exist_ok=True)
os.makedirs(val_gt_dir, exist_ok=True)

raw_files = sorted([f for f in os.listdir(raw_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])

# adjust this % if you want a different split
num_train = int(0.9 * len(raw_files))  # 90% train, 10% val

for i, fname in enumerate(raw_files):
    raw_path = os.path.join(raw_dir, fname)
    ref_path = os.path.join(ref_dir, fname)  # assumes same filenames

    if not os.path.exists(ref_path):
        print("No matching reference for:", fname)
        continue

    if i < num_train:
        shutil.copy2(raw_path, labeled_input_dir)
        shutil.copy2(ref_path, labeled_gt_dir)
    else:
        shutil.copy2(raw_path, val_input_dir)
        shutil.copy2(ref_path, val_gt_dir)

print("Done splitting into labeled1 (train) and val.")

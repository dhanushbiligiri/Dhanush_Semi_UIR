import os
import random
import shutil

random.seed(2022)

raw_dir = "data/raw-890/all"
gt_dir = "data/reference-890"

labeled1_input = "data/labeled1/input"
labeled1_gt    = "data/labeled1/GT"
val_input      = "data/val/input"
val_gt         = "data/val/GT"

os.makedirs(labeled1_input, exist_ok=True)
os.makedirs(labeled1_gt, exist_ok=True)
os.makedirs(val_input, exist_ok=True)
os.makedirs(val_gt, exist_ok=True)

# filenames that exist in BOTH raw and reference
gt_files = [f for f in os.listdir(gt_dir)
            if os.path.isfile(os.path.join(gt_dir, f))]

paired_files = [f for f in gt_files
                if os.path.isfile(os.path.join(raw_dir, f))]

print(f"Found {len(paired_files)} paired images.")

# shuffle and split (90% train, 10% val)
random.shuffle(paired_files)
n_total = len(paired_files)
n_val = int(0.1 * n_total)
val_files = paired_files[:n_val]
train_files = paired_files[n_val:]

print(f"Train: {len(train_files)}, Val: {len(val_files)}")

def copy_pairs(file_list, raw_target, gt_target):
    for fname in file_list:
        src_raw = os.path.join(raw_dir, fname)
        src_gt  = os.path.join(gt_dir, fname)
        dst_raw = os.path.join(raw_target, fname)
        dst_gt  = os.path.join(gt_target, fname)
        shutil.copy2(src_raw, dst_raw)
        shutil.copy2(src_gt, dst_gt)

print("Copying training pairs...")
copy_pairs(train_files, labeled1_input, labeled1_gt)

print("Copying validation pairs...")
copy_pairs(val_files, val_input, val_gt)

print("Done.")

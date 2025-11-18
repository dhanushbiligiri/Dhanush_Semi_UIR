import re
import csv

LOG_FILE = "Final Project/Dhanush_Semi_UIR/logs.txt"
CSV_FILE = "Final Project/Dhanush_Semi_UIR/training_metrics.csv"

# Regex patterns for extracting values
train_pattern = re.compile(
    r"Train-Student Epoch (\d+)\s*\|\s*Ls ([\d.]+)\s*Lu ([\d.]+)"
)


eval_pattern = re.compile(
    r"Eval-Student Epoch (\d+)\s*\|\s*PSNR:\s*([\d.]+), SSIM:\s*([\d.]+)"
)

final_pattern = re.compile(
    r"\[(\d+)\]\s*main_loss:\s*([\d.]+),\s*train psnr:\s*([\d.]+),\s*val psnr:\s*([\d.]+),\s*lr:\s*([\d.]+)"
)

# Storage
records = {}

with open(LOG_FILE, "r") as f:
    for line in f:
        # --- TRAIN LINE ---
        m = train_pattern.search(line)
        if m:
            epoch = int(m.group(1))
            Ls = float(m.group(2))
            Lu = float(m.group(3))
            records.setdefault(epoch, {})
            records[epoch]["epoch"] = epoch
            records[epoch]["Ls"] = Ls
            records[epoch]["Lu"] = Lu

        # --- EVAL LINE ---
        m = eval_pattern.search(line)
        if m:
            epoch = int(m.group(1))
            psnr = float(m.group(2))
            ssim = float(m.group(3))
            records.setdefault(epoch, {})
            records[epoch]["val_psnr_eval"] = psnr
            records[epoch]["val_ssim_eval"] = ssim

        # --- FINAL LINE ---
        m = final_pattern.search(line)
        if m:
            epoch = int(m.group(1))
            main_loss = float(m.group(2))
            train_psnr = float(m.group(3))
            val_psnr = float(m.group(4))
            lr = float(m.group(5))

            records.setdefault(epoch, {})
            records[epoch]["main_loss"] = main_loss
            records[epoch]["train_psnr"] = train_psnr
            records[epoch]["val_psnr"] = val_psnr
            records[epoch]["lr"] = lr

# Write CSV
headers = [
    "epoch", "Ls", "Lu",
    "main_loss", "train_psnr", "val_psnr",
    "val_psnr_eval", "val_ssim_eval", "lr"
]

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()

    for epoch in sorted(records.keys()):
        row = records[epoch]
        writer.writerow({
            "epoch": epoch,
            "Ls": row.get("Ls"),
            "Lu": row.get("Lu"),
            "main_loss": row.get("main_loss"),
            "train_psnr": row.get("train_psnr"),
            "val_psnr": row.get("val_psnr"),
            "val_psnr_eval": row.get("val_psnr_eval"),
            "val_ssim_eval": row.get("val_ssim_eval"),
            "lr": row.get("lr"),
        })

print(f"Saved CSV to: {CSV_FILE}")

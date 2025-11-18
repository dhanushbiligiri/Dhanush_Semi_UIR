import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
CSV_PATH = "Final Project/Dhanush_Semi_UIR/test_nr_iqa_scores.csv"
OUT_DIR = "Final Project/Dhanush_Semi_UIR/plots_nr_iqa"
os.makedirs(OUT_DIR, exist_ok=True)
# ---------------------------

df = pd.read_csv(CSV_PATH)

# Convenience
names = df["filename"]
idx = range(1, len(df) + 1)

# 1) MUSIQ: before vs after, and delta
plt.figure()
plt.plot(idx, df["musiq_orig"], marker="o", label="MUSIQ original")
plt.plot(idx, df["musiq_enh"], marker="o", label="MUSIQ enhanced")
plt.xlabel("Image index")
plt.ylabel("MUSIQ score (higher = better)")
plt.title("MUSIQ: original vs enhanced (test set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "musiq_orig_vs_enh.png"), dpi=300)
plt.close()

plt.figure()
plt.bar(idx, df["musiq_delta_enh_minus_orig"])
plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Image index")
plt.ylabel("Δ MUSIQ (enh - orig)")
plt.title("Per-image MUSIQ change (test set)")
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "musiq_delta_bar.png"), dpi=300)
plt.close()

# 2) NIQE: before vs after, and delta (remember lower = better)
plt.figure()
plt.plot(idx, df["niqe_orig"], marker="o", label="NIQE original")
plt.plot(idx, df["niqe_enh"], marker="o", label="NIQE enhanced")
plt.xlabel("Image index")
plt.ylabel("NIQE (lower = better)")
plt.title("NIQE: original vs enhanced (test set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "niqe_orig_vs_enh.png"), dpi=300)
plt.close()

plt.figure()
plt.bar(idx, df["niqe_delta_orig_minus_enh"])
plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Image index")
plt.ylabel("Δ NIQE (orig - enh)")
plt.title("Per-image NIQE change (positive = improved)")
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "niqe_delta_bar.png"), dpi=300)
plt.close()

# 3) BRISQUE: before vs after, and delta (lower = better)
plt.figure()
plt.plot(idx, df["brisque_orig"], marker="o", label="BRISQUE original")
plt.plot(idx, df["brisque_enh"], marker="o", label="BRISQUE enhanced")
plt.xlabel("Image index")
plt.ylabel("BRISQUE (lower = better)")
plt.title("BRISQUE: original vs enhanced (test set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "brisque_orig_vs_enh.png"), dpi=300)
plt.close()

plt.figure()
plt.bar(idx, df["brisque_delta_orig_minus_enh"])
plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Image index")
plt.ylabel("Δ BRISQUE (orig - enh)")
plt.title("Per-image BRISQUE change (positive = improved)")
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "brisque_delta_bar.png"), dpi=300)
plt.close()

# 4) Summary printout for your report
mean_musiq_orig = df["musiq_orig"].mean()
mean_musiq_enh = df["musiq_enh"].mean()
mean_musiq_delta = df["musiq_delta_enh_minus_orig"].mean()

mean_niqe_orig = df["niqe_orig"].mean()
mean_niqe_enh = df["niqe_enh"].mean()
mean_niqe_delta = df["niqe_delta_orig_minus_enh"].mean()

mean_brisq_orig = df["brisque_orig"].mean()
mean_brisq_enh = df["brisque_enh"].mean()
mean_brisq_delta = df["brisque_delta_orig_minus_enh"].mean()

print("=== NR-IQA Summary (test set) ===")
print(f"MUSIQ   mean: orig={mean_musiq_orig:.2f}, enh={mean_musiq_enh:.2f}, "
      f"Δ(enh-orig)={mean_musiq_delta:.2f}")
print(f"NIQE    mean: orig={mean_niqe_orig:.2f}, enh={mean_niqe_enh:.2f}, "
      f"Δ(orig-enh)={mean_niqe_delta:.2f} (positive = better)")
print(f"BRISQUE mean: orig={mean_brisq_orig:.2f}, enh={mean_brisq_enh:.2f}, "
      f"Δ(orig-enh)={mean_brisq_delta:.2f} (positive = better)")

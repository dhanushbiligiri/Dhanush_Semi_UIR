import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE = "Final Project/Dhanush_Semi_UIR/training_metrics.csv"
OUT_DIR = "Final Project/Dhanush_Semi_UIR/metric_figs"


def ensure_out_dir():
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)


def load_data():
    df = pd.read_csv(CSV_FILE)
    # If there are any completely empty columns / NaNs, keep but handle later
    return df


def plot_loss(df):
    plt.figure()
    plt.plot(df["epoch"], df["main_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Main Loss")
    plt.title("Training Main Loss vs Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=300)
    plt.close()


def plot_psnr(df):
    plt.figure()
    if "train_psnr" in df.columns:
        plt.plot(df["epoch"], df["train_psnr"], label="Train PSNR")
    if "val_psnr" in df.columns:
        plt.plot(df["epoch"], df["val_psnr"], label="Val PSNR (from [ ] line)")
    if "val_psnr_eval" in df.columns:
        plt.plot(df["epoch"], df["val_psnr_eval"], label="Val PSNR (Eval line)", linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "psnr_curve.png"), dpi=300)
    plt.close()


def plot_ssim(df):
    if "val_ssim_eval" not in df.columns:
        return

    plt.figure()
    plt.plot(df["epoch"], df["val_ssim_eval"])
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("Validation SSIM vs Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ssim_curve.png"), dpi=300)
    plt.close()


def plot_ls_lu(df):
    # Skip if Ls/Lu are missing
    if "Ls" not in df.columns or "Lu" not in df.columns:
        return

    plt.figure()
    plt.plot(df["epoch"], df["Ls"], label="Ls (supervised loss)")
    plt.plot(df["epoch"], df["Lu"], label="Lu (unsupervised loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Ls and Lu vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ls_lu_curve.png"), dpi=300)
    plt.close()


def plot_lr(df):
    if "lr" not in df.columns:
        return

    plt.figure()
    plt.plot(df["epoch"], df["lr"])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "lr_curve.png"), dpi=300)
    plt.close()


def main():
    ensure_out_dir()
    df = load_data()

    # Basic sanity: sort by epoch just in case
    df = df.sort_values("epoch")

    plot_loss(df)
    plot_psnr(df)
    plot_ssim(df)
    plot_ls_lu(df)
    plot_lr(df)

    print(f"Done. Plots saved in: {OUT_DIR}/")


if __name__ == "__main__":
    main()

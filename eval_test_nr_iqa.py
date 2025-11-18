import os
import glob
import csv

import torch
from PIL import Image
from torchvision import transforms

# -------------------------------------------------------------------
# Patch torch.cov for older PyTorch (like 1.8.1) so that NIQE works
# -------------------------------------------------------------------
if not hasattr(torch, "cov"):
    def _torch_cov(input: torch.Tensor, correction: int = 1):
        """
        Minimal torch.cov implementation for 1D or 2D tensors.

        - input: shape (N,) or (M, N)
          (we treat last dim as samples)
        - correction: same meaning as in modern torch.cov
        """
        x = input
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, N)

        # center along last dim
        mean = x.mean(dim=-1, keepdim=True)
        x = x - mean

        n = x.shape[-1]
        denom = n - correction if n > correction else 1

        # cov = X X^T / (N - correction)
        cov_matrix = x @ x.transpose(-1, -2) / denom
        return cov_matrix

    torch.cov = _torch_cov

# Now that torch.cov exists, we can safely import pyiqa
import pyiqa


def is_image_file(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Paths ---
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Original low-light test images
    orig_dir = os.path.join(base_dir, 'data', 'unlabeled', 'test', 'input')
    # Enhanced images from test.py
    enh_dir  = os.path.join(base_dir, 'result', 'test')
    out_csv  = os.path.join(base_dir, 'test_nr_iqa_scores.csv')

    if not os.path.isdir(orig_dir):
        raise FileNotFoundError(f"Original dir not found: {orig_dir}")
    if not os.path.isdir(enh_dir):
        raise FileNotFoundError(f"Enhanced dir not found: {enh_dir}")

    # --- Transforms: image -> tensor [0,1] ---
    to_tensor = transforms.ToTensor()

    # --- Create metrics ---
    # MUSIQ: higher is better
    musiq_metric = pyiqa.create_metric('musiq', device=device)
    # NIQE: lower is better
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    # BRISQUE: lower is better
    brisque_metric = pyiqa.create_metric('brisque', device=device)

    # --- Collect all original files ---
    orig_paths = sorted(
        [p for p in glob.glob(os.path.join(orig_dir, '*'))
         if os.path.isfile(p) and is_image_file(p)]
    )

    if not orig_paths:
        print(f"No images found in {orig_dir}")
        return

    results = []

    print(f"Found {len(orig_paths)} original test images.")
    print("Starting NR-IQA evaluation (MUSIQ, NIQE, BRISQUE)...\n")

    for i, orig_path in enumerate(orig_paths, 1):
        name = os.path.basename(orig_path)

        # matching enhanced image by name
        enh_path = os.path.join(enh_dir, name)
        if not os.path.isfile(enh_path):
            print(f"[WARN] Enhanced image missing for {name}, skipping.")
            continue

        # load images
        orig_img = Image.open(orig_path).convert('RGB')
        enh_img  = Image.open(enh_path).convert('RGB')

        orig_t = to_tensor(orig_img).unsqueeze(0).to(device)  # (1,3,H,W)
        enh_t  = to_tensor(enh_img).unsqueeze(0).to(device)

        with torch.no_grad():
            # MUSIQ: higher = better
            musiq_orig = musiq_metric(orig_t).item()
            musiq_enh  = musiq_metric(enh_t).item()
            musiq_delta = musiq_enh - musiq_orig  # >0 means improved

            # NIQE: lower = better
            niqe_orig = niqe_metric(orig_t).item()
            niqe_enh  = niqe_metric(enh_t).item()
            niqe_delta = niqe_orig - niqe_enh      # >0 means enhanced is better

            # BRISQUE: lower = better
            brisque_orig = brisque_metric(orig_t).item()
            brisque_enh  = brisque_metric(enh_t).item()
            brisque_delta = brisque_orig - brisque_enh  # >0 means enhanced is better

        results.append([
            name,
            musiq_orig, musiq_enh, musiq_delta,
            niqe_orig, niqe_enh, niqe_delta,
            brisque_orig, brisque_enh, brisque_delta
        ])

        print(
            f"[{i}/{len(orig_paths)}] {name} | "
            f"MUSIQ: orig={musiq_orig:.4f}, enh={musiq_enh:.4f}, Δ={musiq_delta:+.4f} | "
            f"NIQE: orig={niqe_orig:.4f}, enh={niqe_enh:.4f}, Δ(orig-enh)={niqe_delta:+.4f} | "
            f"BRISQUE: orig={brisque_orig:.4f}, enh={brisque_enh:.4f}, Δ(orig-enh)={brisque_delta:+.4f}"
        )

    # --- Write CSV ---
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'filename',
            'musiq_orig', 'musiq_enh', 'musiq_delta_enh_minus_orig',
            'niqe_orig', 'niqe_enh', 'niqe_delta_orig_minus_enh',
            'brisque_orig', 'brisque_enh', 'brisque_delta_orig_minus_enh'
        ])
        writer.writerows(results)

    # --- Summary ---
    if results:
        avg_musiq_delta = sum(r[3] for r in results) / len(results)
        avg_niqe_delta = sum(r[6] for r in results) / len(results)
        avg_brisque_delta = sum(r[9] for r in results) / len(results)

        print(f"\nSaved scores to: {out_csv}")
        print(f"Average MUSIQ Δ (enh - orig)          = {avg_musiq_delta:+.4f}  (higher is better)")
        print(f"Average NIQE Δ (orig - enh)           = {avg_niqe_delta:+.4f}  (positive means improvement)")
        print(f"Average BRISQUE Δ (orig - enh)        = {avg_brisque_delta:+.4f}  (positive means improvement)")
    else:
        print("No results computed (no matching image pairs).")


if __name__ == '__main__':
    main()

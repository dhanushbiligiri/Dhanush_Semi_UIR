import os
import glob
import csv

import torch
from PIL import Image
from torchvision import transforms
import pyiqa

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths (adjust if your folders differ)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    orig_dir = os.path.join(base_dir, 'data', 'unlabeled', 'test', 'input')
    enh_dir  = os.path.join(base_dir, 'result', 'test')   # where test.py saved outputs
    out_csv  = os.path.join(base_dir, 'test_musiq_scores.csv')

    # Image -> tensor [0,1]
    to_tensor = transforms.ToTensor()

    # MUSIQ metric, higher = better
    metric = pyiqa.create_metric('musiq', device=device)

    # Collect all original files
    orig_paths = sorted(
        [p for p in glob.glob(os.path.join(orig_dir, '*')) 
         if os.path.isfile(p)]
    )

    results = []
    for i, orig_path in enumerate(orig_paths, 1):
        name = os.path.basename(orig_path)

        # matching enhanced image by name
        enh_path = os.path.join(enh_dir, name)
        if not os.path.isfile(enh_path):
            print(f'[WARN] Enhanced image missing for {name}, skipping.')
            continue

        # load images
        orig_img = Image.open(orig_path).convert('RGB')
        enh_img  = Image.open(enh_path).convert('RGB')

        orig_t = to_tensor(orig_img).unsqueeze(0).to(device)  # (1,3,H,W)
        enh_t  = to_tensor(enh_img).unsqueeze(0).to(device)

        with torch.no_grad():
            score_orig = metric(orig_t).item()
            score_enh  = metric(enh_t).item()

        delta = score_enh - score_orig
        results.append([name, score_orig, score_enh, delta])

        print(f'[{i}/{len(orig_paths)}] {name} | '
              f'orig={score_orig:.4f}, enh={score_enh:.4f}, Δ={delta:.4f}')

    # write CSV
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'musiq_orig', 'musiq_enh', 'delta'])
        writer.writerows(results)

    # simple summary
    if results:
        avg_delta = sum(r[3] for r in results) / len(results)
        print(f'\nSaved scores to: {out_csv}')
        print(f'Average MUSIQ improvement Δ = {avg_delta:.4f}')
    else:
        print('No results computed.')

if __name__ == '__main__':
    main()

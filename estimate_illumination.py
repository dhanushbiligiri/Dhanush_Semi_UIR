import os
import cv2
import numpy as np
from glob import glob
from os.path import join
from PIL import Image

def luminance_estimation(img):
    sigma_list = [15, 60, 90]
    img = np.uint8(np.array(img))
    illuminance = np.ones_like(img).astype(np.float32)
    for sigma in sigma_list:
        illuminance1 = np.log10(cv2.GaussianBlur(img, (0, 0), sigma) + 1e-8)
        illuminance1 = np.clip(illuminance1, 0, 255)
        illuminance = illuminance + illuminance1
    illuminance = illuminance / 3
    L = (illuminance - np.min(illuminance)) / (np.max(illuminance) - np.min(illuminance) + 1e-6)
    L = np.uint8(L * 255)
    return L

def process_dir(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    input_lists = glob(join(input_dir, "*.*"))
    print(f"Processing {len(input_lists)} images from {input_dir}")
    for p in input_lists:
        img = Image.open(p).convert("RGB")
        fname = os.path.basename(p)
        L = luminance_estimation(img)
        out_img = Image.fromarray(L)
        out_img.save(os.path.join(output_dir, fname))
    print(f"Done: {output_dir}")

if __name__ == "__main__":
    # labeled train
    process_dir("data/labeled1/input", "data/labeled1/LA")
    # val
    process_dir("data/val/input", "data/val/LA")
    # unlabeled
    process_dir("data/unlabeled/input", "data/unlabeled/LA")

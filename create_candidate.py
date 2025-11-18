import os
import torch
from glob import glob
from os.path import join
from torchvision.transforms import ToPILImage

input_dir = 'data/unlabeled/input'
result_dir = 'data/unlabeled/candidate'
os.makedirs(result_dir, exist_ok=True)

input_lists = glob(join(input_dir, '*.*'))

to_pil = ToPILImage()

print(f"Creating candidates for {len(input_lists)} images...")
for p in input_lists:
    img_name = os.path.basename(p)
    print(f"Creating candidate for: {img_name}")
    # start with zero-image placeholder
    img = torch.zeros((3, 256, 256))
    res = to_pil(img).convert('RGB')
    res.save(os.path.join(result_dir, img_name))

print(f"Done. Created {len(input_lists)} candidate images.")

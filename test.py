import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from PIL import Image

from model import AIMnet
from dataset_all import TestData


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- base dir = folder where test.py lives ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # use your trained checkpoint
    model_root = os.path.join(BASE_DIR, "model", "ckpt", "model_e200.pth")

    # TestData expects: dataroot/input and dataroot/LA
    data_root = os.path.join(BASE_DIR, "data", "test")

    save_path = os.path.join(BASE_DIR, "result", "test")
    os.makedirs(save_path, exist_ok=True)

    dataset = TestData(data_root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"Found {len(dataset)} test images.")

    checkpoint = torch.load(model_root, map_location=device)

    net = AIMnet().to(device)
    net = nn.DataParallel(net)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()

    print(f"Loaded checkpoint from {model_root} (epoch {checkpoint.get('epoch', 'unknown')})")
    print("START INFERENCE!")

    with torch.no_grad():
        for idx, (data_input, data_la) in enumerate(dataloader):
            data_input = Variable(data_input).to(device)
            data_la = Variable(data_la).to(device)

            result, _ = net(data_input, data_la)

            img_path = dataset.A_paths[idx]
            name = os.path.basename(img_path)
            print(f"[{idx+1}/{len(dataset)}] {name}")

            temp_res = result[0].cpu().numpy().transpose(1, 2, 0)
            temp_res = np.clip(temp_res, 0, 1)
            temp_res = (temp_res * 255).astype(np.uint8)
            Image.fromarray(temp_res).save(os.path.join(save_path, name))

    print("Finished! Results saved to:", save_path)


if __name__ == "__main__":
    main()

import torch
import random
from math import log10
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)


# recommend
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        # m.weight.data.normal_(0, 0.02)
        # m.bias.data.zero_()
        # nn.init.xavier_normal_(m.weight.data)
        nn.init.kaiming_normal(m.weight.data, mode='fan_out')
        # nn.init.xavier_normal_(m.bias.data)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def to_psnr(J, gt):
    mse = F.mse_loss(J, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]
    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def create_emamodel(net, ema=True):
    if ema:
        for param in net.parameters():
            param.detach_()
    return net


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_psnr_ssim(recoverd, clean):
    """
    recoverd, clean: torch tensors of shape (N, C, H, W), values in [0, 1]
    Returns:
        avg_psnr, avg_ssim, N
    """
    # Clamp and move to CPU numpy
    recoverd = torch.clamp(recoverd, 0.0, 1.0).detach().cpu().numpy()
    clean = torch.clamp(clean, 0.0, 1.0).detach().cpu().numpy()

    # N, C, H, W
    N = clean.shape[0]
    psnr = 0.0
    ssim = 0.0

    for i in range(N):
        # (C, H, W) -> (H, W, C)
        rec = np.transpose(recoverd[i], (1, 2, 0))
        gt = np.transpose(clean[i], (1, 2, 0))

        # PSNR
        psnr += peak_signal_noise_ratio(gt, rec, data_range=1.0)

        # SSIM – choose ONE of these depending on skimage version:

        # If your skimage supports channel_axis (newer versions):
        ssim += structural_similarity(gt, rec, data_range=1.0, channel_axis=-1)

        # If that errors, comment the line above and uncomment this instead:
        # ssim += structural_similarity(gt, rec, data_range=1.0, multichannel=True)

    psnr /= N
    ssim /= N

    return psnr, ssim, N


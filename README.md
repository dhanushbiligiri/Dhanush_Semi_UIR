# Semi-Supervised Underwater Image Restoration (Semi-UIR) — Reproduction Project

This repository contains a full reproduction of the CVPR 2023 method  
**“Contrastive Semi-Supervised Learning for Underwater Image Restoration via Reliable Bank (Semi-UIR)”**,  
implemented using PyTorch with a working teacher–student framework, reliable bank, illumination maps,  
supervised + unsupervised losses, and full NR-IQA evaluation.

---
```bash
## 1. Project Structure

Dhanush_Semi_UIR/
├── attention.py
├── create_candidate.py
├── dataset_all.py
├── deform_conv.py
├── estimate_illumination.py
├── eval_test_musiq.py
├── eval_test_nr_iqa.py
├── logs.txt
├── logs_parse.py
├── loss/
├── metric_figs/
├── model.py
├── plot_metrics.py
├── plot_test_nr_iqa.py
├── prepare_labeled_val.py
├── splitpairs.py
├── test.py
├── train.py
├── trainer.py
└── utils.py
```

---

## 2. Environment Setup

```bash
conda create -n EE5522 python=3.8
conda activate EE5522
pip install -r requirements.txt

## Major dependencies:

# PyTorch
# torchvision
# mmcv (for deformable convolution)
# pyiqa (MUSIQ/NIQE/BRISQUE)
# scikit-image
# tqdm, numpy, pillow
```
```bash
## 3. Data Preparation

Required folder structure:

data/
├── labeled1/
│   ├── input/
│   ├── GT/
│   └── LA/
├── unlabeled/
│   ├── input/
│   ├── LA/
│   └── candidate/
├── val/
│   ├── input/
│   ├── GT/
│   └── LA/
└── test/
    ├── input/
    └── LA/
```

### Step 1 — Split paired data (train + val)
python prepare_labeled_val.py
### or
python splitpairs.py

### Step 2 — Generate illumination maps (LA)
python estimate_illumination.py

### Step 3 — Initialize the reliable bank (empty placeholders)
python create_candidate.py


Creates zero-image placeholders in unlabeled/candidate/.

## 4. Training

Start training:
```bash
python train.py --data_dir ./data
```

This trains:
- Student model (AIM-Net)
- Teacher model (EMA updates)
- Structure + perceptual + gradient supervised losses
- Contrastive + L1 unsupervised losses
- Reliable bank updated using MUSIQ quality scores
- Checkpoints saved under:
    - model/ckpt/model_e{epoch}.pth

## 5. Inference (Testing)

Prepare your test images at:
```bash
data/test/input/
```

Generate illumination maps:
```bash
python estimate_illumination.py
```

Run inference:
```bash
python test.py
```

Outputs saved to:
```bash
result/test/
```
## 6. Evaluation (NR-IQA: MUSIQ, NIQE, BRISQUE)
Full NR-IQA evaluation
```bash
python eval_test_nr_iqa.py
```

Outputs CSV:
```bash
test_nr_iqa_scores.csv
```

Metrics:
- MUSIQ — higher is better
- NIQE — lower is better
- BRISQUE — lower is better
- MUSIQ-only evaluation

```bash
python eval_test_musiq.py
# Creates test_musiq_scores.csv.
```
## 7. Plotting Training & Evaluation Metrics
Training curves (loss, PSNR, SSIM, LR)
```bash
python logs_parse.py
python plot_metrics.py
```

Saved in:
```bash
metric_figs/
```
NR-IQA comparison plots
```bash
python plot_test_nr_iqa.py
```

Saved in:
```bash
plots_nr_iqa/
Eval_plots/
```
## 8. Model Overview

This reproduction implements the full Semi-UIR framework, including:
- AIM-Net backbone
- Illumination-Guided Modulation (IGM)
- Deformable Convolution (DCN)
- Non-local Sparse Attention
- Atrous multi-scale feature blocks
- Gradient-aware enhancement branch
- Attention Feature Fusion (AFF)
- EMA teacher network
- Reliable bank based on MUSIQ filtering
- Supervised + unsupervised joint optimization

## 9. Reliable Bank Mechanism

### For each unlabeled image:

    teacher_output = EMA_teacher(x)
    
    student_output = student(x)
    
    bank_image = stored best pseudo-label

If MUSIQ(teacher_output) > MUSIQ(student_output)

    and MUSIQ(teacher_output) > MUSIQ(bank_image):
    
        bank_image ← teacher_output


This prevents confirmation bias by ensuring only high-quality pseudo-labels are stored.

## 10. Loss Functions
- Supervised Loss (on labeled pairs)
- Structure loss (MyLoss)
- Perceptual loss (VGG16 features)
- Gradient loss
- Unsupervised Loss (on unlabeled images)
- L1 loss between student output and bank pseudo-label
- Contrastive consistency loss
- Consistency ramp-up during early epochs
- Total loss:
    - L_total = L_supervised + w(t) * L_unsupervised

## 11. Citation of original work
@inproceedings{huang2023contrastive,
  title={Contrastive Semi-supervised Learning for Underwater Image Restoration via Reliable Bank},
  author={Huang, Shirui and Wang, Keyan and Liu, Huan and Chen, Jun and Li, Yunsong},
  booktitle={CVPR},
  pages={18145--18155},
  year={2023}
}

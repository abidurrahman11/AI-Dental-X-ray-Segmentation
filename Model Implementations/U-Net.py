# Block - 1: Load dataset in Kaggle environment
import os
DATASET_PATH = "/kaggle/input/datasets/abidurrahman21/dataset"

print("Images:", len(os.listdir(os.path.join(DATASET_PATH, "images"))))
print("Masks :", len(os.listdir(os.path.join(DATASET_PATH, "masks"))))

# Pairing check
img_files = sorted([f for f in os.listdir(os.path.join(DATASET_PATH, "images")) if f.endswith('.jpg')])
print("\nFirst 3 images:", img_files[:3])
if img_files:
    base = img_files[0].rsplit('.', 1)[0]
    mask_name = base + ".png"
    print("Expected mask:", mask_name)
    print("Mask exists?", os.path.exists(os.path.join(DATASET_PATH, "masks", mask_name)))

# Block - 2: Install required packages in Kaggle environment. this is not needed in local environment. So keep it commented out in local environment.
# !pip install -q segmentation-models-pytorch albumentations


#Block - 3: Visualize some samples from the dataset
# ────────────────────────────────────────────────────────────────
# Step 1 – Imports, Configuration & Seeds
# ────────────────────────────────────────────────────────────────

import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import directed_hausdorff
import segmentation_models_pytorch as smp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")


# Block - 4: Define data loader
from glob import glob
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─── CONFIG ────────────────────────────────────────────────────────────────
# Seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


IMAGE_DIR = os.path.join(DATASET_PATH, "images")
MASK_DIR  = os.path.join(DATASET_PATH, "masks")

# ─── Collect all pairs ─────────────────────────────────────────────────────
img_paths = sorted(glob(os.path.join(IMAGE_DIR, "*.jpg")))
mask_paths = []

for img_p in img_paths:
    base = os.path.basename(img_p).rsplit('.', 1)[0]
    mask_name = base + ".png"
    mask_p = os.path.join(MASK_DIR, mask_name)
    if os.path.exists(mask_p):
        mask_paths.append(mask_p)
    else:
        print(f"Warning: missing mask for {img_p}")

print(f"Loaded {len(img_paths)} image-mask pairs")

# ─── Split: train / val / test (80/10/10) ─────────────────────────────────
train_imgs, test_imgs, train_masks, test_masks = train_test_split(
    img_paths, mask_paths, test_size=0.2, random_state=SEED
)

val_imgs, test_imgs, val_masks, test_masks = train_test_split(
    test_imgs, test_masks, test_size=0.5, random_state=SEED
)

print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")


# Block - 5: Define augmentations and custom dataset
# ─── Augmentations ─────────────────────────────────────────────────────────
train_transform = A.Compose([
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=1.0),   # standard for dental X-rays
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.4),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
    A.GaussNoise(p=0.3),
    A.Normalize(mean=0.0, std=1.0),   # simple [0,1] norm — works well
    ToTensorV2(),
], p=1.0)

val_test_transform = A.Compose([
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=1.0),
    A.Normalize(mean=0.0, std=1.0),
    ToTensorV2(),
], p=1.0)


# ─── Custom Dataset ────────────────────────────────────────────────────────
class DentalDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Read original images and masks as NumPy arrays
        img  = cv2.imread(self.img_paths[idx],  cv2.IMREAD_GRAYSCALE)   # (H, W)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)   # (H, W), values 0 & 1
    
        if self.transform is not None:
            # Apply augmentations
            augmented = self.transform(image=img, mask=mask)
    
            img_tensor  = augmented['image']   # already tensor (C, H, W)
            mask_tensor = augmented['mask']    # already tensor (H, W) or (1, H, W)
    
            # Convert mask tensor → float binary (0.0 / 1.0)
            if mask_tensor.dtype != torch.float32:
                mask_tensor = mask_tensor.float()
    
            # Make sure it's binary 0/1 (handles 0/1 or 0/255 cases safely)
            mask_tensor = (mask_tensor > 0.5).float()   # threshold at 0.5
    
            # Ensure shape is (1, H, W) for segmentation loss
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            elif mask_tensor.dim() == 3 and mask_tensor.shape[0] != 1:
                mask_tensor = mask_tensor[:1]   # take first channel if multiple
    
        else:
            # No transform case (fallback - rare)
            img_tensor = torch.from_numpy(img).float().unsqueeze(0)  # (1, H, W)
            mask_np = (mask > 0).astype(np.float32)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
    
        # Stack grayscale to 3 channels if needed (for pretrained backbones)
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
    
        return img_tensor, mask_tensor
    

# Block - 6: Create Datasets
train_ds = DentalDataset(train_imgs, train_masks, train_transform)
val_ds   = DentalDataset(val_imgs,   val_masks,   val_test_transform)
test_ds  = DentalDataset(test_imgs,  test_masks,  val_test_transform)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=16,shuffle=False, num_workers=4, pin_memory=True)

print(f"Test set size: {len(test_ds)} images")
print("Test loader created successfully!")

# Visualize one sample
img, mask = train_ds[0]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.imshow(img[0].cpu().numpy(), cmap='gray')  # channel 0
ax1.set_title("CLAHE + Augmented Image")
ax1.axis('off')

ax2.imshow(mask[0].cpu().numpy(), cmap='gray')
ax2.set_title("Binary Mask")
ax2.axis('off')

# Simple overlay
overlay = img[0].cpu().numpy().copy()
overlay[mask[0].cpu().numpy() > 0.5] = 1.0  # make teeth white/bright
ax3.imshow(overlay, cmap='gray')
ax3.set_title("Overlay")
ax3.axis('off')

plt.tight_layout()
plt.show()



# Block - 7: Loss metrics, training and validation functions
# Cell 2: Loss + Full Metrics (Accuracy, Dice, F1, IoU + HD95 for test)
dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True)
bce_loss  = nn.BCEWithLogitsLoss()

def combined_loss(pred, target):
    return 0.5 * dice_loss(pred, target) + 0.5 * bce_loss(pred, target)

def compute_epoch_metrics(preds, targets):
    """Compute all 4 metrics for a batch (used in train/val)"""
    pred_prob = torch.sigmoid(preds)
    pred_bin = (pred_prob > 0.5).float()
    
    intersection = (pred_bin * targets).sum(dim=(2,3))
    union = pred_bin.sum(dim=(2,3)) + targets.sum(dim=(2,3)) - intersection
    
    dice = (2 * intersection + 1e-6) / (pred_bin.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + 1e-6)
    iou  = (intersection + 1e-6) / (union + 1e-6)
    acc  = (pred_bin == targets).float().mean(dim=(2,3))
    f1   = dice   # binary case
    
    return {
        'dice': dice.mean().item(),
        'iou':  iou.mean().item(),
        'f1':   f1.mean().item(),
        'acc':  acc.mean().item()
    }

def compute_test_metrics(pred_logits, target, smooth=1e-6):
    """Full metrics + HD95 for final test evaluation"""
    pred_prob = torch.sigmoid(pred_logits)
    pred_bin = (pred_prob > 0.5).float()
    
    intersection = (pred_bin * target).sum(dim=(2,3))
    union = pred_bin.sum(dim=(2,3)) + target.sum(dim=(2,3)) - intersection
    
    dice = (2 * intersection + smooth) / (pred_bin.sum(dim=(2,3)) + target.sum(dim=(2,3)) + smooth)
    iou  = (intersection + smooth) / (union + smooth)
    prec = intersection / (pred_bin.sum(dim=(2,3)) + smooth)
    rec  = intersection / (target.sum(dim=(2,3)) + smooth)
    acc  = (pred_bin == target).float().mean(dim=(2,3))
    f1   = dice
    
    # HD95 (per sample)
    hd95_list = []
    for i in range(pred_bin.shape[0]):
        p = pred_bin[i,0].cpu().numpy()
        t = target[i,0].cpu().numpy()
        if np.sum(p) == 0 or np.sum(t) == 0:
            hd95_list.append(0.0)
        else:
            coords_p = np.argwhere(p > 0.5)
            coords_t = np.argwhere(t > 0.5)
            hd1 = directed_hausdorff(coords_p, coords_t)[0]
            hd2 = directed_hausdorff(coords_t, coords_p)[0]
            hd95_list.append(np.percentile([hd1, hd2], 95))
    
    return {
        'Accuracy': acc.mean().item(),
        'Dice': dice.mean().item(),
        'F1': f1.mean().item(),
        'IoU': iou.mean().item(),
        'Precision': prec.mean().item(),
        'Recall': rec.mean().item(),
        'HD95': np.mean(hd95_list)
    }



# Block - 8: Training and validation functions
# Cell 3: Train & Validate Functions (with all metrics)
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = total_dice = total_iou = total_f1 = total_acc = 0.0
    count = 0
    
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        
        metrics = compute_epoch_metrics(preds, masks)
        
        total_loss += loss.item()
        total_dice += metrics['dice']
        total_iou  += metrics['iou']
        total_f1   += metrics['f1']
        total_acc  += metrics['acc']
        count += 1
    
    return (total_loss/count, total_dice/count, total_iou/count, total_f1/count, total_acc/count)

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_dice = total_iou = total_f1 = total_acc = 0.0
    count = 0
    
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            metrics = compute_epoch_metrics(preds, masks)
            
            total_loss += loss.item()
            total_dice += metrics['dice']
            total_iou  += metrics['iou']
            total_f1   += metrics['f1']
            total_acc  += metrics['acc']
            count += 1
    
    return (total_loss/count, total_dice/count, total_iou/count, total_f1/count, total_acc/count)




# Block - 9: Define U-Net model. Model trainig loop
# ────────────────────────────────────────────────────────────────
# Vanilla U-Net (Classic from-scratch U-Net)
# ────────────────────────────────────────────────────────────────

import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class VanillaUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))
        
        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)

print("Vanilla U-Net defined successfully")



# Block - 10: Train the model
# ────────────────────────────────────────────────────────────────
# Train Vanilla U-Net
# ────────────────────────────────────────────────────────────────

MODEL_NAME = "vanilla_unet"   # Fixed name for this model

model = VanillaUNet(in_channels=3, out_channels=1).to(DEVICE)

LR = 1e-4
NUM_EPOCHS = 65
PATIENCE = 15

# Slightly higher learning rate is often better for vanilla U-Net
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

history = {'epoch': [], 'train_loss':[], 'train_dice':[], 'train_iou':[], 'train_f1':[], 'train_acc':[],
           'val_loss':[], 'val_dice':[], 'val_iou':[], 'val_f1':[], 'val_acc':[]}

best_val_dice = 0.0
patience_counter = 0
best_path = f"best_{MODEL_NAME}.pth"

for epoch in range(1, NUM_EPOCHS + 1):
    t_loss, t_dice, t_iou, t_f1, t_acc = train_one_epoch(model, train_loader, optimizer, combined_loss, DEVICE)
    v_loss, v_dice, v_iou, v_f1, v_acc = validate_one_epoch(model, val_loader, combined_loss, DEVICE)
    scheduler.step()
    
    history['epoch'].append(epoch)
    history['train_loss'].append(t_loss)
    history['train_dice'].append(t_dice)
    history['train_iou'].append(t_iou)
    history['train_f1'].append(t_f1)
    history['train_acc'].append(t_acc)
    history['val_loss'].append(v_loss)
    history['val_dice'].append(v_dice)
    history['val_iou'].append(v_iou)
    history['val_f1'].append(v_f1)
    history['val_acc'].append(v_acc)
    
    print(f"[VANILLA U-Net] Epoch {epoch:3d} | Train Dice: {t_dice:.4f} | Val Dice: {v_dice:.4f}")
    
    if v_dice > best_val_dice:
        best_val_dice = v_dice
        patience_counter = 0
        torch.save(model.state_dict(), best_path)
        print(f"   → New best model saved (Val Dice: {v_dice:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("   Early stopping triggered")
            break

print(f"\nVanilla U-Net training completed. Best Val Dice: {best_val_dice:.4f}")




# Block - 11: Show training history graphs
import matplotlib.pyplot as plt

epochs = history['epoch']

# Define two contrasting colors (consistent across all plots)
train_color = '#1f77b4'   # blue
val_color   = '#d62728'   # red

metrics = [
    ('Dice', 'train_dice', 'val_dice'),
    ('IoU', 'train_iou', 'val_iou'),
    ('F1', 'train_f1', 'val_f1'),
    ('Accuracy', 'train_acc', 'val_acc')
]

for title, t_key, v_key in metrics:
    plt.figure(figsize=(8, 6))

    plt.plot(epochs, history[t_key],
             label=f'Train {title}',
             color=train_color,
             linewidth=2.5)

    plt.plot(epochs, history[v_key],
             label=f'Validation {title}',
             color=val_color,
             linewidth=2.5)

    plt.title(f'{MODEL_NAME.upper()} - {title}', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(title, fontsize=12)

    plt.legend(frameon=False, fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save each figure separately
    filename = f'{MODEL_NAME}_{title.lower()}_curve.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()

    print(f"{title} graph saved as '{filename}'")

print(f"\n{MODEL_NAME.upper()} training completed. Best Val Dice: {best_val_dice:.4f}")




# Block - 12: Test evaluation with all metrics
# Cell 5: Final Test Evaluation + All Metrics Print
model.load_state_dict(torch.load("best_vanilla_unet.pth"))
model.eval()

test_results = {'Accuracy':[], 'Dice':[], 'F1':[], 'IoU':[], 'Precision':[], 'Recall':[], 'HD95':[]}

with torch.no_grad():
    for imgs, masks in test_loader:
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)
        preds = model(imgs)
        m = compute_test_metrics(preds, masks)
        for k in test_results:
            test_results[k].append(m[k])

# Final Results
print(f"\n{'='*60}")
print(f"FINAL TEST RESULTS - {MODEL_NAME.upper()}")
print(f"{'='*60}")
for k in ['Accuracy', 'Dice', 'F1', 'IoU', 'Precision', 'Recall', 'HD95']:
    mean = np.mean(test_results[k])
    std  = np.std(test_results[k])
    print(f"{k:12}: {mean:.4f} ± {std:.4f}")
print(f"{'='*60}")



# Block - 13: Visualize predictions on test set
# ─── Robust Visualization: Original, GT mask, Prediction, Overlay ────────
import random
import matplotlib.pyplot as plt
import numpy as np

model.eval()

NUM_EXAMPLES = 6
indices = random.sample(range(len(test_ds)), NUM_EXAMPLES)

fig, axes = plt.subplots(NUM_EXAMPLES, 4, figsize=(16, 4 * NUM_EXAMPLES))
fig.suptitle(f"Qualitative Results – {MODEL_NAME.upper()} (Best Checkpoint)", 
             fontsize=16, fontweight='bold', y=1.02)

for row, idx in enumerate(indices):
    img_tensor, mask_tensor = test_ds[idx]
    
    # Move to device & add batch dim
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)   # (1, C, H, W)
    
    with torch.no_grad():
        pred_logits = model(img_tensor)               # (1, 1, H, W)
        pred_prob = torch.sigmoid(pred_logits)
        pred_mask = (pred_prob > 0.5).float()         # (1, 1, H, W)
    
    # ── Convert to numpy safely ────────────────────────────────────────
    # Image: take channel 0 (grayscale)
    img_np = img_tensor[0, 0].cpu().numpy()           # (H, W)
    
    # Ground truth mask: remove channel dim if present
    gt_mask = mask_tensor.squeeze().cpu().numpy()     # (H, W)
    
    # Predicted mask
    pred_mask_np = pred_mask[0, 0].cpu().numpy()      # (H, W)
    
    # ── Plotting ────────────────────────────────────────────────────────
    axes[row, 0].imshow(img_np, cmap='gray')
    axes[row, 0].set_title("Original Image")
    axes[row, 0].axis('off')
    
    axes[row, 1].imshow(gt_mask, cmap='gray')
    axes[row, 1].set_title("Ground Truth")
    axes[row, 1].axis('off')
    
    axes[row, 2].imshow(pred_mask_np, cmap='gray')
    axes[row, 2].set_title("Prediction")
    axes[row, 2].axis('off')
    
    # Overlay
    overlay = img_np.copy()
    overlay = np.stack([overlay]*3, axis=-1)          # grayscale → RGB
    red_mask = np.zeros_like(overlay)
    red_mask[pred_mask_np > 0.5] = [1.0, 0.3, 0.3]    # light red
    
    blended = 0.65 * overlay + 0.35 * red_mask
    blended = np.clip(blended, 0, 1)
    
    axes[row, 3].imshow(blended)
    axes[row, 3].set_title("Overlay (pred)")
    axes[row, 3].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig(f"{MODEL_NAME}_qualitative_results.png", dpi=300, bbox_inches='tight')
plt.show()

print(f"Visualization saved: {MODEL_NAME}_qualitative_results.png")
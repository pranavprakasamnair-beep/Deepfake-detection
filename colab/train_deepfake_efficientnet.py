# =============================================================================
# DEEPFAKE DETECTION — EfficientNet-B4 Training on Google Colab
# =============================================================================
# HOW TO USE:
# 1. Go to https://colab.research.google.com
# 2. File > New Notebook
# 3. Copy each section between CELL markers into a separate code cell
# 4. Runtime > Change runtime type > T4 GPU
# 5. Run cells top to bottom
# =============================================================================


# =============================================================================
# CELL 1 — Install Dependencies
# =============================================================================
# Paste this into Cell 1 and run it first.

# !pip install -q timm datasets transformers accelerate


# =============================================================================
# CELL 2 — Imports
# =============================================================================

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import timm
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from google.colab import files


def seed_everything(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)
print("All imports successful.")


# =============================================================================
# CELL 3 — Configuration
# =============================================================================

# ---- Hyperparameters --------------------------------------------------------
BATCH_SIZE    = 32        # images per mini-batch
EPOCHS        = 10        # maximum training epochs
LR            = 1e-4      # AdamW learning rate
WEIGHT_DECAY  = 1e-4      # AdamW weight decay
IMG_SIZE      = 224       # spatial resolution fed to EfficientNet-B4
NUM_CLASSES   = 2         # binary: REAL vs FAKE
PATIENCE      = 3         # early-stopping patience (epochs without val improvement)
NUM_WORKERS   = 2         # DataLoader worker processes
SEED          = 42        # global random seed
MODEL_NAME    = "efficientnet_b4"
SAVE_PATH     = "efficientnet_deepfake.pt"

# ---- Label mapping ----------------------------------------------------------
# Dataset convention  →  label 0 = FAKE,  label 1 = REAL
# We keep the original HuggingFace label integers as-is.
# The model output neuron 0 corresponds to FAKE, neuron 1 corresponds to REAL.
LABEL_NAMES = {0: "FAKE", 1: "REAL"}

# ---- Device -----------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU name     : {torch.cuda.get_device_name(0)}")
    print(f"GPU memory   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# =============================================================================
# CELL 4 — Load Dataset from HuggingFace
# =============================================================================

print("Loading dataset  dima806/deepfake_vs_real_faces  from HuggingFace …")
print("(This may take a few minutes on first run — ~1.5 GB download)")

hf_dataset = load_dataset("dima806/deepfake_vs_real_faces", split="train")

print(f"\nDataset loaded!")
print(f"  Total samples : {len(hf_dataset)}")
print(f"  Features      : {hf_dataset.features}")

# ---- Inspect label distribution ---------------------------------------------
labels_all = hf_dataset["label"]
n_fake = labels_all.count(0)
n_real = labels_all.count(1)
print(f"\n  Label 0 (FAKE) : {n_fake:,}  ({100*n_fake/len(labels_all):.1f}%)")
print(f"  Label 1 (REAL) : {n_real:,}  ({100*n_real/len(labels_all):.1f}%)")

# ---- Train / Validation split (80 / 20) -------------------------------------
total       = len(hf_dataset)
train_size  = int(0.80 * total)
val_size    = total - train_size

# Shuffle indices with fixed seed then split
all_indices = list(range(total))
random.seed(SEED)
random.shuffle(all_indices)

train_indices = all_indices[:train_size]
val_indices   = all_indices[train_size:]

print(f"\n  Train samples : {len(train_indices):,}")
print(f"  Val   samples : {len(val_indices):,}")


# =============================================================================
# CELL 5 — Custom Dataset Class with Augmentation Transforms
# =============================================================================

class DeepfakeDataset(Dataset):
    """
    Wraps the HuggingFace dataset split and applies torchvision transforms.

    Parameters
    ----------
    hf_data   : HuggingFace Dataset object (the full dataset)
    indices   : list[int] — which rows of hf_data to use
    transform : torchvision.transforms pipeline
    """

    def __init__(self, hf_data, indices, transform=None):
        self.hf_data   = hf_data
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample   = self.hf_data[real_idx]

        # HuggingFace returns PIL images when the feature type is Image
        image = sample["image"]
        label = sample["label"]

        # Ensure RGB (some images in the wild are RGBA or grayscale)
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ---- Normalization statistics (ImageNet mean/std — standard for pretrained) -
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---- Training transform (with augmentation) ---------------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
    ),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(
        p=0.25,
        scale=(0.02, 0.20),
        ratio=(0.3, 3.3),
        value=0,
        inplace=False,
    ),
])

# ---- Validation transform (no augmentation — only resize + normalize) -------
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---- Instantiate datasets ---------------------------------------------------
train_dataset = DeepfakeDataset(hf_dataset, train_indices, transform=train_transform)
val_dataset   = DeepfakeDataset(hf_dataset, val_indices,   transform=val_transform)

print(f"Train dataset : {len(train_dataset):,} samples")
print(f"Val   dataset : {len(val_dataset):,}  samples")

# ---- Quick sanity-check: inspect one sample ---------------------------------
sample_img, sample_lbl = train_dataset[0]
print(f"\nSanity check:")
print(f"  Image tensor shape : {sample_img.shape}")
print(f"  Image dtype        : {sample_img.dtype}")
print(f"  Label              : {sample_lbl}  ({LABEL_NAMES[sample_lbl]})")


# =============================================================================
# CELL 6 — Create DataLoaders
# =============================================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,        # drop last incomplete batch for stable BN stats
    persistent_workers=True if NUM_WORKERS > 0 else False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True if NUM_WORKERS > 0 else False,
)

print(f"Train batches per epoch : {len(train_loader)}")
print(f"Val   batches per epoch : {len(val_loader)}")


# =============================================================================
# CELL 7 — Build EfficientNet-B4 Model
# =============================================================================
# Label convention reminder:
#   Model output index 0  →  FAKE  (dataset label 0)
#   Model output index 1  →  REAL  (dataset label 1)

def build_model(model_name: str = MODEL_NAME, num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Creates a pretrained EfficientNet-B4 from the timm library and replaces
    the classification head with a new Linear layer for `num_classes` outputs.

    Returns
    -------
    model : nn.Module already moved to DEVICE
    """
    model = timm.create_model(
        model_name,
        pretrained=True,      # download ImageNet weights
        num_classes=num_classes,
    )
    # Freeze the stem and early blocks to speed up early training;
    # unfreeze everything so the full network can fine-tune.
    # (All parameters are trainable from epoch 1.)
    for param in model.parameters():
        param.requires_grad = True

    model = model.to(DEVICE)
    return model


model = build_model()

# ---- Parameter count --------------------------------------------------------
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model          : {MODEL_NAME}")
print(f"Total params   : {total_params:,}")
print(f"Trainable      : {trainable_params:,}")
print(f"Classifier head: {model.classifier}")


# =============================================================================
# CELL 8 — Training & Validation Functions
# =============================================================================

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999),
    eps=1e-8,
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=1e-6,
)


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Runs one full pass over `loader` updating model weights.

    Returns
    -------
    avg_loss : float  — mean cross-entropy loss over all batches
    accuracy : float  — fraction of correctly classified samples  (0–1)
    """
    model.train()
    running_loss    = 0.0
    correct         = 0
    total_samples   = 0
    n_batches       = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping for training stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # ---- Metrics --------------------------------------------------------
        running_loss  += loss.item() * images.size(0)
        preds          = logits.argmax(dim=1)
        correct       += (preds == labels).sum().item()
        total_samples += images.size(0)

        # ---- Progress logging every 50 batches ------------------------------
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n_batches:
            batch_loss = running_loss / total_samples
            batch_acc  = correct / total_samples
            print(
                f"  Epoch {epoch:>2} | Batch {batch_idx+1:>4}/{n_batches}"
                f" | Loss: {batch_loss:.4f} | Acc: {batch_acc:.4f}",
                end="\r",
                flush=True,
            )

    print()  # newline after \r
    avg_loss = running_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Evaluates the model on `loader` without updating weights.

    Returns
    -------
    avg_loss : float
    accuracy : float  (0–1)
    """
    model.eval()
    running_loss  = 0.0
    correct       = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)

        running_loss  += loss.item() * images.size(0)
        preds          = logits.argmax(dim=1)
        correct       += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = running_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy


print("Training and validation functions defined.")
print(f"Optimizer : AdamW  (lr={LR}, wd={WEIGHT_DECAY})")
print(f"Scheduler : CosineAnnealingLR  (T_max={EPOCHS})")
print(f"Loss      : CrossEntropyLoss   (label_smoothing=0.05)")


# =============================================================================
# CELL 9 — Run Training Loop, Early Stopping, Save Best Model
# =============================================================================

history = {
    "train_loss": [],
    "train_acc":  [],
    "val_loss":   [],
    "val_acc":    [],
    "lr":         [],
}

best_val_acc    = 0.0
best_val_loss   = float("inf")
patience_counter = 0

print("=" * 65)
print(f"Starting training: {EPOCHS} epochs, device={DEVICE}")
print("=" * 65)

for epoch in range(1, EPOCHS + 1):

    current_lr = optimizer.param_groups[0]["lr"]
    print(f"\nEpoch {epoch}/{EPOCHS}  |  LR = {current_lr:.2e}")
    print("-" * 55)

    # ---- Train --------------------------------------------------------------
    train_loss, train_acc = train_one_epoch(
        model, train_loader, optimizer, criterion, DEVICE, epoch
    )

    # ---- Validate -----------------------------------------------------------
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

    # ---- Step LR scheduler --------------------------------------------------
    scheduler.step()

    # ---- Log metrics --------------------------------------------------------
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["lr"].append(current_lr)

    print(f"  Train Loss : {train_loss:.4f}  |  Train Acc : {train_acc*100:.2f}%")
    print(f"  Val   Loss : {val_loss:.4f}  |  Val   Acc : {val_acc*100:.2f}%")

    # ---- Save best model (highest val accuracy) -----------------------------
    if val_acc > best_val_acc:
        best_val_acc  = val_acc
        best_val_loss = val_loss
        torch.save(model.state_dict(), SAVE_PATH)
        patience_counter = 0
        print(f"  ✓ New best val acc {best_val_acc*100:.2f}% — model saved to '{SAVE_PATH}'")
    else:
        patience_counter += 1
        print(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")

    # ---- Early stopping check -----------------------------------------------
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping triggered at epoch {epoch} (no val improvement for {PATIENCE} epochs).")
        break

print("\n" + "=" * 65)
print(f"Training complete!")
print(f"  Best Val Accuracy : {best_val_acc*100:.2f}%")
print(f"  Best Val Loss     : {best_val_loss:.4f}")
print(f"  Model saved to    : '{SAVE_PATH}'")
print("=" * 65)

# ---- Plot training curves ---------------------------------------------------
epochs_ran = len(history["train_loss"])
x          = list(range(1, epochs_ran + 1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
axes[0].plot(x, history["train_loss"], label="Train Loss", marker="o")
axes[0].plot(x, history["val_loss"],   label="Val Loss",   marker="s")
axes[0].set_title("Loss per Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].legend()
axes[0].grid(True)

# Accuracy curve
axes[1].plot(x, [a * 100 for a in history["train_acc"]], label="Train Acc", marker="o")
axes[1].plot(x, [a * 100 for a in history["val_acc"]],   label="Val Acc",   marker="s")
axes[1].axhline(y=90, color="red", linestyle="--", linewidth=0.8, label="90% target")
axes[1].set_title("Accuracy per Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(True)

plt.suptitle("EfficientNet-B4 Deepfake Detection Training", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Training curves saved to 'training_curves.png'")


# =============================================================================
# CELL 10 — Evaluate on Validation Set, Print Classification Report
# =============================================================================

# ---- Reload the best saved weights -----------------------------------------
print(f"Loading best weights from '{SAVE_PATH}' …")
model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
model.eval()
print("Weights loaded.\n")

# ---- Collect all predictions ------------------------------------------------
all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(DEVICE, non_blocking=True)
        logits = model(images)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

# ---- Overall accuracy -------------------------------------------------------
overall_acc = (all_preds == all_labels).mean()
print(f"Overall Validation Accuracy : {overall_acc * 100:.2f}%")
print()

# ---- Per-class classification report ---------------------------------------
# target_names: index 0 → FAKE, index 1 → REAL
report = classification_report(
    all_labels,
    all_preds,
    target_names=["FAKE (0)", "REAL (1)"],
    digits=4,
)
print("Classification Report:")
print("-" * 55)
print(report)

# ---- Confusion matrix -------------------------------------------------------
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix (rows=true, cols=predicted):")
print(f"              Pred FAKE   Pred REAL")
print(f"  True FAKE     {cm[0,0]:>6}      {cm[0,1]:>6}")
print(f"  True REAL     {cm[1,0]:>6}      {cm[1,1]:>6}")

# ---- Plot confusion matrix --------------------------------------------------
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
im = ax_cm.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax_cm)
tick_marks = [0, 1]
ax_cm.set_xticks(tick_marks)
ax_cm.set_yticks(tick_marks)
ax_cm.set_xticklabels(["FAKE (0)", "REAL (1)"])
ax_cm.set_yticklabels(["FAKE (0)", "REAL (1)"])

# Annotate cells
thresh = cm.max() / 2.0
for i in range(2):
    for j in range(2):
        ax_cm.text(
            j, i, f"{cm[i, j]:,}",
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=14,
        )

ax_cm.set_xlabel("Predicted Label")
ax_cm.set_ylabel("True Label")
ax_cm.set_title("Validation Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("Confusion matrix saved to 'confusion_matrix.png'")

# ---- Per-class accuracy ------------------------------------------------------
fake_mask       = all_labels == 0
real_mask       = all_labels == 1
fake_acc        = (all_preds[fake_mask] == 0).mean() * 100
real_acc        = (all_preds[real_mask] == 1).mean() * 100
print(f"\nPer-class Accuracy:")
print(f"  FAKE (label 0) : {fake_acc:.2f}%")
print(f"  REAL (label 1) : {real_acc:.2f}%")


# =============================================================================
# CELL 11 — Download Saved Model to Local Machine
# =============================================================================

print("Files available in the current working directory:")
for fname in sorted(os.listdir(".")):
    size_mb = os.path.getsize(fname) / 1e6 if os.path.isfile(fname) else 0
    if os.path.isfile(fname):
        print(f"  {fname:<40} {size_mb:.2f} MB")

print(f"\nDownloading '{SAVE_PATH}' to your local machine …")
files.download(SAVE_PATH)

print(f"\nDownloading 'training_curves.png' …")
files.download("training_curves.png")

print(f"\nDownloading 'confusion_matrix.png' …")
files.download("confusion_matrix.png")

print("\n✓ All downloads initiated.")
print("Check your browser's default Downloads folder.")
print()
print("=" * 65)
print("DONE! Summary:")
print(f"  Model file       : {SAVE_PATH}")
print(f"  Val Accuracy     : {overall_acc * 100:.2f}%")
print(f"  Label 0          : FAKE")
print(f"  Label 1          : REAL")
print("=" * 65)

# Google Colab — Deepfake Detection with EfficientNet-B4
### Step-by-step guide for students

---

## What does this notebook do?

It trains an **EfficientNet-B4** deep learning model to tell apart **real** human face images from **AI-generated (deepfake)** face images. It uses ~140,000 face images from HuggingFace, runs on a free T4 GPU inside Google Colab, and saves the trained model so you can download it.

---

## Prerequisites

| Requirement | Details |
|---|---|
| Google account | Free Gmail is fine |
| Internet connection | Needed to download the dataset (~1.5 GB) |
| Time | ~60–90 minutes for full training |
| Skill level | Beginner-friendly |

---

## Step 1 — Open Google Colab

1. Open your browser and go to **[https://colab.research.google.com](https://colab.research.google.com)**
2. Sign in with your Google account if prompted.
3. Click **File → New notebook** (top-left menu).
4. A blank notebook opens with one empty code cell.

---

## Step 2 — Enable the Free T4 GPU

> [!IMPORTANT]
> You MUST do this BEFORE running any code, otherwise training will be extremely slow (CPU-only).

1. In the top menu click **Runtime → Change runtime type**
2. Under **Hardware accelerator**, select **T4 GPU**
3. Click **Save**
4. You will see a small GPU chip icon (🔲) appear in the top-right corner once it connects.

---

## Step 3 — Create the Code Cells

Open the file [`train_deepfake_efficientnet.py`](train_deepfake_efficientnet.py) in any text editor (Notepad, VS Code, etc.).

The script is divided into **11 sections**, each marked with a header like:

```
# =============================================================================
# CELL 1 — Install Dependencies
# =============================================================================
```

For each section:
1. Click the **+ Code** button in Colab to add a new code cell.
2. Copy **everything between** the two `# ===…===` header lines (including the code below the header) and paste it into that cell.
3. Repeat for all 11 cells.

> [!TIP]
> You can also rename each cell by adding a comment at the top — Colab uses the first comment as a section label in the table of contents on the left sidebar.

---

## Step 4 — Run the Cells in Order

Click the ▶ (play) button on the **left side of each cell**, starting from Cell 1.

**Wait for each cell to finish before running the next one.** A cell is done when:
- The ▶ button stops spinning, and
- You see output printed below the cell.

| Cell | What it does | Expected wait time |
|---|---|---|
| **Cell 1** | Installs `timm`, `datasets`, `transformers` | ~30 seconds |
| **Cell 2** | Imports all Python libraries | ~5 seconds |
| **Cell 3** | Sets hyperparameters and detects GPU | ~2 seconds |
| **Cell 4** | Downloads the 140k face dataset from HuggingFace | **2–5 minutes** |
| **Cell 5** | Defines the Dataset class + image augmentations | ~2 seconds |
| **Cell 6** | Creates DataLoaders (batching system) | ~2 seconds |
| **Cell 7** | Downloads pretrained EfficientNet-B4 weights | ~30 seconds |
| **Cell 8** | Defines train/validate functions | ~2 seconds |
| **Cell 9** | **Runs training for up to 10 epochs** | **45–75 minutes** |
| **Cell 10** | Evaluates accuracy, prints classification report | ~2 minutes |
| **Cell 11** | Downloads model + plots to your PC | ~10 seconds |

---

## Step 5 — Watch the Training Progress

During **Cell 9**, you will see output like this every epoch:

```
Epoch 1/10  |  LR = 1.00e-04
-------------------------------------------------------
  Epoch  1 | Batch  50/3500 | Loss: 0.4821 | Acc: 0.7812
  Epoch  1 | Batch 100/3500 | Loss: 0.4123 | Acc: 0.8125
  ...
  Train Loss : 0.3201  |  Train Acc : 88.45%
  Val   Loss : 0.2890  |  Val   Acc : 91.23%
  ✓ New best val acc 91.23% — model saved to 'efficientnet_deepfake.pt'
```

- The model automatically saves whenever the validation accuracy improves.
- Training stops early if accuracy doesn't improve for **3 consecutive epochs** (saves time).

---

## Step 6 — Check Your Results (Cell 10)

After training, Cell 10 prints a **classification report** like:

```
               precision    recall  f1-score   support

   FAKE (0)      0.9341    0.9278    0.9309     14012
   REAL (1)      0.9284    0.9347    0.9315     13988

   accuracy                          0.9312     28000
```

And a **confusion matrix** showing how many images were correctly/incorrectly classified.

> [!NOTE]
> - **Label 0 = FAKE** (AI-generated / deepfake face)
> - **Label 1 = REAL** (genuine human face photo)

---

## Step 7 — Download Your Model (Cell 11)

Cell 11 automatically triggers browser downloads for:

| File | Description |
|---|---|
| `efficientnet_deepfake.pt` | Trained model weights (PyTorch state dict) |
| `training_curves.png` | Loss & accuracy plots across epochs |
| `confusion_matrix.png` | Confusion matrix visualization |

> [!TIP]
> If the download dialog doesn't appear, check your browser's pop-up blocker settings. Allow pop-ups from `colab.research.google.com`.

Files will appear in your browser's default **Downloads** folder.

---

## How to Use the Saved Model Later

```python
import torch
import timm

# Recreate the model architecture
model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=2)

# Load your trained weights
model.load_state_dict(torch.load('efficientnet_deepfake.pt', map_location='cpu'))
model.eval()

# Labels:  0 = FAKE,   1 = REAL
LABEL_NAMES = {0: "FAKE", 1: "REAL"}
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `CUDA out of memory` | Reduce `BATCH_SIZE` from 32 to 16 in Cell 3 |
| Dataset download stuck | Re-run Cell 4; HuggingFace servers can be slow |
| `ModuleNotFoundError` | Make sure Cell 1 ran successfully; re-run it |
| Session disconnected | Colab sessions expire after ~12 hours of idle; if disconnected, re-run all cells from the top |
| Accuracy below 85% | Try running for more epochs by increasing `EPOCHS = 15` in Cell 3 |
| Download didn't start | Click **Runtime → Run all** on Cell 11 manually after training |

> [!WARNING]
> Google Colab free tier has a **GPU usage limit** (~4–6 hours per day). If you get a "GPU limit reached" message, wait a few hours or use a Colab Pro subscription.

---

## Expected Performance

| Metric | Expected Value |
|---|---|
| Validation Accuracy | **> 90%** |
| Training time (T4 GPU, 10 epochs) | ~60–90 minutes |
| Model file size | ~70 MB |
| Dataset size | ~140,000 images |

---

## File Structure After Training

```
Google Colab working directory (/content/)
├── efficientnet_deepfake.pt    ← trained model weights
├── training_curves.png         ← loss & accuracy curves
└── confusion_matrix.png        ← confusion matrix plot
```

---

## Questions?

If you run into any issues not covered here, check:
- [HuggingFace Datasets docs](https://huggingface.co/docs/datasets)
- [timm model zoo](https://huggingface.co/docs/timm)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)

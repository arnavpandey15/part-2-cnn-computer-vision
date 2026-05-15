# Part 2 – Computer Vision Problem Formulation and CNN Prototype

> **Course**: Applied Neural Networks, Computer Vision, NLP, and AI Solution Design
> **Dataset**: Vehicle Surface Defect Detection (480 images · 4 classes · 96×96 RGB)
> **Framework**: TensorFlow 2 / Keras · scikit-learn · Pillow · Matplotlib

---

## Repository Structure

```
part-2-cnn-computer-vision/
├── README.md
├── notebook.ipynb                         ← All 7 tasks (runnable end-to-end)
├── requirements.txt
├── images/                                ← Dataset (one subfolder per class)
│   ├── dent/        (120 images)
│   ├── normal/      (120 images)
│   ├── scratch/     (120 images)
│   └── stain/       (120 images)
├── results/
│   ├── accuracy_loss_curves.png           ← Training & validation curves (all configs)
│   └── confusion_matrix.png              ← Confusion matrices (all configs)
└── sample_predictions/
    └── prediction_outputs.png             ← 16 test predictions with labels
```

---

## Task 1 – Problem Identification

### Selected Problem Type: Multi-Class Image Classification

Each image in this dataset shows a vehicle surface from exactly one of four conditions. The goal is to assign a single label to the whole image — making this a **closed-set, single-label multi-class classification** problem.

| Problem Type | Chosen? | Reason |
|---|---|---|
| ✅ Image Classification | **YES** | One label per image; no spatial localisation required |
| ❌ Object Detection | No | No bounding boxes needed — defect fills the frame |
| ❌ Semantic Segmentation | No | No pixel-level masks required |
| ❌ Instance Segmentation | No | No separate defect instance distinction needed |

**Problem statement**: *Given a 64×64 RGB image of a vehicle surface, predict whether the surface shows a dent, a scratch, a stain, or is normal (undamaged).*

---

## Task 2 – Dataset Exploration

| Property | Value |
|---|---|
| Total images | 480 |
| Number of classes | 4 — dent, normal, scratch, stain |
| Images per class | 120 (perfectly balanced) |
| Original image size | 96×96 pixels |
| Colour mode | RGB (3 channels) |
| File format | PNG |
| Class imbalance | None — 25% per class |

**Visual characteristics**:
- `dent` — localised deformation creating shadow gradients and curved surface breaks
- `normal` — uniform, flat, unmarked surface texture
- `scratch` — thin linear marks; high-frequency edge signals
- `stain` — discolouration blobs with distinct hue shift from base colour

---

## Task 3 – Image Preprocessing

| Step | Action | Detail |
|---|---|---|
| 1 | Resize | 96×96 → 64×64 pixels (uniform input, faster training) |
| 2 | Normalise | Divide by 255 → pixel values in [0.0, 1.0] |
| 3 | Split | Stratified 70/15/15 → 336 train / 72 val / 72 test |
| 4 | Augment | RandomFlip (horizontal) + RandomRotation (±10°) inside model |

Augmentation is applied **inside the Keras model as a layer**, meaning it is active only during training and automatically disabled at inference — no data leakage.

---

## Task 4 – CNN Model Architecture

```
Input (64×64×3)
    ↓
[Augmentation — RandomFlip + RandomRotation]
    ↓
Conv2D(32, 3×3, padding=same) → BatchNorm → ReLU → MaxPool(2×2)    Block 1
    ↓
Conv2D(64, 3×3, padding=same) → BatchNorm → ReLU → MaxPool(2×2)    Block 2
    ↓
Conv2D(128, 3×3, padding=same) → BatchNorm → ReLU → MaxPool(2×2)   Block 3
    ↓
Flatten
    ↓
Dense(256, ReLU) → Dropout(0.5)
    ↓
Dense(4, Softmax)    →    [P(dent), P(normal), P(scratch), P(stain)]
```

| Layer | Output shape | Role |
|---|---|---|
| Input | 64×64×3 | Raw normalised RGB image |
| Conv_1 (32 filters) | 64×64×32 | Low-level features: edges, colour gradients |
| MaxPool_1 | 32×32×32 | Spatial compression; translation invariance |
| Conv_2 (64 filters) | 32×32×64 | Mid-level: textures, scratch patterns |
| MaxPool_2 | 16×16×64 | Further compression |
| Conv_3 (128 filters) | 16×16×128 | High-level: complex defect shapes |
| MaxPool_3 | 8×8×128 | Final spatial compression |
| **Flatten** | 8192 | Converts 3D feature maps → 1D vector |
| Dense_1 (256) | 256 | Learns defect class representations |
| Dropout (0.5) | 256 | Prevents overfitting |
| Output (4, Softmax) | 4 | Class probability distribution |

**Training config**: Adam (lr=0.001) · Binary Cross-Entropy · EarlyStopping (patience=10) · ReduceLROnPlateau

---

## Task 5 – Training and Evaluation Results

### Baseline Model

| Metric | Value |
|---|---|
| Training Accuracy | 95.24% |
| Validation Accuracy | 87.50% |
| Training Loss | 0.1435 |
| Validation Loss | 0.2549 |
| **Test Accuracy** | **87.50%** |
| **Macro F1-Score** | **0.8723** |
| **ROC-AUC (OvR macro)** | **0.9982** |

### Hyperparameter Experiment Comparison

| Model | Filters | LR | Batch | BN | Train Acc | Val Acc | Test Acc | F1 | AUC |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline** | 32/64/128 | 0.001 | 32 | ✅ | 95.2% | 87.5% | 87.5% | 0.872 | **0.9982** |
| Exp1 – Deeper (4 blocks) | 32/64/128/256 | 0.001 | 32 | ✅ | 91.1% | 29.2% | 26.4% | 0.165 | 0.621 |
| **Exp2 – No BatchNorm** ★ | 32/64/128 | 0.001 | 32 | ❌ | 92.6% | **95.8%** | **93.1%** | **0.929** | 0.9936 |
| Exp3 – High LR + Large batch | 32/64/128 | 0.01 | 64 | ✅ | 58.3% | 25.0% | 29.2% | 0.171 | 0.597 |

### Per-Class Results (Best Model: Exp2 – No BatchNorm)

| Class | Precision | Recall | F1 | Observation |
|---|---|---|---|---|
| dent | 0.86 | 1.00 | 0.92 | Perfect recall — no missed dents |
| normal | 0.94 | 0.94 | 0.94 | Reliable; minimal confusion |
| scratch | 0.93 | 0.78 | 0.85 | Hardest class — edge patterns overlap with dents |
| **stain** | **1.00** | **1.00** | **1.00** | Perfect — distinct colour signature |
| **macro avg** | **0.93** | **0.93** | **0.93** | — |

See `results/` and `sample_predictions/` for all visualisations.

---

## Task 6 – CNN Concept Explanation

### What is convolution?

Convolution is the core operation of a CNN. A small learnable filter (e.g. 3×3 pixels) slides across the input image and computes the dot product between its weights and each local patch of pixels. This produces a **feature map** — a 2D response showing where in the image that particular pattern (edge, curve, colour boundary) was detected. By stacking many filters, each layer learns to detect different visual patterns simultaneously.

### Why is pooling used?

MaxPooling takes a small region (e.g. 2×2) and keeps only its maximum value. This serves two purposes:

1. **Spatial compression** — halves the height and width, reducing computation and memory.
2. **Translation invariance** — if a feature shifts slightly within the pooling window, the output remains the same. A scratch detected slightly left or right still produces the same pooled activation.

Without pooling, even a small image at depth would generate enormous feature maps, making dense layers prohibitively large.

### Why is ReLU commonly used in CNNs?

ReLU (Rectified Linear Unit) is defined as `f(x) = max(0, x)`. It is preferred because:

- **No vanishing gradient**: Unlike sigmoid or tanh, ReLU's gradient is always 1 for positive values, allowing clean gradient flow through many layers during backpropagation.
- **Sparsity**: Negative activations are zeroed out, creating sparse representations that are computationally efficient and often more discriminative.
- **Speed**: It requires a single comparison operation — far cheaper than computing exponentials (sigmoid/tanh).

### Why are CNNs better than regular feed-forward networks for image data?

| Issue | Feed-Forward Network | CNN |
|---|---|---|
| Parameters | A 64×64×3 input → Dense(256) requires 3,145,728 weights just for layer 1 | Conv(32, 3×3) requires only 896 weights — shared across all positions |
| Spatial structure | Flattens the image — destroys pixel neighbourhood information | Processes spatially — nearby pixels stay related |
| Translation invariance | None — a scratch at position (10,15) vs (20,30) appears unrelated | MaxPooling provides local invariance |
| Depth scalability | Deep MLPs suffer from vanishing gradients and over-parameterisation | Skip connections, BN, and ReLU enable very deep CNNs |

CNNs exploit the **local connectivity** and **spatial hierarchy** of image data — edges form textures, textures form shapes, shapes form objects — in a way that fully-connected networks fundamentally cannot.

---

## Task 7 – Business Use Case Mapping

### Domain: Manufacturing — Automated Visual Quality Control

**Problem**: In automotive manufacturing, every vehicle body panel must be inspected for surface defects (dents, scratches, stains) before painting and assembly. Manual inspection by human workers is:
- Slow: a trained inspector examines one panel per 2–3 minutes
- Inconsistent: fatigue and lighting variation cause missed defects
- Expensive: requires dedicated quality control staff on every line

**How this CNN solution applies**:

A camera mounted on the production line captures images of each panel at fixed intervals. The trained CNN classifies each image in under 50 milliseconds, flagging panels with defects for rework or rejection before they advance to the next stage.

| Stakeholder | Benefit |
|---|---|
| Production line | Throughput increases — no manual inspection bottleneck |
| Quality control | Consistent detection regardless of shift, lighting, or inspector fatigue |
| Cost | Defects caught early cost ~10× less to fix than post-assembly rework |
| Data | Every prediction is logged — enables trend analysis (e.g. machine causing repeated dents) |

**Extensions**:
- Localise defects with object detection (YOLO/Faster R-CNN) to pinpoint exact repair area
- Deploy on edge hardware (NVIDIA Jetson) directly on the production line
- Extend classes to include rust, paint bubbles, welding defects

This approach is already used by companies like BMW, Toyota, and Tesla, where CNN-based inspection systems operate at line speed with >98% defect detection rates.

---

## How to Run

```bash
git clone https://github.com/<your-username>/part-2-cnn-computer-vision
cd part-2-cnn-computer-vision
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

All outputs are saved to `results/` and `sample_predictions/` automatically.

---

## Requirements

```
tensorflow>=2.10
scikit-learn>=1.2
pandas>=1.5
numpy>=1.23
matplotlib>=3.6
Pillow>=9.0
jupyter>=1.0
```
# Image Deblurring Project - Technical Documentation

**Last Updated:** November 11, 2025

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Model Architecture](#model-architecture)
3. [Training Pipeline](#training-pipeline)
4. [Hyperparameters](#hyperparameters)
5. [Loss Functions](#loss-functions)
6. [Tiling & Stitching](#tiling--stitching)
7. [Experiment Tracking](#experiment-tracking)
8. [Deployment](#deployment)

---

## Environment Setup

### PyTorch Installation (Windows + Python 3.10)

```powershell
# Set Python path
$env:PYTHONPATH = "C:\Users\as2491\git\image_deblurring"

# Install CUDA-enabled PyTorch
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

**Key Points:**
- CUDA version works on both GPU and CPU (auto-detects)
- Training requires GPU (RTX 4070: 12GB VRAM)
- Inference works on CPU (Azure Functions)

**Performance:**
- GPU: ~99ms for 2048×2048 image
- CPU: ~2-5s for same image

---

## Model Architecture

### DeblurUNet (U-Net Variant)

**Structure:** 4-level encoder-decoder with skip connections

```python
class DeblurUNet(nn.Module):
    # Encoder: 32 → 64 → 128 → 256 channels
    # Bottleneck: 512 channels
    # Decoder: 256 → 128 → 64 → 32 channels
    # Final: 1×1 conv to 3 channels (NO activation)
```

**Key Design Choices:**

| Choice | Reason |
|--------|--------|
| **No final activation** | Allows flexible output range, better gradients, prevents over-smoothing |
| **GroupNorm (8 groups)** | Stable with small batch sizes (works with batch=8) |
| **Bilinear upsample + conv** | Avoids checkerboard artifacts from TransposeConv |
| **Skip connections** | Preserves fine details from encoder |

**Model Stats:**
- Parameters: 8.6M
- Input: [B, 3, 256, 256] (RGB patches)
- Output: [B, 3, 256, 256] (deblurred, unconstrained range)
- Post-processing: Clamp to [-1, 1] for display

**Why No Activation?**
```python
# Final layer
out = self.final(d1)  # Raw conv output
out = torch.clamp(out, -1, 1)  # Clamp ONLY during inference
return out
```

- **Training:** Unrestricted outputs enable better gradient flow
- **Inference:** Clamping ensures valid pixel values
- **Validation:** Outputs naturally stay in [-1.5, 1.5] due to loss function

---

## Training Pipeline

### Data Loading (GoPro Dataset)

**Optimized Memory Strategy:**

```python
class DeblurDataset:
    def __getitem__(self, idx):
        # Load 1280×720 image
        blur_img = Image.open(blur_path)
        
        # Extract 256×256 patch IMMEDIATELY
        patch = random_crop(blur_img, 256)
        
        # Augment (flips, 90° rotations)
        augmented = apply_augmentation(patch)
        
        # Normalize to [-1, 1]
        return transform(augmented)
```

**Memory Savings:** 94% reduction (921,600 → 65,536 pixels per batch item)

**Augmentation:**
- Horizontal flip (50%)
- Vertical flip (50%)
- 90° rotations (25% each: 0°, 90°, 180°, 270°)
- All synchronized between blur/sharp pairs

### Training Loop

```python
for epoch in range(num_epochs):
    # 1. Train
    for blur, sharp in train_loader:
        outputs = model(blur)
        loss = criterion(outputs, sharp)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # 2. Evaluate
    val_loss, val_psnr, val_ssim = evaluate(model, val_loader)
    
    # 3. Save best checkpoint
    if val_psnr > best_psnr:
        save_checkpoint(f'best_model_epoch_{epoch}_psnr_{val_psnr:.2f}.pth')
    
    # 4. Update learning rate
    scheduler.step()
    
    # 5. Early stopping
    if patience_exceeded:
        break
```

---

## Hyperparameters

### Final Configuration

```python
# Core settings
batch_size = 24              # Max for 12GB VRAM
learning_rate = 2e-4         # AdamW sweet spot
num_epochs = 200             # With early stopping
patience = 50                # Early stopping threshold

# Optimization
optimizer = AdamW(lr=2e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(T_max=200, eta_min=1e-6)
gradient_clip = 1.0          # Prevents exploding gradients

# Loss
alpha = 0.84                 # MS-SSIM weight
beta = 0.16                  # Charbonnier weight
```

### Rationale

**Batch Size (24):**
- GPU Memory: ~10.5GB / 12GB used
- MS-SSIM stability: Needs batch_size ≥ 16
- Trade-off: Larger = more stable, but batch_size=32 causes OOM

**Learning Rate (2e-4):**
- Too high (1e-3): Unstable, oscillations
- Too low (1e-5): Slow convergence (200+ epochs)
- 2e-4: Fast, stable convergence (~50-100 epochs)

**Gradient Clipping (1.0):**
- **What:** Limits L2 norm of all gradients: `||g|| ≤ max_norm`
- **Why:** Prevents exploding gradients in deep U-Net
- **Impact:** Training stability, enables higher LR

**Without clipping:**
```
Epoch 27: train_loss=47.2, val_psnr=0.0  ← Explosion!
Epoch 28: train_loss=NaN  ← Crashed
```

**With clipping:**
```
Epoch 27: grad_norm=1.35 → clipped to 1.0
Training continues smoothly
```

**CosineAnnealingLR:**
- Smooth decay: 2e-4 → 1e-6 over 200 epochs
- No manual tuning needed
- Better than step decay (no sudden drops)

---

## Loss Functions

### Evolution: Charbonnier → Combined Loss

**Initial:** Charbonnier Loss (Smooth L1)

```python
class CharbonnierLoss(nn.Module):
    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff**2 + epsilon**2))
```

**Pros:** Robust to outliers, good PSNR (~24-25 dB)  
**Cons:** Over-smooths edges, poor perceptual quality

**Final:** Combined Loss (MS-SSIM + Charbonnier)

```python
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.84, beta=0.16):
        self.ms_ssim = MS_SSIM(data_range=2.0, channel=3)
        
    def forward(self, pred, target):
        # Clamp to prevent NaN
        pred = torch.clamp(pred, -1, 1)
        target = torch.clamp(target, -1, 1)
        
        # MS-SSIM with fallback
        try:
            ms_ssim_val = self.ms_ssim(pred, target)
            ms_ssim_loss = 1 - torch.clamp(ms_ssim_val, 0, 1)
        except:
            ms_ssim_loss = torch.mean(torch.abs(pred - target))
        
        # Charbonnier
        diff = pred - target
        charbonnier = torch.mean(torch.sqrt(diff**2 + epsilon**2))
        
        return alpha * ms_ssim_loss + beta * charbonnier
```

**Why 84% MS-SSIM / 16% Charbonnier?**
- MS-SSIM: Perceptual quality, structure preservation
- Charbonnier: Pixel accuracy, stable gradients
- 84/16 ratio: Empirically best for edge sharpness

**Error Handling:**
- MS-SSIM can fail with small batches or NaN values
- Fallback to Charbonnier ensures training continues
- Detailed logging tracks failures

**Results:**
- Charbonnier only: PSNR 24.5 dB, blurry edges
- Combined: PSNR 26.8 dB, sharp edges, better SSIM

---

## Tiling & Stitching

### Problem: Training vs Inference Size Mismatch

- **Training:** 256×256 patches (memory efficient)
- **Inference:** Arbitrary sizes (1920×1080, 2048×2048, etc.)

### Solution: Sliding Window with Overlap

**Tiling (Input → Tiles):**

```python
def tile_tensor(img, tile_size=512, overlap=64):
    stride = tile_size - overlap  # 448
    tiles = []
    coords = []
    
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            tile = img[:, :, y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            coords.append((x, y, x+tile_size, y+tile_size))
    
    return tiles, coords
```

**Example:** 1920×1080 image → 12 tiles (4×3 grid)

**Stitching (Tiles → Output):**

```python
def stitch_tiles(tiles, coords, image_shape, overlap=64):
    output = torch.zeros(image_shape)
    weight = torch.zeros(image_shape)
    
    for tile, (x1, y1, x2, y2) in zip(tiles, coords):
        # Create feathering weight map
        tile_weight = create_blend_weight(tile.shape, overlap)
        
        # Weighted accumulation
        output[:, y1:y2, x1:x2] += tile * tile_weight
        weight[:, y1:y2, x1:x2] += tile_weight
    
    return output / weight  # Normalize
```

**Feathering (Linear Blend):**

```
Weight map for tile with 64px overlap:

1.0 ┤         ████████████         Center
0.5 ┤      ██              ██      
0.0 ┤██                        ██  Edges
    └──────────────────────────────
    0    64              448    512

Overlap regions blend linearly between tiles
```

**Why Overlap?**
- Prevents visible seams at tile boundaries
- Reduces CNN edge artifacts
- 64px (12.5%) overlap is optimal

**Usage:**

```python
# Training: Direct patches
patch = dataset[idx]  # 256×256

# Inference: Tiled processing
tiles, coords = tile_tensor(large_image, tile_size=512, overlap=64)
outputs = [model(tile) for tile in tiles]
result = stitch_tiles(outputs, coords, large_image.shape, overlap=64)
```

**Performance:**
- Tile size: 512×512 (larger than training for efficiency)
- Overlap: 64px (balance quality vs compute)
- Device-agnostic: Works on GPU and CPU

---

## Experiment Tracking

### MLflow Integration

**Setup:**

```python
import mlflow

mlflow.start_run()

# Log hyperparameters
mlflow.log_param("batch_size", 24)
mlflow.log_param("learning_rate", 2e-4)

# Training loop
for epoch in range(num_epochs):
    mlflow.log_metric("train_loss", loss, step=epoch)
    mlflow.log_metric("psnr", psnr, step=epoch)
    mlflow.log_metric("ssim", ssim, step=epoch)

# Save model
mlflow.pytorch.log_model(model, "deblur_model", 
                         signature=signature)
mlflow.end_run()
```

**Benefits:**

| Feature | Value |
|---------|-------|
| **Parameter tracking** | All hyperparameters logged automatically |
| **Metric visualization** | Interactive charts for 50+ experiments |
| **Model versioning** | Complete history with Git commit hash |
| **Reproducibility** | Environment snapshot (requirements.txt) |
| **Comparison** | Side-by-side run comparison |

**UI Access:**

```powershell
mlflow ui
# → http://localhost:5000
```

**Model Signature:**

```python
def create_model_signature(model, device):
    example = torch.randn(1, 3, 256, 256).to(device)
    output = model(example)
    return infer_signature(example.cpu().numpy(), 
                          output.cpu().numpy())
```

**Stores:**
- Input shape: [B, 3, 256, 256]
- Output shape: [B, 3, 256, 256]
- Data types: float32

---

## Checkpointing Strategy

### Best-Model-Only Approach

```python
best_psnr = 0.0

for epoch in range(num_epochs):
    val_psnr = evaluate(model, val_loader)
    
    if val_psnr > best_psnr:
        best_psnr = val_psnr
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'psnr': val_psnr,
            'ssim': val_ssim
        }
        
        path = f'checkpoints/best_model_epoch_{epoch}_psnr_{val_psnr:.2f}.pth'
        torch.save(checkpoint, path)
```

**Why Best-Only?**
- Minimal disk usage (~10-20 files vs 200)
- Automatic selection (highest PSNR in filename)
- Prevents overfitting (saves peak performance)

**Example Progression:**

```
best_model_epoch_1_psnr_22.23.pth
best_model_epoch_10_psnr_25.79.pth
best_model_epoch_53_psnr_26.80.pth  ← Final best
```

**Loading:**

```python
checkpoint = torch.load('checkpoints/best_model_epoch_53_psnr_26.80.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## Training Results

### Final Performance

| Metric | Value |
|--------|-------|
| **Best PSNR** | 26.80 dB (epoch 53) |
| **Best SSIM** | 0.77 |
| **Training Time** | 2-8 hours (RTX 4070) |
| **Convergence** | 50-100 epochs |
| **Total Runs** | 50+ experiments |

### Training Curve (Typical)

```
PSNR Progress:

27 ┤                      ──────  ← Plateau at 26.8
26 ┤                 ─────
25 ┤            ─────
24 ┤       ─────
23 ┤  ─────
22 ┼─
   0    50   100   150   200 epochs
   
   Early stopping triggered at epoch ~80-120
```

### Key Lessons

**What Worked:**
- ✅ Combined loss (MS-SSIM + Charbonnier)
- ✅ Gradient clipping (prevents crashes)
- ✅ Patch extraction in Dataset (94% memory savings)
- ✅ Best-model checkpointing
- ✅ Error handling in loss function

**What Failed:**
- ❌ Charbonnier-only loss (poor edges)
- ❌ No gradient clipping (crashes at epoch 27)
- ❌ Batch size 32 (OOM error)
- ❌ Learning rate 1e-3 (unstable)
- ❌ No MS-SSIM error handling (training crashes)

---

## Deployment

### Model Export

```python
# Load best checkpoint
checkpoint = torch.load('checkpoints/best_model_epoch_53_psnr_26.80.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Log to MLflow
signature = create_model_signature(model, device)
mlflow.pytorch.log_model(
    model, 
    artifact_path="deblur_model",
    registered_model_name="deblur_model_v4",
    signature=signature
)
```

### Azure Functions Integration

**Inference Pipeline:**

```python
def run_inference(img_tensor, model):
    # Tiled processing for large images
    tiles, coords = tile_tensor(img_tensor, tile_size=512, overlap=64)
    
    outputs = []
    for tile in tiles:
        with torch.no_grad():
            output = model(tile)
        outputs.append(output)
    
    result = stitch_tiles(outputs, coords, img_tensor.shape, overlap=64)
    return result
```

**Device Compatibility:**
- Training: GPU (CUDA)
- Deployment: CPU (Azure Functions)
- Same PyTorch installation works for both

---

## Quick Reference

### Commands

```powershell
# Training
python src/train.py --train_data data/train --test_data data/val --registered_model_name deblur_v4 --batch_size 24 --num_epochs 200 --learning_rate 0.0002 --patience 50

# MLflow UI
mlflow ui

# Verify GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### File Structure

```
image_deblurring/
├── src/
│   ├── train.py              # Training script
│   ├── model_class.py        # DeblurUNet
│   ├── enhanced_loss.py      # CombinedLoss
│   ├── utils.py              # Tiling, metrics
│   └── data_ingestion.py     # Dataset
├── checkpoints/              # Best models
├── mlruns/                   # MLflow artifacts
└── documentation/
    └── doc.md                # This file
```

### Key Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| **Training** | Train Loss | < 0.10 |
| **Validation** | PSNR | > 26 dB |
| **Validation** | SSIM | > 0.75 |
| **Inference** | Speed (GPU) | < 100ms |
| **Inference** | Speed (CPU) | < 5s |
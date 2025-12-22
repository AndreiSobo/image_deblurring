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

- **Training:** 256×256 pa


tches (memory efficient)
- **Inference:** Arbitrary sizes (1920×1080, 2048×2048, etc.)

### Solution: Sliding Window with Overlap

**Tiling (Input → Tiles):**

```python
def tile_tensor(img, tile_size, overlap=64):
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
- Different PyTorch builds required for each environment

### Azure Functions Deployment

**Successful Deployment - November 17, 2025**

The Azure Function was successfully deployed using Azure Functions Core Tools. The deployment process required specific configuration and dependency management to work within Azure's resource constraints.

#### Deployment Command

```bash
cd function_app/
func azure functionapp publish imageDeblur
```

#### Required Configuration Files

**1. local.settings.json** (Critical for deployment)

This file is required by Azure Functions Core Tools to detect the project type and runtime:

```json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "",
    "FUNCTIONS_WORKER_RUNTIME": "python"
  }
}
```

**Purpose:**
- `FUNCTIONS_WORKER_RUNTIME`: Tells Azure Functions this is a Python project
- Without this file, deployment fails with: `"Worker runtime cannot be 'None'"`
- The file must be in the `function_app/` directory (same level as `function_app.py`)

**2. requirements.txt** (Optimized for Azure)

Azure Functions Consumption Plan has memory limitations during remote builds. The standard PyTorch package (797 MB) causes out-of-memory errors (exit code 137). Solution: Use CPU-only PyTorch builds.

```txt
azure-functions>=1.18.0
Pillow>=12.0.0
numpy>=2.0.0,<3.0.0
scikit-image>=0.24.0
scipy>=1.10.0
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.4.1+cpu
torchvision==0.19.1+cpu
```

**Key Points:**
- **`--extra-index-url`** (not `--index-url`): Adds PyTorch's repository while keeping PyPI for other packages
- **`+cpu` suffix**: Downloads CPU-only builds (~200 MB vs 797 MB)
- **Order matters**: Place `--extra-index-url` after standard packages
- **PyTorch 2.4.1**: Latest stable version compatible with Azure Functions Python 3.10

#### Deployment Issues Encountered

**Issue 1: Worker Runtime Detection**
```
Error: Worker runtime cannot be 'None'
Solution: Create local.settings.json with FUNCTIONS_WORKER_RUNTIME="python"
```

**Issue 2: Out-of-Memory During Build**
```
Error: Exit code 137 while installing torch-2.4.1 (797.1 MB)
Solution: Switch to torch==2.4.1+cpu (~200 MB) using --extra-index-url
```

**Issue 3: Package Repository Configuration**
```
Error: Could not find azure-functions (using --index-url)
Solution: Use --extra-index-url instead to search both PyPI and PyTorch repos
```

#### PyTorch Version Strategy

| Environment | PyTorch | TorchVision | Build Type | Notes |
|-------------|---------|-------------|------------|-------|
| **Training (Windows)** | 2.9.0 | 0.24.0 | CUDA 12.1 | GPU-accelerated training |
| **Azure Functions (Linux)** | 2.4.1 | 0.19.1 | CPU-only | Optimized for deployment size |

**Compatibility Result: ✅ Success**

The model trained with PyTorch 2.9.0 (CUDA) was successfully deployed and runs on PyTorch 2.4.1 (CPU). PyTorch maintains backward compatibility for model state dicts across minor versions within the same major version (2.x).

**Performance:**
- CPU inference: ~2-5 seconds for 2048×2048 images
- Sufficient for Azure Functions use case
- No accuracy degradation observed

#### Deployment Workflow Summary

```bash
# 1. Navigate to function app directory
cd /home/andrei/git/image_deblurring/function_app

# 2. Ensure local.settings.json exists with correct runtime
cat local.settings.json
# Should show: "FUNCTIONS_WORKER_RUNTIME": "python"

# 3. Verify requirements.txt uses CPU-only PyTorch
cat requirements.txt
# Should include: --extra-index-url https://download.pytorch.org/whl/cpu
#                 torch==2.4.1+cpu
#                 torchvision==0.19.1+cpu

# 4. Deploy to Azure
func azure functionapp publish imageDeblur

# 5. Test deployment
curl https://imagedeblur-baajcphucvd2ddha.northeurope-01.azurewebsites.net/api/test
```

**Deployment Output:**
- Remote build: ~2 minutes
- Package size: ~500 MB (with CPU PyTorch)
- Python version: 3.10.4 (auto-detected)
- Function runtime: v4
- Region: North Europe

**Updated November 17, 2025**

---

## Development Workflow & Deployment

**Date:** December 22, 2025

### Architecture Overview

This project uses a **local-first development workflow** with production deployment to Azure.

**Production (Azure):**
```
React App (Azure Static Web App)
    ↓ HTTPS
Azure Function (Python/PyTorch)
https://black-forest-0e6a17503.3.azurestaticapps.net → https://imagedeblur-baajcphucvd2ddha.northeurope-01.azurewebsites.net
```

**Local Development (Recommended):**
```
React App (Vite Dev Server)
    ↓ HTTPS
Azure Function (Production)
http://localhost:3000 → https://imagedeblur-baajcphucvd2ddha.northeurope-01.azurewebsites.net
```

### Project Structure

```
image_deblurring/
├── client/                          # React frontend (Vite + Tailwind)
│   ├── src/App.jsx                  # Main component
│   ├── vite.config.js               # Dev server configuration
│   ├── staticwebapp.config.json     # Azure Static Web App routing
│   ├── dist/                        # Build output (deployed to Azure)
│   └── package.json
│
├── function_app/                    # Azure Function (Python) - Production Backend
│   ├── deblur_func/                 # Image deblurring endpoint
│   ├── src/                         # ML utilities (tiling, stitching)
│   ├── model/                       # PyTorch model
│   ├── function_app.py              # Function app entry point
│   └── requirements.txt             # Python dependencies (CPU PyTorch)
│
└── src/                             # ML training code
    ├── train.py                     # Training script
    ├── model_class.py               # U-Net architecture
    └── utils.py                     # Data processing utilities
```

### Why Local-First Development?

**Current Approach (Deploy Every Change via CI/CD):**
- ❌ Slow feedback loop: 1-3 minutes per change
- ❌ Wastes resources: GitHub Actions minutes, Azure deployments
- ❌ Production breakage: Work-in-progress code goes live
- ❌ Poor testing: Can't experiment freely

**Local-First Approach (Recommended):**
- ✅ Instant feedback: <1 second with hot module replacement (HMR)
- ✅ Safe experimentation: Test freely without affecting production
- ✅ Better quality: Deploy only tested, complete features
- ✅ Cost effective: Fewer CI/CD runs
- ✅ Cleaner Git history: Meaningful commits

### Recommended Workflow

```bash
# 1. Local Development (Daily Work)
cd client
npm run dev              # Starts Vite dev server at localhost:3000
# Make changes → browser auto-refreshes instantly
# Test thoroughly with production Azure Function backend

# 2. Commit When Feature is Complete
git add .
git commit -m "feat: add new feature"
git push                 # Triggers CI/CD → deploys to Azure

# 3. Verify in Production
# Visit: https://black-forest-0e6a17503.3.azurestaticapps.net
```

**Key Point:** Use **localhost:3000 for development**, **Azure Static Web App for production**. The Vite dev server provides instant feedback as you code, then deploy only when features are complete and tested.

### Key Configuration Files

#### `client/vite.config.js`
```javascript
export default defineConfig({
    plugins: [react()],
    base: '/',
    server: {
        port: 3000,      // Local development server
    },
    build: {
        rollupOptions: {
            output: {
                manualChunks: undefined,
            },
        },
    },
    publicDir: 'public',
})
```

**Note:** No proxy configuration needed since the React app calls Azure Functions directly via full URL.

#### `client/public/staticwebapp.config.json`

**✅ CORRECT LOCATION** - This is the only copy you need!

```json
{
    "navigationFallback": {
        "rewrite": "/index.html",           // SPA routing
        "exclude": ["/api/*"]               // CRITICAL: Don't rewrite API calls
    },
    "responseOverrides": {
        "404": { "rewrite": "/index.html", "statusCode": 200 }
    }
}
```

**How Vite Handles This File:**
1. Source: `client/public/staticwebapp.config.json`
2. Build: Vite copies `public/` contents to `client/dist/` during `npm run build`
3. Deploy: GitHub Actions deploys `client/dist/` to Azure
4. Azure reads: `staticwebapp.config.json` from deployed files

**Why `/api/*` exclusion is critical:**
- Without it: `/api/imagedeblur` → rewritten to `/index.html` → returns HTML instead of JSON
- With it: `/api/imagedeblur` → passes through to Azure Function (correct behavior)

**File Organization (December 2025 Cleanup):**
- ❌ Removed: Root-level `staticwebapp.config.json` (deprecated)
- ❌ Removed: `client/staticwebapp.config.json` (wrong location)
- ❌ Removed: `static/` directory (legacy pre-React site)
- ✅ Keep: `client/public/staticwebapp.config.json` (only correct location)
- ℹ️ Auto-generated: `client/dist/staticwebapp.config.json` (build artifact, do not edit)

### GitHub Actions Workflow

**`.github/workflows/azure-static-web-apps-black-forest-0e6a17503.yml`:**

```yaml
steps:
  - uses: actions/checkout@v3
  
  - name: Setup Node.js
    uses: actions/setup-node@v3
    with:
      node-version: "18"
      cache-dependency-path: client/package-lock.json
  
  - name: Install + Build
    run: |
      cd client
      npm ci              # Reproducible installs
      npm run build       # Vite → client/dist/
  
  - name: Deploy
    uses: Azure/static-web-apps-deploy@v1
    with:
      app_location: "/client/dist"    # Deploy built files
      api_location: ""                # No integrated API
      skip_app_build: true            # Already built above
```

**Key settings:**
- `app_location: "/client/dist"` - Deploy built React app, not source
- `api_location: ""` - Using separate Function App (not integrated)
- `skip_app_build: true` - Build already done in previous step (faster, more control)
- `cache-dependency-path` - Reuse node_modules between runs

### Frontend-Backend Communication

The React app calls Azure Functions **directly** in both development and production:

```jsx
// client/src/App.jsx (line 362)
const response = await fetch('https://imagedeblur-baajcphucvd2ddha.northeurope-01.azurewebsites.net/api/imagedeblur', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({ image: base64Image }),
});
```

**Architecture:**
- No backend proxy needed
- Same endpoint for local development and production
- Azure Function handles CORS properly

### Local Development Setup

```bash
# Install dependencies (first time only)
cd /home/andrei/git/image_deblurring/client
npm install

# Start Vite development server
npm run dev

# App available at: http://localhost:3000
# - Hot Module Replacement (HMR): Changes appear instantly
# - Calls production Azure Function for processing
```

**What Vite Dev Server Provides:**
- ⚡ Lightning-fast hot reload (<1 second)
- 🔥 Instant updates as you edit files
- 🎯 Same behavior as production
- 🚀 Built-in build optimization

**Important:** The Vite dev server is **NOT** a Node.js backend. It's a development tool that serves your React app locally with hot reload capabilities.

### Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Frontend | React 18 + Vite | Fast HMR, modern build tool |
| Styling | Tailwind CSS | Utility-first, small bundle |
| **Frontend** | React 18 + Vite | Fast HMR, modern build tool |
| **Styling** | Tailwind CSS | Utility-first, small bundle |
| **Icons** | Lucide React | Tree-shakeable SVG icons |
| **Backend** | Azure Functions (Python) | Serverless ML inference |
| **ML Framework** | PyTorch (CPU) | Deep learning model |
| **Deployment** | Azure Static Web Apps | Serverless, global CDN |
| **CI/CD** | GitHub Actions | Automated build & deploy |

### Local vs Production Environments

| Aspect | Local Development | Production |
|--------|------------------|------------|
| **Frontend** | http://localhost:3000 | https://black-forest-0e6a17503.3.azurestaticapps.net |
| **Backend** | Azure Function (production endpoint) | Azure Function |
| **Purpose** | Development, testing, experimentation | Live application for users |
| **Deployment** | None (runs on your machine) | GitHub Actions CI/CD |
| **Changes** | Instant with hot reload | 1-3 minutes via CI/CD |
| **When to Use** | All development work | Final verification, user access |

### Development Best Practices

**DO:**
- ✅ Work locally for all development (localhost:3000)
- ✅ Test thoroughly before committing
- ✅ Commit complete features, not work-in-progress
- ✅ Use meaningful commit messages
- ✅ Verify in production after deployment

**DON'T:**
- ❌ Edit code and push immediately to see changes
- ❌ Use production site as development environment
- ❌ Commit untested code "to see if it works"
- ❌ Make multiple small commits for tiny changes

### Common Issues

**1. API returns HTML instead of JSON**
- **Cause:** Missing `/api/*` exclusion in `staticwebapp.config.json`
- **Fix:** Add `"exclude": ["/api/*"]` to `navigationFallback`

**2. Local dev server won't start**
- **Cause:** Port 3000 already in use or dependencies not installed
- **Fix:** Run `npm install` in client folder, or use `npx kill-port 3000`
- **Cause:** Azure doesn't know client-side routes
- **Fix:** `navigationFallback` rewrites all routes to `/index.html`

**4. CORS errors**
- **Cause:** Azure Function not configured for cross-origin
- **Fix:** Add CORS in Azure Portal or function code headers

### Migration Checklist

**Frontend:**
- [x] Vite + React + Tailwind setup
- [x] Convert HTML → JSX components
- [x] Configure `vite.config.js` proxy
- [x] Create `staticwebapp.config.json` with `/api/*` exclusion
- [x] Update API calls to relative URLs

**Deployment:**
- [x] GitHub Actions workflow
- [x] Build step: `npm ci && npm run build`
- [x] Deploy `client/dist/` with `skip_app_build: true`
- [x] Configure CORS in Azure Function

**Verification:**
- [x] React app loads
- [x] API calls work
- [x] Page refresh doesn't 404
- [x] No CORS errors

**Updated December 20, 2025**

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

# Azure Functions Deployment
cd function_app/
func azure functionapp publish imageDeblur
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
├── function_app/             # Azure Functions deployment
│   ├── function_app.py       # Main function app
│   ├── host.json             # Function host config
│   ├── requirements.txt      # Azure dependencies (CPU PyTorch)
│   ├── local.settings.json   # Runtime configuration (required)
│   ├── deblur_func/          # Deblur function blueprint
│   ├── model/                # Deployed model files
│   └── src/                  # Shared utilities
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

### Inference

Inference is done both locally and via the Azure Function
- locally, I use the inference_script.py to output images the command for this is using the models saved by MLflow:
<code>python -m src.inference_script --model_path "/home/andrei/git/image_deblurring/function_app/model/deblurmodelv8.pth" --input_folder ./data/input/ --output_folder ./data/output/ --date "30.11"<code> 

### Azure Function

Functions in imageDeblur:
    imageDeblur - [httpTrigger]
        Invoke url: https://imagedeblur-baajcphucvd2ddha.northeurope-01.azurewebsites.net/api/imagedeblur

    test_function - [httpTrigger]
        Invoke url: https://imagedeblur-baajcphucvd2ddha.northeurope-01.azurewebsites.net/api/test

---

## Troubleshooting: Static Web App + Azure Functions Integration

**Date:** December 14, 2025

This section documents the challenges encountered when integrating an Azure Static Web App with a separately deployed Azure Function App, and the solutions that resolved them.

### Problem Overview

After successfully deploying the Azure Function App using `func azure functionapp publish imageDeblur`, the Static Web App frontend could not communicate with the backend, resulting in errors when attempting to deblur images.

---

### Issue 1: HTTP 405 Method Not Allowed

**Symptom:**
```
❌ Deblurring failed: API Error: 405 . Please try again or use a smaller image.
```

**Root Cause:**

The frontend was calling `/api/imageDeblur` as a **relative URL**, expecting the function to be integrated with the Static Web App. However, the function was deployed as a **separate Azure Function App** with its own domain (`https://imagedeblur-baajcphucvd2ddha.northeurope-01.azurewebsites.net`).

**Incorrect Configuration:**
```javascript
// static/app.js
const API_CONFIG = {
    imageDeblur: '/api/imageDeblur'  // ❌ Relative URL points to Static Web App
};
```

This caused the browser to make requests to:
```
https://black-forest-0e6a17503.3.azurestaticapps.net/api/imageDeblur
                                                       ↑
                                            This endpoint doesn't exist!
```

**Solution:**

Use the **full Azure Function App URL** in the frontend configuration:

```javascript
// static/app.js
const API_CONFIG = {
    imageDeblur: 'https://imagedeblur-baajcphucvd2ddha.northeurope-01.azurewebsites.net/api/imagedeblur'
};
```

**File:** `static/app.js` (line 45)

---

### Issue 2: CORS "Failed to fetch"

**Symptom:**
```
❌ Deblurring failed: Failed to fetch. Please try again or use a smaller image.
```

**Root Cause:**

After fixing the URL, the browser blocked requests due to **Cross-Origin Resource Sharing (CORS)** restrictions. The Static Web App domain (`https://black-forest-0e6a17503.3.azurestaticapps.net`) is different from the Function App domain (`https://imagedeblur-baajcphucvd2ddha.northeurope-01.azurewebsites.net`), triggering CORS protection.

**Why CORS Failed:**

While the function code included CORS headers:
```python
# function_app/deblur_func/__init__.py
headers = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type,Authorization,X-Requested-With"
}
```

Azure Function Apps require **CORS configuration at the platform level** in addition to code-level headers.

**Solution:**

Enable CORS in the Azure Function App using **Azure Portal** or **Azure CLI**.

**Option 1: Azure Portal (Recommended for Beginners)**

1. Navigate to [Azure Portal](https://portal.azure.com)
2. Open your Function App (`imageDeblur`)
3. In the left menu, select **CORS** under **API** section
4. Add allowed origins:
   - For testing: `*` (allows all origins)
   - For production: `https://black-forest-0e6a17503.3.azurestaticapps.net`
5. Click **Save**

**Option 2: Azure CLI**

```bash
# Allow all origins (testing)
az functionapp cors add \
  --name imageDeblur \
  --resource-group <your-resource-group> \
  --allowed-origins "*"

# Or specify exact origin (production)
az functionapp cors add \
  --name imageDeblur \
  --resource-group <your-resource-group> \
  --allowed-origins "https://black-forest-0e6a17503.3.azurestaticapps.net"
```

**Note:** The checkbox "Enable Access-Control-Allow-Credentials" is **not needed** unless your frontend sends authentication cookies or tokens. For simple POST requests with JSON data, leave it unchecked.

---

### Common Misconceptions Clarified

**Misconception 1: Routes Configuration Files**

Initially, we modified `static/routes.json` and `staticwebapp.config.json` thinking they controlled the function routing. However, these files only affect routing **within** the Static Web App itself.

**Reality:**
- `staticwebapp.config.json`: Routes for static files and Static Web App-integrated functions
- When using a **separate Function App**, these configurations are irrelevant
- The frontend must use the full Function App URL

**Misconception 2: `/api/` Prefix Conventions**

When Azure Functions are **integrated** with Static Web Apps (via GitHub Actions `api_location` setting), they automatically become available at `/api/*` routes.

**Two Deployment Architectures:**

| Approach | Function Location | Frontend URL | CORS Needed? |
|----------|------------------|--------------|--------------|
| **Integrated** | Part of Static Web App build | `/api/imageDeblur` (relative) | ❌ No (same domain) |
| **Separate** | Independent Function App | `https://func-app.azurewebsites.net/api/...` (absolute) | ✅ Yes (cross-origin) |

**This project uses:** Separate deployment (via `func azure functionapp publish`)

---

### Architecture Decision Tree

**When building future Static Web App + Azure Functions projects:**

```
┌─────────────────────────────────────────────────────┐
│ Do you need the function to scale independently     │
│ or be reused by multiple frontends?                 │
└────────────────┬────────────────────────────────────┘
                 │
        ┌────────┴─────────┐
        │                  │
       YES                NO
        │                  │
        ▼                  ▼
┌──────────────┐   ┌──────────────┐
│   SEPARATE   │   │  INTEGRATED  │
│ FUNCTION APP │   │ WITH STATIC  │
│              │   │   WEB APP    │
└──────┬───────┘   └──────┬───────┘
       │                  │
       ▼                  ▼
Deploy with:         Deploy with:
├─ func publish      ├─ GitHub Actions
├─ Full URL in JS    │   api_location: "function_app"
├─ Configure CORS    ├─ Relative URL in JS
└─ More flexibility  │   imageDeblur: '/api/imageDeblur'
                     └─ CORS automatic
```

---

### Best Practices for Future Projects

#### 1. **Choose Deployment Strategy Early**

**Separate Function App (This Project):**
```bash
# Deploy function independently
cd function_app/
func azure functionapp publish imageDeblur
```

**Pros:**
- Independent scaling
- Reusable across multiple frontends
- Separate monitoring and logging

**Cons:**
- Requires CORS configuration
- More complex setup
- Two separate deployments

**Integrated with Static Web App:**
```yaml
# .github/workflows/azure-static-web-apps.yml
- name: Build And Deploy
  uses: Azure/static-web-apps-deploy@v1
  with:
    app_location: "/static"
    api_location: "function_app"  # ← Deploys functions automatically
```

**Pros:**
- Single deployment
- No CORS issues
- Simpler configuration

**Cons:**
- Coupled deployments
- Less flexible scaling

#### 2. **API URL Configuration**

Always use environment variables or configuration files for API URLs:

```javascript
// Good: Flexible configuration
const API_CONFIG = {
    imageDeblur: process.env.FUNCTION_URL || 
                 'https://imagedeblur-baajcphucvd2ddha.northeurope-01.azurewebsites.net/api/imagedeblur'
};

// Avoid: Hardcoded relative paths with separate deployments
const API_CONFIG = {
    imageDeblur: '/api/imageDeblur'  // ❌ Only works with integrated functions
};
```

#### 3. **CORS Checklist for Separate Function Apps**

When using a separate Azure Function App:

- [ ] Add CORS configuration in Azure Portal (API → CORS)
- [ ] Include proper allowed origins (not just `*` in production)
- [ ] Verify CORS headers in function code
- [ ] Test preflight OPTIONS requests
- [ ] Check browser console for CORS errors

#### 4. **Testing Strategy**

**Test functions independently first:**
```bash
# Test function directly (bypasses CORS)
curl -X POST https://imagedeblur-baajcphucvd2ddha.northeurope-01.azurewebsites.net/api/imagedeblur \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64-data>"}'
```

**Then test from browser:**
- Check Network tab in Developer Tools
- Look for CORS errors in Console
- Verify response headers include `Access-Control-Allow-Origin`

#### 5. **Documentation**

Always document:
- Deployment method chosen (separate vs integrated)
- Full function URLs (including all endpoints)
- CORS configuration applied
- Any routing configurations in `staticwebapp.config.json`

---

### Quick Diagnostic Guide

**When you see "405 Method Not Allowed":**
1. ✅ Verify the URL in frontend matches deployed function URL
2. ✅ Check browser Network tab for actual request URL
3. ✅ Confirm function accepts the HTTP method (GET/POST/OPTIONS)

**When you see "Failed to fetch" or CORS errors:**
1. ✅ Check if function and frontend are on different domains
2. ✅ Verify CORS is enabled in Azure Portal
3. ✅ Confirm allowed origins include your Static Web App URL
4. ✅ Test function directly with curl (should work)
5. ✅ Check browser console for specific CORS error messages

**When routes.json changes don't work:**
1. ✅ Remember: routes.json only affects Static Web App routing
2. ✅ If using separate Function App, routes.json is irrelevant
3. ✅ Use full function URL instead of relying on rewrites

---

### Resolution Summary

**Final Working Configuration:**

**Frontend (`static/app.js`):**
```javascript
const API_CONFIG = {
    imageDeblur: 'https://imagedeblur-baajcphucvd2ddha.northeurope-01.azurewebsites.net/api/imagedeblur'
};
```

**Azure Function App CORS (Portal):**
```
Allowed Origins:
- https://portal.azure.com
- https://black-forest-0e6a17503.3.azurestaticapps.net
```

**Function Code (`function_app/deblur_func/__init__.py`):**
```python
headers = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type,Authorization,X-Requested-With"
}

if req.method == "OPTIONS":
    return func.HttpResponse(status_code=204, headers=headers)
```

**Result:** ✅ Successfully processes deblurring requests from Static Web App

**Updated December 14, 2025**
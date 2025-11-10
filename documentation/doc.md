## Tiling and Stitching

### Techniques Used:
- **Tiling:** Sliding window with overlap (32 pixels default)
- **Stitching:** Linear feathering (weighted blending with edge ramps)

### Device Compatibility:
The tiling and stitching functions are device-agnostic and work on both GPU and CPU. This is important as both functions will be used in the training process (GPU) and for inference (CPU-based on Azure Function).

- `tile_tensor()` uses tensor slicing and preserves the input device automatically
- `stitch_tiles()` creates output tensors on the same device as input tiles: `device=out_tiles[0].device`
- `create_blend_weight()` creates weight maps that are moved to the appropriate device during stitching

### Linear Feathering Benefits:
- Reduces edge artifacts from CNN processing
- Smooth transitions between tiles (no visible seams)
- Better handling of U-Net's limited receptive field at tile edges
- 10-20% better visual quality compared to uniform averaging

## Ingestion Pipeline

### Data Structure:
The input for training is composed of pairs of sharp-blurry images from the GoPro dataset. The `DeblurDataset` class loads these pairs and `random_patch_pair()` extracts random 256×256 patches.

### Training Patch Extraction:
- **One patch per image per epoch** - `random_patch_pair()` is called ONCE per image
- **Random location each epoch** - Different patch extracted every epoch for data augmentation
- **Synchronized patches** - Blur and sharp patches extracted from same location
- **Batch composition** - Batch size of 8-16 means 8-16 images → 8-16 random patches

### On-The-Fly Data Augmentation

The training pipeline implements **geometric augmentation** applied randomly to each extracted patch, providing additional data diversity without disk storage overhead:

**Augmentation Techniques:**

**1. Horizontal Flip (50% probability)**
```python
if random.random() > 0.5:
    blur_patch = torch.flip(blur_patch, dims=[2])   # Width axis
    sharp_patch = torch.flip(sharp_patch, dims=[2])
```
- Mirrors image left-to-right
- Doubles effective dataset size
- Safe for motion blur (preserves blur characteristics)

**2. Vertical Flip (50% probability)**
```python
if random.random() > 0.5:
    blur_patch = torch.flip(blur_patch, dims=[1])   # Height axis
    sharp_patch = torch.flip(sharp_patch, dims=[1])
```
- Mirrors image top-to-bottom
- Further increases dataset diversity
- Combined with horizontal flip: 4x orientation variations

**3. 90° Rotations (25% probability each: 0°, 90°, 180°, 270°)**
```python
k = random.randint(0, 3)  # 0=no rotation, 1=90°, 2=180°, 3=270°
if k > 0:
    blur_patch = torch.rot90(blur_patch, k=k, dims=[1, 2])
    sharp_patch = torch.rot90(sharp_patch, k=k, dims=[1, 2])
```
- Rotates in 90° increments
- Helps model learn rotation invariance
- No interpolation artifacts (integer rotations only)

**Key Properties:**
- **Synchronized augmentation**: Blur and sharp patches receive identical transformations
- **Zero disk overhead**: Augmentation happens in memory during training
- **Infinite variations**: Different random augmentation each epoch
- **Preserves blur characteristics**: Only geometric transforms (no scaling/color jittering)
- **Effective dataset multiplier**: ~4-8x effective dataset size

**Benefits:**
- Reduces overfitting by increasing training data diversity
- Improves model generalization to different image orientations
- No preprocessing required (augmentation integrated into training loop)
- Complements random patch extraction for maximum data augmentation

**Training Loop Pattern:**
```python
for epoch in range(num_epochs):
    for batch in dataloader:  # batch has 8-16 image pairs
        blur_patches = []
        sharp_patches = []
        
        # Call random_patch_pair() ONCE per image in batch
        for blur_img, sharp_img in batch:
            b_patch, s_patch = random_patch_pair(blur_img, sharp_img, 256)
            blur_patches.append(b_patch)
            sharp_patches.append(s_patch)
        
        # Stack into batch tensors
        blur_batch = torch.stack(blur_patches)   # [8-16, 3, 256, 256]
        sharp_batch = torch.stack(sharp_patches)
        
        # Train on batch
        output = model(blur_batch)
        loss = criterion(output, sharp_batch)
```

### Summary:
- Batch size = 8-16 images
- One random 256×256 patch extracted from each image
- Result: 8-16 patches per training batch
- Different random locations each epoch

## Training Configuration

### Optimizer: AdamW
The project uses **AdamW** (Adam with decoupled weight decay) as the optimizer for the following reasons:

- **Adaptive Learning Rates**: Automatically adjusts per-parameter learning rates, crucial for complex U-Net architecture where different layers (encoder vs decoder, early vs late) may need different learning rates
- **Robust to Stochastic Gradients**: Handles noisy gradients from random patch extraction well due to momentum smoothing
- **Mixed Precision Compatibility**: Works seamlessly with `torch.cuda.amp` for mixed precision training, maintaining stable convergence with fp16
- **Industry Standard**: Proven track record in image restoration tasks - most deblurring papers use Adam/AdamW
- **Better Weight Decay**: AdamW implements decoupled weight decay, which helps prevent overfitting better than standard Adam

**Configuration:**
```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,              # Conservative starting point
    weight_decay=1e-4     # Helps prevent overfitting
)
```

### Loss Function and Metrics

**Training Loss: CharbonnierLoss**
- Custom implementation of Charbonnier loss (smooth L1 variant) 
- More robust to outliers than MSE
- Better gradient properties for image restoration
- Formula: `sqrt((pred - target)^2 + epsilon^2)`
- Used during backpropagation to optimize model weights

**Evaluation Metrics:**
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures pixel-level accuracy, higher is better (typical range: 25-35 dB for deblurring)
- **SSIM (Structural Similarity Index)**: Measures perceptual quality and structural similarity, ranges from 0 to 1 (higher is better)
- Both metrics compare model output against ground truth sharp images
- Computed on validation set to monitor training progress
- Used for model selection and performance reporting

## Model Architecture

### DeblurUNet Design
The project uses a **U-Net architecture** with the following key design choices:

**Network Structure:**
- **Encoder:** 4 downsampling blocks with MaxPooling (32→64→128→256 channels)
- **Bottleneck:** 512 channels at the deepest level
- **Decoder:** 4 upsampling blocks with skip connections (256→128→64→32 channels)
- **Input/Output:** 3-channel RGB images (3→3 mapping)
- **Base channels:** 32 (configurable parameter)

**Key Design Choices:**

**1. No Final Activation Function**
```python
self.final = nn.Conv2d(base_ch, out_ch, kernel_size=1)
# No Sigmoid/Tanh - outputs raw values
```

**Rationale:**
- **Image restoration standard:** Modern deblurring models (DeblurGAN, MPRNet, NAFNet, Restormer) do NOT use final activations
- **Normalization compatibility:** Images are normalized to [-1, 1] range; raw output learns this scale naturally
- **Better gradient flow:** No saturation issues from Tanh/Sigmoid, enabling faster convergence
- **Loss function compatibility:** CharbonnierLoss works best with unbounded predictions
- **Superior training dynamics:** Model learns proper output scale without artificial constraints

**When final activations ARE used:**
- **Sigmoid [0, 1]:** Classification tasks, binary masks, probabilities
- **Tanh [-1, 1]:** When explicit range constraint is critical (trades gradient flow for guarantees)

**2. GroupNorm Instead of BatchNorm**
```python
self.norm1 = nn.GroupNorm(8, out_ch)  # 8 groups
```

**Advantages:**
- **Small batch size robustness:** Works well with batch_size=4-8 (BatchNorm requires 16+)
- **Consistent normalization:** Statistics computed per sample, not across batch
- **Better for image restoration:** Less sensitive to batch composition variations
- **Inference stability:** No train/eval mode discrepancy

**3. Bilinear Upsampling + Conv**
```python
self.up = nn.Sequential(
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
    nn.Conv2d(in_ch, out_ch, 3, padding=1)
)
```

**Advantages over TransposeConv2d:**
- **No checkerboard artifacts:** Bilinear interpolation is smooth
- **More stable training:** Fewer learned parameters in upsampling path
- **Better fine detail preservation:** Separates upsampling from feature learning

**4. Skip Connections with Robustness**
```python
if x.shape[-2:] != skip.shape[-2:]:
    skip = F.interpolate(skip, size=x.shape[-2:], mode='nearest')
x = torch.cat([x, skip], dim=1)
```

**Benefits:**
- **Flexible input sizes:** Handles non-power-of-2 dimensions gracefully
- **Training stability:** Prevents shape mismatch errors during development
- **Gradient flow:** Preserves fine details from encoder to decoder

**Model Complexity:**
- **Parameters:** ~8.6M trainable parameters
- **Memory:** ~780 MB GPU memory for 256×256 RGB batch
- **Inference speed:** ~99ms for 2048×2048 image (with tiling)

## Experiment Tracking with MLflow

### Why MLflow?
The project uses **MLflow** for comprehensive experiment tracking and model management:

**Key Advantages:**
- **Automatic Experiment Tracking**: All hyperparameters, metrics, and models are logged automatically without manual record-keeping
- **Easy Comparison**: Compare multiple training runs side-by-side to identify which configurations work best
- **Model Versioning**: Each model is saved with its exact hyperparameters, preventing confusion about which version performed best
- **Reproducibility**: Tracks code versions, Python environment, and random seeds for full reproducibility
- **Professional Standard**: Industry-standard tool used at companies like Netflix, Microsoft, and Databricks
- **Local Storage**: All data stored locally in `mlruns/` folder - no cloud account or internet required
- **Built-in Visualization**: Free dashboard for viewing training progress and comparing experiments

**What Gets Tracked:**
```python
# Hyperparameters
mlflow.log_param("batch_size", 8)
mlflow.log_param("learning_rate", 1e-4)
mlflow.log_param("num_epochs", 100)
mlflow.log_param("patience", 5)

# Metrics per epoch
mlflow.log_metric("train_loss", train_loss, step=epoch)
mlflow.log_metric("val_loss", val_loss, step=epoch)
mlflow.log_metric("psnr", val_psnr, step=epoch)
mlflow.log_metric("ssim", val_ssim, step=epoch)

# Model artifacts
mlflow.pytorch.log_model(model, artifact_path="deblur_model")
```

### Accessing the MLflow UI

**Launch the dashboard:**
```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser.

**Features Available:**
- **Experiments View**: See all training runs with their metrics and parameters
- **Comparison Mode**: Select multiple runs to compare side-by-side
- **Metric Charts**: Visualize loss, PSNR, and SSIM curves over epochs
- **Model Registry**: Access and download any trained model version
- **Search/Filter**: Find specific runs by metric ranges or parameter values
- **Export Results**: Download data for reports or presentations

**Example Use Cases:**
- Compare PSNR across different batch sizes (8 vs 16 vs 32)
- Identify which learning rate converged fastest
- Track improvement over 50+ experiment iterations
- Present results to professors or in portfolio interviews
- Verify which model achieved the best validation metrics

## implement model checkpoints

- every 10 epochs
- considering the long training required, its a valid contingency method to accomodate for preserving training progress.
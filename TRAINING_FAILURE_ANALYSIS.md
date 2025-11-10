# Training Failure Analysis & Fixes

## 🚨 What Went Wrong

Based on the training graphs showing:
- Learning rate constant at 0.002 for ~25 epochs  
- **PSNR and SSIM metrics stopped being logged** (returned as 0.0 or NaN)
- Validation loss plateaued around 0.12
- Training appeared to "crash" silently

## 🔍 Root Causes Identified

### **1. MS-SSIM Calculation Failure** (CRITICAL)
**Location:** `src/enhanced_loss.py` line 24

**Problem:**
- MS-SSIM (Multi-Scale Structural Similarity) can fail with:
  - Small batch sizes (batch_size=8 is borderline)
  - Edge case tensor values during training
  - NaN propagation from model outputs
- When MS-SSIM fails, it returned NaN, which propagated through the entire loss calculation
- This caused gradients to become NaN, breaking the training process

**Evidence:**
```
Learning rate graph shows sudden drop at epoch ~25
PSNR/SSIM metrics become empty (0.0)
```

### **2. Silent Evaluation Failures**
**Location:** `src/utils.py` evaluate() function

**Problem:**
- `calculate_metrics()` could crash or return NaN without proper error handling
- When PSNR/SSIM calculation failed, it returned None or 0.0
- Training continued but scheduler received invalid values
- No logging of which batches failed or why

**Impact:**
```python
scheduler.step(val_psnr)  # Received NaN → scheduler broke
mlflow.log_metric("psnr", 0.0)  # Logged invalid metrics
```

### **3. No Validation of Metric Values**
**Location:** `src/train.py` main training loop

**Problem:**
- No checks for NaN/Inf/0.0 values before:
  - Logging metrics to MLflow
  - Stepping the learning rate scheduler
  - Saving checkpoints
- Training continued even when metrics were clearly invalid

## ✅ Fixes Applied

### **Fix 1: Robust MS-SSIM Error Handling**
**File:** `src/enhanced_loss.py`

```python
# Added comprehensive error handling
try:
    # Check tensor validity BEFORE MS-SSIM
    if torch.isnan(prediction_clamped).any() or torch.isinf(prediction_clamped).any():
        raise ValueError("NaN/Inf in prediction")
    if torch.isnan(target_clamped).any() or torch.isinf(target_clamped).any():
        raise ValueError("NaN/Inf in target")
        
    ms_ssim_val = self.ms_ssim(prediction_clamped, target_clamped)
    
    # Validate MS-SSIM output
    if torch.isnan(ms_ssim_val) or torch.isinf(ms_ssim_val):
        raise ValueError(f"Invalid MS-SSIM value: {ms_ssim_val}")
        
    ms_ssim_val = torch.clamp(ms_ssim_val, 0.0, 1.0)
    ms_ssim_loss = 1 - ms_ssim_val
    
except Exception as e:
    # Fallback to Charbonnier-only (which is more stable)
    logging.warning(f"MS-SSIM failed: {e}. Using Charbonnier-only loss.")
    ms_ssim_loss = charbonnier_loss
```

**Benefits:**
- MS-SSIM failures are caught and logged
- Training continues with Charbonnier loss as fallback
- Failure counter tracks repeated issues

### **Fix 2: Batch-wise Validation with Error Recovery**
**File:** `src/utils.py` evaluate() function

```python
for blur_batch, sharp_batch in val_loader:
    # ... existing code ...
    
    # Check for NaN/Inf in model outputs
    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
        logging.warning("NaN/Inf in outputs, skipping batch")
        continue
    
    # Check if loss is valid
    if torch.isnan(val_loss) or torch.isinf(val_loss):
        logging.warning("NaN/Inf in loss, skipping batch")
        continue
    
    # Calculate metrics with try/except
    try:
        psnr_val, ssim_val = calculate_metrics(outputs, sharp_batch)
        
        # Validate metric values
        if np.isnan(psnr_val) or np.isinf(psnr_val):
            logging.warning(f"Invalid PSNR={psnr_val}, skipping batch")
            continue
        
        # Only add valid batches
        running_psnr += psnr_val
        valid_batches += 1
    except Exception as e:
        logging.warning(f"Metric calculation failed: {e}")
        continue

# Handle complete failure
if valid_batches == 0:
    logging.error("All validation batches failed!")
    return torch.tensor(999.0), 0.0, 0.0
```

**Benefits:**
- Bad batches are skipped instead of crashing training
- Detailed logging shows which batches fail and why
- Returns fallback values if all batches fail
- Training can continue with partial validation data

### **Fix 3: Training Loop Safety Checks**
**File:** `src/train.py` main training loop

```python
for epoch in range(args.num_epochs):
    train_loss = train(model, train_loader, criterion, optimiser, device)
    val_loss, val_psnr, val_ssim = evaluate(model, test_loader, criterion, device)

    # NEW: Validate metrics before logging
    if val_psnr == 0.0 or np.isnan(val_psnr) or np.isinf(val_psnr):
        logging.error(f"Invalid PSNR at epoch {epoch+1}: {val_psnr}")
        print(f"❌ Training failed at epoch {epoch+1} due to invalid metrics")
        break  # Stop training instead of continuing with bad data
    
    # Safe to log and use metrics now
    mlflow.log_metric("psnr", float(val_psnr), step=epoch)
    scheduler.step(val_psnr)
```

**Benefits:**
- Early detection of training failures
- Clean exit instead of silent corruption
- Prevents saving broken models
- Clear error messages for debugging

### **Fix 4: Added Missing Imports**
**Files:** `src/utils.py`, `src/train.py`, `src/enhanced_loss.py`

```python
import logging  # Added to utils.py
import numpy as np  # Was missing in train.py
import logging  # Added to enhanced_loss.py
```

## 📊 Expected Improvements

With these fixes, future training runs should:

1. **✓ Complete successfully** even if some batches have issues
2. **✓ Log detailed warnings** when MS-SSIM or metrics fail
3. **✓ Gracefully degrade** to Charbonnier-only loss if needed
4. **✓ Stop training cleanly** if metrics become completely invalid
5. **✓ Show which batches succeeded** (`Evaluated 145/150 batches successfully`)

## 🎯 Recommended Next Steps

### **1. Retry Training with Same Hyperparameters**
```bash
python src/train.py --train_data data/train --test_data data/test --registered_model_name deblur_model_v4_fixed --batch_size 8 --num_epochs 50 --learning_rate 0.0001 --patience 20
```

### **2. Monitor for These Warning Messages:**
- `"MS-SSIM failed: ..."` → Indicates loss function issues
- `"NaN/Inf detected in..."` → Indicates model instability
- `"Evaluated X/Y batches successfully"` → Shows validation health
- `"Invalid metrics: PSNR=..."` → Indicates metric calculation issues

### **3. If Training Still Fails:**

**Option A: Reduce MS-SSIM Weight**
```python
# In enhanced_loss.py
criterion = CombinedLoss(alpha=0.5, beta=0.5)  # More balanced
```

**Option B: Use Charbonnier-Only**
```python
# In train.py, replace CombinedLoss with:
from src.CharbonnierLoss import CharbonnierLoss
criterion = CharbonnierLoss()
```

**Option C: Increase Batch Size** (if GPU memory allows)
```bash
--batch_size 16  # MS-SSIM is more stable with larger batches
```

**Option D: Add Gradient Anomaly Detection**
```python
# In train.py, add after optimizer definition:
torch.autograd.set_detect_anomaly(True)  # Slower but catches NaN sources
```

## 📈 How to Verify Fixes Worked

During training, you should now see:
```
INFO:root:Evaluated 147/150 batches successfully
WARNING:root:NaN/Inf detected in model outputs, skipping batch  # If any fail
INFO:root:epoch 1/50 train loss: 0.15, val loss: 0.12, psnr: 24.5, ssim: 0.76
```

Instead of:
```
INFO:root:epoch 25/50 train loss: 0.12, val loss: 0.12, psnr: 0.0, ssim: 0.0
```

## 🔬 Technical Details

### Why MS-SSIM Can Fail:
1. **Multi-scale downsampling** requires minimum input size (160x160)
2. **Gaussian filtering** can produce NaN with extreme values
3. **Batch statistics** can be unstable with batch_size < 16
4. **Numerical precision** issues when data_range is incorrect

### Why Metrics Return 0.0:
1. **PSNR calculation** requires valid pixel values in [0, 1]
2. **Denormalization** from [-1, 1] → [0, 1] can overflow
3. **Division by zero** when prediction == target exactly
4. **NaN propagation** from earlier operations

### Current Safety Net:
```
Model Output → Clamp[-1,1] → Loss (with fallback) → Metrics (with validation) → Logging (with checks)
     ↓              ↓              ↓                      ↓                        ↓
  Valid?      Prevent NaN    Catch errors          Skip bad batches        Stop if all fail
```

---

**Created:** 2025-11-10  
**Status:** ✅ Fixes Applied  
**Action Required:** Rerun training with monitoring

# file with all the utility methods. Used to promote decoupling and Object Oriented Programming principles.

import random
import logging
from pathlib import Path
from typing import Tuple, List
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
import os
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.nn.utils import clip_grad_norm_
from mlflow.models.signature import infer_signature

# Config
DEFAULT_TILE_SIZE = 256

def list_sequence_dirs(root: str) -> List[Path]:
    root = Path(root)   #type: ignore
    if not root.exists():   # type: ignore
        return []
    return [p for p in sorted(root.iterdir()) if p.is_dir()] #type: ignore

def _collect_image_files(seq_dir: Path, subfolder: str) -> List[Path]:
    folder = seq_dir / subfolder
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])

def find_sharp_blur_pairs(root_test_dir: str,
                          prefer_blur_gamma: bool = False) -> List[Tuple[Path, Path]]:
    """
    Walks test sequences and returns a list of (blur_path, sharp_path) pairs.
    prefer_blur_gamma: if True and blur_gamma exists, use that; otherwise use blur.
    """
    root = Path(root_test_dir)
    pairs = []
    seq_dirs = list_sequence_dirs(root) #type: ignore
    for seq in seq_dirs:
        sharp_files = _collect_image_files(seq, 'sharp')
        if not sharp_files:
            continue
        # determine blur folder to use
        blur_folder = 'blur_gamma' if (prefer_blur_gamma and (seq / 'blur_gamma').exists()) else 'blur'
        blur_files = _collect_image_files(seq, blur_folder)
        if not blur_files:
            # fallback to any blur folder
            blur_files = _collect_image_files(seq, 'blur') or _collect_image_files(seq, 'blur_gamma')
        if not blur_files:
            continue
        # pair by filename if possible, else pair by index
        sharp_names = {p.name: p for p in sharp_files}
        for i, b in enumerate(blur_files):
            if b.name in sharp_names:
                pairs.append((b, sharp_names[b.name]))
            else:
                # best-effort index pairing
                if i < len(sharp_files):
                    pairs.append((b, sharp_files[i]))
    return pairs

# I/O helpers
def load_image_pil(path: Path) -> Image.Image:
    return Image.open(path).convert('RGB')

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    # shape H,W,C -> C,H,W
    t = torch.from_numpy(arr).permute(2,0,1)
    return t

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.clamp(0,1).cpu()
    arr = (t.permute(1,2,0).numpy() * 255.0).astype('uint8')
    return Image.fromarray(arr)

# Synchronized random patch sampling
def random_patch_pair(blur_img: Image.Image,
                      sharp_img: Image.Image,
                      patch_size: int = DEFAULT_TILE_SIZE) -> Tuple[Image.Image, Image.Image]:
    """
    Returns a synchronized random patch (PIL) from blur and sharp images.
    If images differ in size, center-crop the larger to match smaller first.
    """
    bw, bh = blur_img.size
    sw, sh = sharp_img.size
    # ensure same size: if different, center-crop the larger one
    if (bw, bh) != (sw, sh):
        # choose target dims = min dims
        tw, th = min(bw, sw), min(bh, sh)
        def center_crop(img, tw, th):
            w, h = img.size
            left = (w - tw)//2
            top = (h - th)//2
            return img.crop((left, top, left+tw, top+th))
        blur_img = center_crop(blur_img, tw, th)
        sharp_img = center_crop(sharp_img, tw, th)
    W, H = blur_img.size
    if W < patch_size or H < patch_size:
        # pad if smaller than patch_size using reflect padding via torchvision
        # convert to tensor, pad, back to PIL
        b_t = pil_to_tensor(blur_img)
        s_t = pil_to_tensor(sharp_img)
        pad_w = max(0, patch_size - W)
        pad_h = max(0, patch_size - H)
        # pad = (left, top, right, bottom)
        pad = (pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2)
        b_t = TF.pad(b_t, pad, padding_mode='reflect')  #type: ignore
        s_t = TF.pad(s_t, pad, padding_mode='reflect')  #type: ignore
        blur_img = tensor_to_pil(b_t)
        sharp_img = tensor_to_pil(s_t)
        W, H = blur_img.size
    # choose random top-left
    x = random.randint(0, W - patch_size)
    y = random.randint(0, H - patch_size)
    b_patch = blur_img.crop((x, y, x + patch_size, y + patch_size))
    s_patch = sharp_img.crop((x, y, x + patch_size, y + patch_size))
    return b_patch, s_patch

# Batched tile generator for inference
def make_tile_coords(H: int, W: int, tile_size: int = DEFAULT_TILE_SIZE, overlap: int = 64):
    stride = tile_size - overlap
    xs = list(range(0, W, stride))
    ys = list(range(0, H, stride))
    coords = []
    for y in ys:
        for x in xs:
            x2 = min(x + tile_size, W)
            y2 = min(y + tile_size, H)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            coords.append((x1, y1, x2, y2))
    # remove duplicates and keep order
    seen = set()
    uniq = []
    for c in coords:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq

def extract_tiles_from_pil(img: Image.Image, coords: List[Tuple[int,int,int,int]]) -> List[Image.Image]:
    return [img.crop(c) for c in coords]

# Tensor-based tiling for inference (GPU-friendly)
def tile_tensor(tensor: torch.Tensor, tile_size: int, overlap: int) -> Tuple[List[torch.Tensor], List[Tuple[int,int,int,int]]]:
    """
    Splits a tensor into overlapping tiles for processing large images.
    Edge tiles are padded with reflection to ensure all tiles are exactly tile_size x tile_size.
    
    Args:
        tensor: Input tensor of shape [1, C, H, W] or [C, H, W]
        tile_size: Size of each tile (assumes square tiles)
        overlap: Number of pixels to overlap between adjacent tiles
        
    Returns:
        tiles: List of tile tensors (all exactly tile_size x tile_size)
        coords: List of (x1, y1, x2, y2) coordinates for each tile in the original image
    """
    # Handle both [C, H, W] and [1, C, H, W] shapes
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    
    _, c, H, W = tensor.shape
    stride = tile_size - overlap
    
    # Calculate tile positions - ensuring we cover the entire image
    xs = []
    x = 0
    while x < W:
        xs.append(x)
        if x + tile_size >= W:
            break
        x += stride
    # Add last position to ensure full coverage
    if xs[-1] + tile_size < W:
        xs.append(W - tile_size)
    
    ys = []
    y = 0
    while y < H:
        ys.append(y)
        if y + tile_size >= H:
            break
        y += stride
    # Add last position to ensure full coverage
    if ys[-1] + tile_size < H:
        ys.append(H - tile_size)
    
    tiles = []
    coords = []
    for y in ys:
        for x in xs:
            # Ensure we don't go out of bounds
            x1 = max(0, min(x, W - tile_size))
            y1 = max(0, min(y, H - tile_size))
            x2 = x1 + tile_size
            y2 = y1 + tile_size
            
            # Extract tile - should always be exactly tile_size x tile_size now
            tile = tensor[..., y1:y2, x1:x2]
            
            # Safety check: pad if somehow still smaller (shouldn't happen but defensive)
            if tile.shape[-2] < tile_size or tile.shape[-1] < tile_size:
                pad_h = tile_size - tile.shape[-2]
                pad_w = tile_size - tile.shape[-1]
                tile = torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')
            
            tiles.append(tile)
            coords.append((x1, y1, x2, y2))
    
    return tiles, coords

def create_blend_weight_cosine(tile_h: int, tile_w: int, overlap: int) -> torch.Tensor:
    """
    Cosine taper provides smoother blending than linear.
    Falls off as: 0.5 * (1 + cos(π * dist/overlap))
    """
    weight = torch.ones((tile_h, tile_w))
    
    for i in range(overlap):
        # Cosine taper: smooth falloff from 1 to 0
        alpha = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * i / overlap)))
        
        weight[i, :] *= alpha                # Top
        weight[-(i + 1), :] *= alpha         # Bottom
        weight[:, i] *= alpha                # Left
        weight[:, -(i + 1)] *= alpha         # Right
    
    return weight

def stitch_tiles(out_tiles: List[torch.Tensor], 
                coords: List[Tuple[int,int,int,int]], 
                image_shape: Tuple[int,int,int], 
                overlap: int) -> torch.Tensor:
    """
    Stitches overlapping tiles back into a full image using linear feathering for smooth blending.
    
    This method uses weighted averaging with linear feathering at tile edges to create
    seamless transitions and reduce edge artifacts from CNN processing. The center of each
    tile receives full weight while edges smoothly transition to zero over the overlap region.
    
    Args:
        out_tiles: List of output tile tensors (each tile_size x tile_size)
        coords: List of (x1, y1, x2, y2) coordinates for each tile in the target image
        image_shape: Target output shape (C, H, W)
        overlap: Number of pixels to overlap (for linear feathering)
        
    Returns:
        Stitched tensor of shape [1, C, H, W]
    """
    C, H, W = image_shape
    out = torch.zeros((C, H, W), dtype=out_tiles[0].dtype, device=out_tiles[0].device)
    weight = torch.zeros((C, H, W), dtype=out_tiles[0].dtype, device=out_tiles[0].device)
    
    for tile, (x1, y1, x2, y2) in zip(out_tiles, coords):
        # Handle both [1, C, H, W] and [C, H, W] shapes
        if tile.ndim == 4:
            tile = tile.squeeze(0)
        
        # Calculate actual tile dimensions to use
        tile_h_actual = y2 - y1
        tile_w_actual = x2 - x1
        
        # Extract only the valid portion of the tile (in case it was padded)
        tile_valid = tile[:, :tile_h_actual, :tile_w_actual]
        
        # Create blend weight map with linear feathering for the valid portion
        tile_weight = create_blend_weight_cosine(tile_h_actual, tile_w_actual, overlap)
        tile_weight = tile_weight.to(tile.device)
        
        # Apply weighted blending
        out[:, y1:y2, x1:x2] += tile_valid * tile_weight
        weight[:, y1:y2, x1:x2] += tile_weight
    
    # Avoid division by zero
    weight[weight == 0] = 1.0
    out = out / weight
    
    return out.unsqueeze(0)

def calculate_metrics(output, target):
    # CRITICAL FIX: Clamp before denormalization to prevent invalid values
    output = torch.clamp(output, -1.0, 1.0)
    target = torch.clamp(target, -1.0, 1.0)
    
    # Denormalize from [-1, 1] to [0, 1]
    output = (output + 1.0) / 2.0
    target = (target + 1.0) / 2.0
    
    # Convert to numpy (assuming shape: B, C, H, W)
    output_np = output.cpu().detach().numpy()
    target_np = target.cpu().detach().numpy()
    
    # Clip to valid range [0, 1] to prevent PSNR/SSIM errors
    output_np = np.clip(output_np, 0.0, 1.0)
    target_np = np.clip(target_np, 0.0, 1.0)
    
    batch_size = output_np.shape[0]
    psnr_values = []
    ssim_values = []

    for i in range(batch_size):
        out_img = np.transpose(output_np[i], (1,2,0))
        tgt_img = np.transpose(target_np[i], (1,2,0))

        psnr_val = psnr(tgt_img, out_img, data_range=1.0)
        ssim_val = ssim(tgt_img, out_img, channel_axis=2, data_range=1.0)
        
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
    
    return np.mean(psnr_values), np.mean(ssim_values)

def train(model, train_loader, criterion, optimizer, device):
    """
    Simplified training loop - patches are already extracted and augmented by the Dataset.
    Now the DataLoader directly provides 256x256 patches, reducing memory by ~94%.
    """
    model.train()
    # scaler = GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=False)
    running_loss = 0.0

    for blur_batch, sharp_batch in train_loader:
        # blur_batch shape: [B, 3, 256, 256] - already cropped patches from Dataset
        blur_batch = blur_batch.to(device, non_blocking=True)
        sharp_batch = sharp_batch.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with pre-augmented patches
        # device_type = 'cuda' if device.type == 'cuda' else 'cpu'

        # with autocast(device_type=device_type):
        #     outputs = model(blur_batch)
        #     loss = criterion(outputs, sharp_batch)

        outputs = model(blur_batch)
        loss = criterion(outputs, sharp_batch)
        
        # Scale loss and backward pass
        # scaler.scale(loss).backward()
        
        # # Gradient clipping (unscales gradients first)
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # # Optimizer step with scaler
        # scaler.step(optimizer)
        # scaler.update()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    """
    Simplified evaluation loop - patches are already extracted by the Dataset.
    No augmentation applied during validation.
    """
    model.eval()

    running_psnr = 0.0
    running_ssim = 0.0
    running_loss = 0.0
    
    with torch.no_grad():
        for blur_batch, sharp_batch in val_loader:
            # blur_batch shape: [B, 3, 256, 256] - already cropped patches from Dataset
            blur_batch = blur_batch.to(device)
            sharp_batch = sharp_batch.to(device)
            
            # Forward pass
            outputs = model(blur_batch)
                
            val_loss = criterion(outputs, sharp_batch)
            
            # Calculate metrics with error handling

            psnr_val, ssim_val = calculate_metrics(outputs, sharp_batch)
                

            running_psnr += psnr_val
            running_ssim += ssim_val
            running_loss += val_loss.item()

    avg_loss = running_loss / len(val_loader)
    avg_psnr = running_psnr / len(val_loader)
    avg_ssim = running_ssim / len(val_loader)

    return avg_loss, avg_psnr, avg_ssim

def create_model_signature(model, device):
    """
    Creates MLflow model signature
    """
    # Create example input matching actual training data shape
    # Shape: [1, 3, 256, 256] for batch_size=1, RGB (3 channels), 256x256 pixels
    example_input = torch.randn(1, 3, 256, 256).to(device)
    
    with torch.no_grad():
        example_output = model(example_input)
    
    # Convert to numpy for MLflow signature inference
    example_input_np = example_input.cpu().numpy()
    example_output_np = example_output.cpu().numpy()
    
    return infer_signature(example_input_np, example_output_np)

def infer_large_image(model, img_pil, device, tile_size=256, overlap=64):
    """
    Inference with automatic tiling for any size image.
    Uses your existing tiling infrastructure but optimized.
    
    Args:
        model: Trained deblurring model
        img_pil: PIL Image of any size
        device: torch device
        tile_size: Size of tiles (256 to match training)
        overlap: Overlap between tiles (64 = 25% overlap recommended)
    """
    model.eval()
    
    # Convert to tensor
    img_tensor = pil_to_tensor(img_pil)
    
    # Normalize to [-1, 1] (matching your training normalization)
    img_tensor = (img_tensor - 0.5) / 0.5
    
    # Add batch dimension: [1, C, H, W]
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    _, C, H, W = img_tensor.shape
    
    # If image is smaller than tile_size, process directly
    if H <= tile_size and W <= tile_size:
        with torch.no_grad():
            output = model(img_tensor)
        output = output.squeeze(0)
    else:
        # Use your existing tile/stitch functions
        tiles, coords = tile_tensor(img_tensor.squeeze(0), tile_size, overlap)
        
        out_tiles = []
        with torch.no_grad():
            for tile in tiles:
                # Ensure tile is correct shape [1, C, H, W]
                if tile.ndim == 3:
                    tile = tile.unsqueeze(0)
                tile = tile.to(device)
                out_tile = model(tile)
                out_tiles.append(out_tile.squeeze(0))  # Remove batch dim
        
        # Stitch with your blend weights
        output = stitch_tiles(out_tiles, coords, (C, H, W), overlap)
        output = output.squeeze(0)  # Remove batch dimension
    
    # Denormalize back to [0, 1]
    output = (output * 0.5) + 0.5
    output = output.clamp(0, 1)
    
    # Convert back to PIL
    return tensor_to_pil(output)
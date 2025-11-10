# file with all the utility methods. Used to promote decoupling and Object Oriented Programming principles.

import random
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
def make_tile_coords(H: int, W: int, tile_size: int = DEFAULT_TILE_SIZE, overlap: int = 32):
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
    
    Args:
        tensor: Input tensor of shape [1, C, H, W] or [C, H, W]
        tile_size: Size of each tile (assumes square tiles)
        overlap: Number of pixels to overlap between adjacent tiles
        
    Returns:
        tiles: List of tile tensors
        coords: List of (x1, y1, x2, y2) coordinates for each tile
    """
    # Handle both [C, H, W] and [1, C, H, W] shapes
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    
    _, c, H, W = tensor.shape
    stride = tile_size - overlap
    xs = list(range(0, W, stride))
    ys = list(range(0, H, stride))
    
    tiles = []
    coords = []
    for y in ys:
        for x in xs:
            x1 = x
            y1 = y
            x2 = min(x1 + tile_size, W)
            y2 = min(y1 + tile_size, H)
            tile = tensor[..., y1:y2, x1:x2]
            tiles.append(tile)
            coords.append((x1, y1, x2, y2))
    
    return tiles, coords

def create_blend_weight(tile_h: int, tile_w: int, overlap: int) -> torch.Tensor:
    """
    Create a 2D weight map with linear feathering on edges for smooth tile blending.
    
    This creates a weight map that is 1.0 in the center and smoothly transitions
    to 0.0 at the edges over the overlap region. This reduces edge artifacts when
    stitching tiles back together.
    
    Args:
        tile_h: Height of the tile
        tile_w: Width of the tile
        overlap: Number of pixels for the feathering transition on each edge
        
    Returns:
        Weight map tensor of shape [tile_h, tile_w] with values in [0, 1]
    """
    weight = torch.ones((tile_h, tile_w))
    
    # Create linear ramp from 0 to 1 over the overlap region
    for i in range(overlap):
        alpha = (i + 1) / (overlap + 1)  # +1 to avoid 0 at edges
        
        # Apply feathering to all four edges
        weight[i, :] *= alpha              # Top edge
        weight[-(i + 1), :] *= alpha       # Bottom edge
        weight[:, i] *= alpha              # Left edge
        weight[:, -(i + 1)] *= alpha       # Right edge
    
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
        out_tiles: List of output tile tensors
        coords: List of (x1, y1, x2, y2) coordinates for each tile
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
        
        # Create blend weight map with linear feathering
        tile_h, tile_w = tile.shape[-2], tile.shape[-1]
        tile_weight = create_blend_weight(tile_h, tile_w, overlap)
        tile_weight = tile_weight.to(tile.device)
        
        # Apply weighted blending
        out[:, y1:y2, x1:x2] += tile * tile_weight
        weight[:, y1:y2, x1:x2] += tile_weight
    
    # Avoid division by zero
    weight[weight == 0] = 1.0
    out = out / weight
    
    return out.unsqueeze(0)

def calculate_metrics(output, target):
    # Denormalize from [-1, 1] to [0, 1]
    output = (output + 1.0) / 2.0
    target = (target + 1.0) / 2.0
    
    # Convert to numpy (assuming shape: B, C, H, W)
    output_np = output.cpu().detach().numpy()
    target_np = target.cpu().detach().numpy()
    
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

def train(model, train_loader, criterion, optimizer, device, accumulation_steps=1):
    model.train()
    scaler = GradScaler('cuda' if device.type == 'cuda' else 'cpu')
    running_loss = 0.0

    for blur_batch, sharp_batch in train_loader:
        # blur_batch shape: [B, 3, H, W], e.g., [8, 3, 720, 1280]
        blur_batch = blur_batch.to(device)
        sharp_batch = sharp_batch.to(device)
        
        optimizer.zero_grad()
        
        # Extract random patches from each image in batch and apply augmentation
        blur_patches = []
        sharp_patches = []
        
        for i in range(blur_batch.shape[0]):
            # Get single image from batch: [3, H, W]
            blur_img = blur_batch[i]
            sharp_img = sharp_batch[i]
            
            # Denormalize from [-1, 1] to [0, 1] for PIL conversion
            blur_img = (blur_img + 1.0) / 2.0
            sharp_img = (sharp_img + 1.0) / 2.0
            
            # Convert to PIL for random_patch_pair
            blur_pil = tensor_to_pil(blur_img)
            sharp_pil = tensor_to_pil(sharp_img)
            
            # Extract random patch (returns PIL Images)
            blur_patch_pil, sharp_patch_pil = random_patch_pair(blur_pil, sharp_pil, patch_size=256)
            
            # Convert back to tensors
            blur_patch = pil_to_tensor(blur_patch_pil)
            sharp_patch = pil_to_tensor(sharp_patch_pil)
            
            # Normalize back to [-1, 1]
            blur_patch = blur_patch * 2.0 - 1.0
            sharp_patch = sharp_patch * 2.0 - 1.0
            
            # Apply on-the-fly augmentation
            # Horizontal flip
            if random.random() > 0.5:
                blur_patch = torch.flip(blur_patch, dims=[2])  # dims=[2] for width in [C, H, W]
                sharp_patch = torch.flip(sharp_patch, dims=[2])
            
            # Vertical flip
            if random.random() > 0.5:
                blur_patch = torch.flip(blur_patch, dims=[1])  # dims=[1] for height in [C, H, W]
                sharp_patch = torch.flip(sharp_patch, dims=[1])
            
            # 90 degree rotations
            k = random.randint(0, 3)
            if k > 0:
                blur_patch = torch.rot90(blur_patch, k=k, dims=[1, 2])  # dims=[1,2] for [C, H, W]
                sharp_patch = torch.rot90(sharp_patch, k=k, dims=[1, 2])
            
            blur_patches.append(blur_patch)
            sharp_patches.append(sharp_patch)
        
        # Stack patches back into batch: [B, 3, 256, 256]
        blur_patch_batch = torch.stack(blur_patches).to(device)
        sharp_patch_batch = torch.stack(sharp_patches).to(device)
        
        # Forward pass with augmented patches
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        with autocast(device_type=device_type):
            outputs = model(blur_patch_batch)
            loss = criterion(outputs, sharp_patch_batch)
        
        # Scale loss and backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping (unscales gradients first)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step with scaler
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
    
    return running_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()

    running_psnr = 0.0
    running_ssim = 0.0
    running_loss = 0.0
    
    with torch.no_grad():
        for blur_batch, sharp_batch in val_loader:
            blur_batch = blur_batch.to(device)
            sharp_batch = sharp_batch.to(device)
            
            # Extract random patches from each image in batch (no augmentation for validation)
            blur_patches = []
            sharp_patches = []
            
            for i in range(blur_batch.shape[0]):
                # Get single image from batch: [3, H, W]
                blur_img = blur_batch[i]
                sharp_img = sharp_batch[i]
                
                # Denormalize from [-1, 1] to [0, 1]
                blur_img = (blur_img + 1.0) / 2.0
                sharp_img = (sharp_img + 1.0) / 2.0
                
                # Convert to PIL
                blur_pil = tensor_to_pil(blur_img)
                sharp_pil = tensor_to_pil(sharp_img)
                
                # Extract random patch
                blur_patch_pil, sharp_patch_pil = random_patch_pair(blur_pil, sharp_pil, patch_size=256)
                
                # Convert back to tensors
                blur_patch = pil_to_tensor(blur_patch_pil)
                sharp_patch = pil_to_tensor(sharp_patch_pil)
                
                # Normalize back to [-1, 1]
                blur_patch = blur_patch * 2.0 - 1.0
                sharp_patch = sharp_patch * 2.0 - 1.0
                
                blur_patches.append(blur_patch)
                sharp_patches.append(sharp_patch)
            
            # Stack patches back into batch
            blur_patch_batch = torch.stack(blur_patches).to(device)
            sharp_patch_batch = torch.stack(sharp_patches).to(device)
            
            # Forward pass
            outputs = model(blur_patch_batch)
            val_loss = criterion(outputs, sharp_patch_batch)
            
            # Calculate metrics
            psnr_val, ssim_val = calculate_metrics(outputs, sharp_patch_batch)
            running_psnr += psnr_val
            running_ssim += ssim_val
            running_loss += val_loss
    
    avg_psnr = running_psnr / len(val_loader)
    avg_ssim = running_ssim / len(val_loader)
    avg_loss = running_loss / len(val_loader)
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
import random
import logging
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.nn.utils import clip_grad_norm_
from mlflow.models.signature import infer_signature


# Config
DEFAULT_TILE_SIZE = 256


def list_sequence_dirs(root: str) -> List[Path]:
    """List all sequence directories in the root directory."""
    root = Path(root) # type: ignore
    if not root.exists(): #type: ignore
        return []
    return [p for p in sorted(root.iterdir()) if p.is_dir()] #type: ignore


def _collect_image_files(seq_dir: Path, subfolder: str) -> List[Path]:
    """Collect all image files from a subfolder."""
    folder = seq_dir / subfolder
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])


def find_sharp_blur_pairs(root_test_dir: str,
                          prefer_blur_gamma: bool = False) -> List[Tuple[Path, Path]]:
    """
    Walk test sequences and return a list of (blur_path, sharp_path) pairs.
    
    Args:
        root_test_dir: Root directory containing sequence folders
        prefer_blur_gamma: If True and blur_gamma exists, use that; otherwise use blur
    """
    root = Path(root_test_dir)
    pairs = []
    seq_dirs = list_sequence_dirs(root)     #type: ignore
    
    for seq in seq_dirs:
        sharp_files = _collect_image_files(seq, 'sharp')
        if not sharp_files:
            continue
            
        # Determine blur folder to use
        blur_folder = 'blur_gamma' if (prefer_blur_gamma and (seq / 'blur_gamma').exists()) else 'blur'
        blur_files = _collect_image_files(seq, blur_folder)
        
        if not blur_files:
            # Fallback to any blur folder
            blur_files = _collect_image_files(seq, 'blur') or _collect_image_files(seq, 'blur_gamma')
        
        if not blur_files:
            continue
            
        # Pair by filename if possible, else pair by index
        sharp_names = {p.name: p for p in sharp_files}
        for i, b in enumerate(blur_files):
            if b.name in sharp_names:
                pairs.append((b, sharp_names[b.name]))
            else:
                # Best-effort index pairing
                if i < len(sharp_files):
                    pairs.append((b, sharp_files[i]))
    
    return pairs


# Image I/O helpers
def load_image_pil(path: Path) -> Image.Image:
    """Load an image as PIL Image."""
    return Image.open(path).convert('RGB')


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to torch tensor [C, H, W] in range [0, 1]."""
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)  # H,W,C -> C,H,W
    return t


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert torch tensor [C, H, W] in range [0, 1] to PIL Image."""
    t = t.clamp(0, 1).cpu()
    arr = (t.permute(1, 2, 0).numpy() * 255.0).astype('uint8')
    return Image.fromarray(arr)


# Tiling and stitching functions
def tile_tensor(tensor: torch.Tensor, tile_size: int, overlap: int) -> Tuple[List[torch.Tensor], List[Tuple[int,int,int,int]]]:
    """
    Split a tensor into overlapping tiles for processing large images.
    
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
    
    # Calculate tile positions
    xs = []
    x = 0
    while x < W:
        xs.append(x)
        if x + tile_size >= W:
            break
        x += stride
    
    # Add last position if not already covered
    if xs[-1] + tile_size < W:
        xs.append(W - tile_size)
    
    ys = []
    y = 0
    while y < H:
        ys.append(y)
        if y + tile_size >= H:
            break
        y += stride
    
    # Add last position if not already covered
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
            
            # Extract tile
            tile = tensor[..., y1:y2, x1:x2]
            
            # Safety check: pad if somehow still smaller
            if tile.shape[-2] < tile_size or tile.shape[-1] < tile_size:
                pad_h = tile_size - tile.shape[-2]
                pad_w = tile_size - tile.shape[-1]
                tile = torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')
            
            tiles.append(tile)
            coords.append((x1, y1, x2, y2))
    
    return tiles, coords


def create_blend_weight_cosine(tile_h: int, tile_w: int, overlap: int) -> torch.Tensor:
    """
    Create cosine blend weights for smooth tile stitching.
    
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
    Stitch overlapping tiles back into a full image using cosine feathering.
    
    Args:
        out_tiles: List of output tile tensors
        coords: List of (x1, y1, x2, y2) coordinates for each tile
        image_shape: Target output shape (C, H, W)
        overlap: Number of pixels to overlap (for feathering)
        
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
        
        # Calculate actual tile dimensions
        tile_h_actual = y2 - y1
        tile_w_actual = x2 - x1
        
        # Extract only the valid portion
        tile_valid = tile[:, :tile_h_actual, :tile_w_actual]
        
        # Create blend weight map
        tile_weight = create_blend_weight_cosine(tile_h_actual, tile_w_actual, overlap)
        tile_weight = tile_weight.to(tile.device)
        
        # Apply weighted blending
        out[:, y1:y2, x1:x2] += tile_valid * tile_weight
        weight[:, y1:y2, x1:x2] += tile_weight
    
    # Avoid division by zero
    weight[weight == 0] = 1.0
    out = out / weight
    
    return out.unsqueeze(0)


# Metric calculation
def calculate_metrics(output: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """
    Calculate PSNR and SSIM metrics between output and target.
    
    Args:
        output: Model output tensor [-1, 1] range, shape [B, C, H, W]
        target: Target tensor [-1, 1] range, shape [B, C, H, W]
        
    Returns:
        (mean_psnr, mean_ssim)
    """
    # Clamp before denormalization to prevent invalid values
    output = torch.clamp(output, -1.0, 1.0)
    target = torch.clamp(target, -1.0, 1.0)
    
    # Denormalize from [-1, 1] to [0, 1]
    output = (output + 1.0) / 2.0
    target = (target + 1.0) / 2.0
    
    # Convert to numpy
    output_np = output.cpu().detach().numpy()
    target_np = target.cpu().detach().numpy()
    
    # Clip to valid range [0, 1]
    output_np = np.clip(output_np, 0.0, 1.0)
    target_np = np.clip(target_np, 0.0, 1.0)
    
    batch_size = output_np.shape[0]
    psnr_values = []
    ssim_values = []
    
    for i in range(batch_size):
        # Convert from [C, H, W] to [H, W, C]
        out_img = np.transpose(output_np[i], (1, 2, 0))
        tgt_img = np.transpose(target_np[i], (1, 2, 0))
        
        try:
            psnr_val = psnr(tgt_img, out_img, data_range=1.0)
            ssim_val = ssim(tgt_img, out_img, channel_axis=2, data_range=1.0)
            
            # Sanity check
            if np.isnan(psnr_val) or np.isinf(psnr_val):
                logging.warning(f"Invalid PSNR value: {psnr_val}, skipping...")
                continue
            if np.isnan(ssim_val) or np.isinf(ssim_val):
                logging.warning(f"Invalid SSIM value: {ssim_val}, skipping...")
                continue
            
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            
        except Exception as e:
            logging.error(f"Error calculating metrics for sample {i}: {e}")
            continue
    
    if len(psnr_values) == 0:
        logging.error("No valid PSNR/SSIM values calculated!")
        return 0.0, 0.0
    
    return np.mean(psnr_values), np.mean(ssim_values) #type: ignore


# Training and evaluation functions
def train(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Returns:
        Average training loss
    """
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    for blur_batch, sharp_batch in train_loader:
        blur_batch = blur_batch.to(device, non_blocking=True)
        sharp_batch = sharp_batch.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        outputs = model(blur_batch)
        loss = criterion(outputs, sharp_batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        
        running_loss += loss.item()
        num_batches += 1
    
    return running_loss / num_batches


def evaluate(model, val_loader, criterion, device):
    """
    Evaluate the model on validation set.
    
    Returns:
        (avg_loss, avg_psnr, avg_ssim)
    """
    model.eval()
    
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for blur_batch, sharp_batch in val_loader:
            blur_batch = blur_batch.to(device)
            sharp_batch = sharp_batch.to(device)
            
            # Forward pass
            outputs = model(blur_batch)
            
            # Calculate loss
            val_loss = criterion(outputs, sharp_batch)
            
            # Calculate metrics
            psnr_val, ssim_val = calculate_metrics(outputs, sharp_batch)
            
            running_loss += val_loss.item()
            running_psnr += psnr_val
            running_ssim += ssim_val
            num_batches += 1
    
    avg_loss = running_loss / num_batches
    avg_psnr = running_psnr / num_batches
    avg_ssim = running_ssim / num_batches
    
    return avg_loss, avg_psnr, avg_ssim


def create_model_signature(model, device):
    """
    Create MLflow model signature for the deblurring model.
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


def infer_large_image(model, img_pil: Image.Image, device, tile_size: int = 256, overlap: int = 64) -> Image.Image:
    """
    Run inference on a large image using tiled processing.
    
    Args:
        model: Trained deblurring model
        img_pil: PIL Image of any size
        device: torch device
        tile_size: Size of tiles (256 to match training)
        overlap: Overlap between tiles (64 = 25% overlap)
    
    Returns:
        Deblurred PIL Image
    """
    model.eval()
    
    # Convert to tensor
    img_tensor = pil_to_tensor(img_pil)
    
    # Normalize to [-1, 1] (matching training normalization)
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
        # Use tiling for large images
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
        
        # Stitch with blend weights
        output = stitch_tiles(out_tiles, coords, (C, H, W), overlap)
        output = output.squeeze(0)  # Remove batch dimension
    
    # Denormalize back to [0, 1]
    output = (output * 0.5) + 0.5
    output = output.clamp(0, 1)
    
    # Convert back to PIL
    return tensor_to_pil(output)
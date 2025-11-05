# file with all the utility methods. Used to promote decoupling and Object Oriented Programming principles.

import random
from pathlib import Path
from typing import Tuple, List, Optional

from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF

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
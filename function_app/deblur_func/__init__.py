# file: __init__.py
import io
import os
import sys
import time
from typing import Tuple

from PIL import Image
import torch
import torchvision.transforms as T

import azure.functions as func  # Azure Functions Python runtime

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import tile_tensor, stitch_tiles

# Module-global model container so it persists across calls while instance is warm
MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TILE_SIZE = 512
TILE_OVERLAP = 64  

def load_model(model_path: str):
    global MODEL
    if MODEL is not None:
        return MODEL
    # Replace with your model class import if needed
    # from model_def import DeblurNet
    # model = DeblurNet(...)
    model = torch.load(model_path, map_location=DEVICE)
    model.eval()
    if DEVICE.type == "cuda":
        model.to(DEVICE)
    MODEL = model
    return MODEL

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    transform = T.Compose([
        T.ToTensor(),  # [0,1]
    ])
    return transform(img).unsqueeze(0)  # 1CHW batch    # type: ignore

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.clamp(0.0, 1.0).cpu().squeeze(0)
    img = T.ToPILImage()(t)
    return img

def run_inference(img_tensor: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    # img_tensor shape [1, C, H, W]
    _, _, H, W = img_tensor.shape
    if max(H, W) <= TILE_SIZE:
        inp = img_tensor.to(DEVICE)
        with torch.no_grad():
            out = model(inp)
        return out.cpu()
    # tiled inference using utils functions
    tiles, coords = tile_tensor(img_tensor, TILE_SIZE, TILE_OVERLAP)
    out_tiles = []
    for t in tiles:
        t = t.to(DEVICE)
        with torch.no_grad():
            out_t = model(t)
        out_tiles.append(out_t.cpu())
    # Convert shape to tuple for type safety
    img_shape = (img_tensor.shape[1], img_tensor.shape[2], img_tensor.shape[3])
    stitched = stitch_tiles(out_tiles, coords, image_shape=img_shape, overlap=TILE_OVERLAP)
    return stitched

def preprocess_image_bytes(body: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(body)).convert("RGB")
    tensor = pil_to_tensor(img)  # [1,C,H,W] float32 [0,1]
    return tensor

def postprocess_tensor_to_bytes(tensor: torch.Tensor, fmt="JPEG") -> bytes:
    pil = tensor_to_pil(tensor)
    buf = io.BytesIO()
    pil.save(buf, format=fmt, quality=95)
    return buf.getvalue()

def main(req: func.HttpRequest) -> func.HttpResponse:
    start = time.time()
    model_path = os.environ.get("MODEL_PATH", "model.pth")
    try:
        load_model(model_path)
    except Exception as e:
        return func.HttpResponse(f"Model load error: {e}", status_code=500)

    try:
        body = req.get_body()
        inp_tensor = preprocess_image_bytes(body)
        out_tensor = run_inference(inp_tensor, MODEL) #type: ignore
        out_bytes = postprocess_tensor_to_bytes(out_tensor)
        elapsed = time.time() - start
        headers = {"Content-Type": "image/jpeg", "X-Inference-Time": f"{elapsed:.3f}s"}
        return func.HttpResponse(body=out_bytes, headers=headers, status_code=200)
    except Exception as e:
        return func.HttpResponse(f"Inference error: {e}", status_code=500)
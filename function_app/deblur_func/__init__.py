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
import logging
import base64
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))

logger.info(f"Current directory: {current_dir}")
logger.info(f"Source directory: {src_dir}")

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    logger.info(f"Added {src_dir} to sys.path")

# Import model and utilities
try:
    from model_class import DeblurUNet
    from utils import tile_tensor, stitch_tiles, infer_large_image
    logger.info("Successfully imported model_class and utils")
except ImportError as e:
    logger.error(f"Failed to import dependencies: {e}")
    logger.error(f"sys.path: {sys.path}")
    raise

bp = func.Blueprint()
logger.info("Blueprint created")

# Module-global model container so it persists across calls while instance is warm
MODEL = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

def load_model():
    global MODEL
    if MODEL is None:  # Fixed: was "is not None"
        try:
            checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'deblurmodelv8.pth')
            
            logging.info(f"Attempting to load model from: {checkpoint_path}")
            
            if not os.path.exists(checkpoint_path):
                logging.error(f"Model file not found at: {checkpoint_path}")
                return None
            
            model = DeblurUNet()

            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(DEVICE)
            model.eval()
            
            MODEL = model
            logging.info(f"Model loaded successfully from: {checkpoint_path}")
        except Exception as e:
            logging.exception(f"Failed to load model: {e}")
            return None
    return MODEL

@bp.route(route="imageDeblur", methods={"GET", "POST", "OPTIONS"}, auth_level=func.AuthLevel.ANONYMOUS)
def imageDeblur(req: func.HttpRequest) -> func.HttpResponse:
    """Deblurring function using the attached PyTorch model"""
    logging.info("imageDeblur function processed a request")

    # CORS headers
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type,Authorization,X-Requested-With"
    }

    if req.method == "OPTIONS":
            return func.HttpResponse(status_code=204, headers=headers)
    
    start_time = time.time()
    
    try:
        # get request body
        req_body = req.get_json()
        if not req_body:
             return func.HttpResponse(
                  json.dumps({"error": "Request body is required"}),
                  status_code=400,
                  headers=headers,
                  mimetype="application/json"
             )
        
        image_b64 = req_body.get('image')

        # validate 
        if not image_b64:
             return func.HttpResponse(
                  json.dumps({
                       "success": False,
                       "error": "Image data is required in this field"
                  }),
                  status_code=400,
                  headers=headers,
                  mimetype="application/json"
             )

        # decode base 64 image
        try:
            image_data = base64.b64decode(image_b64)
            image_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            logging.error(f"Failed to decode image: {str(e)}")
            return func.HttpResponse(
                json.dumps({
                    "success": False,
                    "error": f"Invalid image data: {str(e)}"
                }),
                status_code=400,
                headers=headers,
                mimetype="application/json"
            )

        model = load_model()
        
        generated_image = infer_large_image(model=model, img_pil=image_pil, device=DEVICE)

        # encode to base64
        buffer = io.BytesIO()
        generated_image.save(buffer, format='JPEG', quality=90)
        img_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

        processing_time = start_time - time.time()
        
        logging.info(f"Deblurring completed in {processing_time:x2f}")

        # send image back
        response = {
            "success": True,
            "deblurred_image": f"data:image/jpeg;base64,{img_string}",
            "original_image": f"data:image/jpeg;base64,{image_b64}",
            "processing_time": f"{processing_time:.1f}",
            "image_size": f"{generated_image.size[0]}x{generated_image.size[1]}"
        }
        return func.HttpResponse(
             json.dumps(response),
             status_code=200,
             headers=headers,
             mimetype="application/json"
        )
        

    except Exception as e:
        logging.exception("Error in deblur function: %s", e)
        return func.HttpResponse(
            json.dumps({
                "success": False,
                "error": str(e),
                "message": "Deblurring failed"
            }),
            status_code=500,
            headers=headers,
            mimetype="application/json"
        )
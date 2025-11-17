# file: __init__.py
import io
import os
import sys
import time
import base64
import json
import azure.functions as func
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint immediately (no heavy imports yet)
bp = func.Blueprint()
logger.info("Blueprint created")

# Global variables for lazy loading
MODEL = None
DEVICE = None
_torch = None
_Image = None
_infer_large_image = None
_DeblurUNet = None

def _lazy_imports():
    """Lazy import heavy dependencies only when function is called"""
    global _torch, _Image, _infer_large_image, _DeblurUNet, DEVICE
    
    if _torch is None:
        logger.info("Loading heavy dependencies...")
        
        import torch
        from PIL import Image
        
        # Add src directory to path for imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
        
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
            logger.info(f"Added {src_dir} to sys.path")
        
        from model_class import DeblurUNet
        from utils import infer_large_image
        
        _torch = torch
        _Image = Image
        _DeblurUNet = DeblurUNet
        _infer_large_image = infer_large_image
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Heavy dependencies loaded. Using device: {DEVICE}")

def load_model():
    """Load the model once and cache it"""
    global MODEL
    
    # Ensure dependencies are loaded
    _lazy_imports()
    
    if MODEL is None:
        try:
            checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'deblurmodelv8.pth')
            
            logger.info(f"Attempting to load model from: {checkpoint_path}")
            
            if not os.path.exists(checkpoint_path):
                logger.error(f"Model file not found at: {checkpoint_path}")
                return None
            
            model = _DeblurUNet()   #type: ignore

            checkpoint = _torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)  #type: ignore
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(DEVICE)
            model.eval()
            
            MODEL = model
            logger.info(f"Model loaded successfully from: {checkpoint_path}")
        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            return None
    return MODEL

@bp.route(route="imageDeblur", methods={"GET", "POST", "OPTIONS"}, auth_level=func.AuthLevel.ANONYMOUS)
def imageDeblur(req: func.HttpRequest) -> func.HttpResponse:
    """Deblurring function using the attached PyTorch model"""
    logger.info("imageDeblur function processed a request")
    
    # Load heavy dependencies on first request
    _lazy_imports()

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
            image_pil = _Image.open(io.BytesIO(image_data)).convert("RGB") # type: ignore
        except Exception as e:
            logger.error(f"Failed to decode image: {str(e)}")
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
        
        generated_image = _infer_large_image(model=model, img_pil=image_pil, device=DEVICE) #type: ignore

        # encode to base64
        buffer = io.BytesIO()
        generated_image.save(buffer, format='JPEG', quality=90)
        img_string = base64.b64encode(buffer.getvalue()).decode("utf-8")

        processing_time = time.time() - start_time
        
        logger.info(f"Deblurring completed in {processing_time:.2f}s")

        # send image back
        response = {
            "success": True,
            "deblurred_image": f"data:image/jpeg;base64,{img_string}",
            "original_image": f"data:image/jpeg;base64,{image_b64}",
            "processing_time": f"{processing_time:.1f}s",
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
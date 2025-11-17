import azure.functions as func
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# create Function instance
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

logger.info(f"Python path: {sys.path}")
logger.info(f"Current directory: {current_dir}")

# register the endpoint
try:
    from deblur_func import bp as deblur_bp
    app.register_functions(deblur_bp)
    logger.info("Successfully registered deblur_func blueprint")
except Exception as e:
    logger.exception("Failed to register deblur_func blueprint: %s", e)
    raise  # Re-raise to make the error visible

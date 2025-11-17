import azure.functions as func
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# create Function instance
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

logger.info("Function app initialized")

# Test function - simple HTTP trigger to verify deployment works
@app.route(route="test", methods=["GET"])
def test_function(req: func.HttpRequest) -> func.HttpResponse:
    """Simple test function"""
    logger.info("Test function called")
    return func.HttpResponse("Test function works!", status_code=200)

# Register the blueprint - import happens at module level
# Azure Functions v2 requires this to be at module level for discovery
try:
    from deblur_func import bp as deblur_bp
    app.register_functions(deblur_bp)
    logger.info("Successfully registered deblur_func blueprint")
except Exception as e:
    logger.error(f"Failed to register blueprint: {e}")
    # Don't raise - let the test function still work

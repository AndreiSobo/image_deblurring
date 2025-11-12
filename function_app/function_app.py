import azure.functions as func

# create Function instance
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# register the endpoints
try:
    from deblur_func import bp as deblur_bp
    app.register_functions(deblur_bp)
except Exception as e:
    import logging
    logging.info("Failed to register deblur_func blueprint: %s", e)

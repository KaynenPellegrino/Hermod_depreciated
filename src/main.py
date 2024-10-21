# src/main.py

import logging
from fastapi import FastAPI
from src.routes import api_router
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/main.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def create_app() -> FastAPI:
    """
    Creates and configures the FastAPI application.
    """
    try:
        app = FastAPI(title="Hermod AI Assistant")

        # Initialize components
        config_manager = ConfigurationManager()
        notification_manager = NotificationManager()

        # Include API routers
        app.include_router(api_router)

        # Add middleware, event handlers, etc.
        # Example: Add a startup event handler
        @app.on_event("startup")
        async def startup_event():
            logger.info("Starting up Hermod AI Assistant.")
            # Perform startup tasks here

        # Example: Add a shutdown event handler
        @app.on_event("shutdown")
        async def shutdown_event():
            logger.info("Shutting down Hermod AI Assistant.")
            # Perform cleanup tasks here

        logger.info("Hermod AI Assistant application initialized.")

        return app
    except Exception as e:
        logger.exception(f"Failed to create the application: {e}")
        raise e


app = create_app()

if __name__ == "__main__":
    import uvicorn

    # Run the application using uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

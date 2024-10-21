# src/app/middleware.py

import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from typing import Callable, Awaitable

# Configure logging
logger = logging.getLogger("uvicorn.error")  # Or configure a custom logger

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging requests and responses.
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        # Log the request
        logger.info(f"Incoming request: {request.method} {request.url}")
        start_time = time.time()

        # Process the request
        response = await call_next(request)

        # Log the response
        process_time = (time.time() - start_time) * 1000  # In milliseconds
        logger.info(
            f"Completed response: {response.status_code} in {process_time:.2f} ms"
        )
        return response


class PerformanceMiddleware(BaseHTTPMiddleware):
    """
    Middleware for measuring the performance of requests.
    Adds a custom header 'X-Process-Time' to the response.
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for handling authentication.
    Checks for a valid 'Authorization' header in the request.
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        # Extract the authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header:
            # Implement your authentication logic here
            token = auth_header.split(" ")[1] if " " in auth_header else auth_header
            # Validate the token (e.g., JWT validation)
            user = await self.authenticate_token(token)
            if user:
                # Attach the user to the request state
                request.state.user = user
            else:
                return Response("Unauthorized", status_code=401)
        else:
            return Response("Unauthorized", status_code=401)
        return await call_next(request)

    async def authenticate_token(self, token: str):
        # Implement your token validation logic here
        # For example, decode JWT and verify claims
        # Return user object if valid, else None
        from src.app.models import User  # Import your User model
        # Example placeholder logic:
        if token == "valid_token":
            return User(id=1, username="testuser")
        else:
            return None


class CORSMiddleware(BaseHTTPMiddleware):
    """
    Middleware for handling Cross-Origin Resource Sharing (CORS).
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"
        return response


class ExceptionHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for handling exceptions and returning custom error responses.
    """

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        try:
            return await call_next(request)
        except Exception as exc:
            logger.exception(f"Unhandled exception: {exc}")
            return Response(
                content="Internal Server Error",
                status_code=500,
                media_type="text/plain",
            )

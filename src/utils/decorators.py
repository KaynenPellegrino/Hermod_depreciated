# src/utils/decorators.py

import asyncio
import functools
import logging
import time
from typing import Any, Callable, TypeVar, Optional
from fastapi import Request, HTTPException, status

# Type variable used for type hinting
F = TypeVar('F', bound=Callable[..., Any])

# Configure the logger for this module
logger = logging.getLogger(__name__)

def log_execution(func: F) -> F:
    """
    Decorator that logs the execution of a function, including its arguments and execution time.
    Supports both synchronous and asynchronous functions.
    """

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger.info(f"Entering function '{func.__name__}' with args: {args}, kwargs: {kwargs}")
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(f"Exiting function '{func.__name__}' - Execution time: {end_time - start_time:.4f} seconds")
        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger.info(f"Entering function '{func.__name__}' with args: {args}, kwargs: {kwargs}")
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(f"Exiting function '{func.__name__}' - Execution time: {end_time - start_time:.4f} seconds")
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper  # type: ignore
    else:
        return sync_wrapper  # type: ignore

def authenticate(func: F) -> F:
    """
    Decorator that enforces authentication.
    Expects the first argument to be 'request' of type 'Request'.
    """

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        request: Request = kwargs.get('request') or (args[0] if args else None)
        if not isinstance(request, Request):
            raise RuntimeError("Request object not found in args or kwargs")
        token = request.headers.get("Authorization")
        if not token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
        try:
            scheme, _, param = token.partition(" ")
            if scheme.lower() != 'bearer':
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication scheme")
            # Implement your token verification logic here
            from src.utils.security import verify_token
            user = await verify_token(param)
            if not user:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token or user not found")
            # Attach the user to the request state
            request.state.user = user
        except Exception as e:
            logger.exception("Authentication failed")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed")
        return await func(*args, **kwargs)

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        request: Request = kwargs.get('request') or (args[0] if args else None)
        if not isinstance(request, Request):
            raise RuntimeError("Request object not found in args or kwargs")
        token = request.headers.get("Authorization")
        if not token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
        try:
            scheme, _, param = token.partition(" ")
            if scheme.lower() != 'bearer':
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication scheme")
            # Implement your token verification logic here
            from src.utils.security import verify_token
            user = verify_token(param)
            if not user:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token or user not found")
            # Attach the user to the request state
            request.state.user = user
        except Exception as e:
            logger.exception("Authentication failed")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication failed")
        return func(*args, **kwargs)

    if asyncio.iscoroutinefunction(func):
        return async_wrapper  # type: ignore
    else:
        return sync_wrapper  # type: ignore

def cache_results(ttl: Optional[int] = None):
    """
    Decorator that caches the result of a function call based on its arguments.
    Optionally accepts a time-to-live (ttl) in seconds.
    """
    def decorator(func: F) -> F:
        cache = {}
        lock = asyncio.Lock() if asyncio.iscoroutinefunction(func) else None

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            async with lock:
                if key in cache:
                    cached_result, timestamp = cache[key]
                    if ttl is None or (time.time() - timestamp) < ttl:
                        return cached_result
                    else:
                        del cache[key]
            result = await func(*args, **kwargs)
            async with lock:
                cache[key] = (result, time.time())
            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            if key in cache:
                cached_result, timestamp = cache[key]
                if ttl is None or (time.time() - timestamp) < ttl:
                    return cached_result
                else:
                    del cache[key]
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator

def measure_time(func: F) -> F:
    """
    Decorator that measures and logs the execution time of a function.
    Supports both synchronous and asynchronous functions.
    """

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.info(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper  # type: ignore
    else:
        return sync_wrapper  # type: ignore

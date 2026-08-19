"""
Retry helper utilities using tenacity for robust error handling and retries.

This module provides decorators and functions for retrying operations that may fail
due to transient issues like network problems, authentication errors, or temporary
service unavailability.
"""

import logging
from typing import Any, Callable, Optional, Type, Union
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_exception,
    before_sleep_log,
    after_log,
)

logger = logging.getLogger(__name__)


def is_authentication_error(exception: Exception) -> bool:
    """
    Check if an exception indicates an authentication/authorization error.
    
    Args:
        exception: The exception to check
        
    Returns:
        True if the exception indicates auth failure, False otherwise
    """
    error_str = str(exception).lower()
    error_type = type(exception).__name__.lower()
    
    # Check for common authentication error indicators
    auth_indicators = [
        "401", "unauthorized", "authentication", "token", "expired", 
        "invalid", "forbidden", "403", "access denied", "credential"
    ]
    
    return any(indicator in error_str or indicator in error_type for indicator in auth_indicators)


def retry_on_failure(
    max_attempts: int = 2,
    wait_min: float = 1.0,
    wait_max: float = 10.0,
    retry_on_auth_errors: bool = True,
    retry_on_exceptions: Optional[Union[Type[Exception], tuple]] = None,
    logger_name: Optional[str] = None,
) -> Callable:
    """
    Decorator for retrying operations with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts (default: 2)
        wait_min: Minimum wait time between retries in seconds (default: 1.0)
        wait_max: Maximum wait time between retries in seconds (default: 10.0)
        retry_on_auth_errors: Whether to retry on authentication errors (default: True)
        retry_on_exceptions: Specific exception types to retry on (default: None)
        logger_name: Logger name for retry messages (default: None)
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @retry_on_failure(max_attempts=3, wait_min=2.0)
        def call_llm_api():
            return llm_client.invoke(messages)
    """
    retry_logger = logging.getLogger(logger_name) if logger_name else logger
    
    # Build retry condition
    retry_conditions = []
    
    if retry_on_auth_errors:
        retry_conditions.append(retry_if_exception(is_authentication_error))
    
    if retry_on_exceptions:
        if isinstance(retry_on_exceptions, tuple):
            retry_conditions.append(retry_if_exception_type(retry_on_exceptions))
        else:
            retry_conditions.append(retry_if_exception_type(retry_on_exceptions))
    
    # If no specific conditions, retry on any Exception
    if not retry_conditions:
        retry_conditions.append(retry_if_exception_type(Exception))
    
    # Combine conditions with OR logic
    retry_condition = retry_conditions[0]
    for condition in retry_conditions[1:]:
        retry_condition = retry_condition | condition
    
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=wait_min, max=wait_max),
        retry=retry_condition,
        before_sleep=before_sleep_log(retry_logger, logging.WARNING),
        after=after_log(retry_logger, logging.INFO),
        reraise=True,
    )


def retry_llm_operation(
    operation: Callable,
    client_factory: Callable,
    max_attempts: int = 2,
    recreate_client_on_auth_error: bool = True,
    logger_name: Optional[str] = None,
) -> Any:
    """
    Retry an LLM operation with automatic client recreation on auth errors.
    
    This function is specifically designed for LLM operations where the client
    may need to be recreated when authentication tokens expire.
    
    Args:
        operation: Function that takes an LLM client and returns result
        client_factory: Function that creates a fresh LLM client
        max_attempts: Maximum number of attempts (default: 2)
        recreate_client_on_auth_error: Whether to recreate client on auth errors (default: True)
        logger_name: Logger name for retry messages (default: None)
        
    Returns:
        Result from the operation
        
    Raises:
        Last exception if all attempts fail
        
    Example:
        def my_operation(client):
            return client.invoke(messages)
        
        result = retry_llm_operation(
            operation=my_operation,
            client_factory=lambda: build_llm_client(),
            max_attempts=3
        )
    """
    retry_logger = logging.getLogger(logger_name) if logger_name else logger
    
    client = None
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            # Create client on first attempt or after auth error
            if client is None:
                if attempt == 0:
                    retry_logger.debug("Creating initial LLM client")
                else:
                    retry_logger.info(f"Recreating LLM client for attempt {attempt + 1}")
                client = client_factory()
            
            # Execute the operation
            retry_logger.debug(f"Executing LLM operation (attempt {attempt + 1})")
            result = operation(client)
            
            if attempt > 0:
                retry_logger.info(f"LLM operation succeeded on attempt {attempt + 1}")
            
            return result
            
        except Exception as e:
            last_exception = e
            retry_logger.warning(f"LLM operation failed (attempt {attempt + 1}): {e}")
            
            # Check if we should recreate client on auth error
            if recreate_client_on_auth_error and is_authentication_error(e):
                retry_logger.info("Authentication error detected, will recreate client on next attempt")
                client = None
            
            # If this is the last attempt, don't wait
            if attempt == max_attempts - 1:
                retry_logger.error(f"LLM operation failed after {max_attempts} attempts")
                break
            
            # Wait before next attempt (exponential backoff)
            import time
            wait_time = min(2 ** attempt, 10)  # Cap at 10 seconds
            retry_logger.info(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
    
    # Re-raise the last exception
    raise last_exception


def retry_with_fresh_client(
    max_attempts: int = 2,
    recreate_on_auth_error: bool = True,
    logger_name: Optional[str] = None,
) -> Callable:
    """
    Decorator for LLM operations that need client recreation on auth errors.
    
    The decorated function should accept a 'client_factory' parameter and
    return a function that takes the client as its first argument.
    
    Args:
        max_attempts: Maximum number of attempts (default: 2)
        recreate_on_auth_error: Whether to recreate client on auth errors (default: True)
        logger_name: Logger name for retry messages (default: None)
        
    Returns:
        Decorated function with retry and client recreation logic
        
    Example:
        @retry_with_fresh_client(max_attempts=3)
        def call_llm(client_factory):
            def _call(client):
                return client.invoke(messages)
            return retry_llm_operation(_call, client_factory, max_attempts=3)
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Extract client_factory from kwargs
            client_factory = kwargs.pop('client_factory', None)
            if client_factory is None:
                raise ValueError("client_factory parameter is required")
            
            # Create operation function
            def operation(client):
                return func(*args, client=client, **kwargs)
            
            # Use retry_llm_operation
            return retry_llm_operation(
                operation=operation,
                client_factory=client_factory,
                max_attempts=max_attempts,
                recreate_client_on_auth_error=recreate_on_auth_error,
                logger_name=logger_name,
            )
        
        return wrapper
    return decorator
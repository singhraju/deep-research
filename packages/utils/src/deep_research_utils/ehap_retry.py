"""
EHAP Token Retry Utilities

Provides tenacity-based retry wrappers for LLM invocations and API requests
that automatically handle EHAP token expiration and refresh.

Usage:
    from deep_research_utils.ehap_retry import llm_invoke, structured_llm_invoke, post_req
    
    # Initialize EHAP and LLM
    ehap = EHAPBase()
    llm = ChatOpenAI(api_key=ehap.get_token(), ...)
    
    # Use retry-wrapped invocations
    response = llm_invoke(llm, ehap, messages=[...])
    structured_response = structured_llm_invoke(llm, ehap, messages=[...], schema=MySchema)
    api_response = post_req(ehap, endpoint="/api/endpoint", body={...})
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
from tenacity import (
    retry,
    stop_after_attempt,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
    RetryCallState,
    wait_exponential
)
from openai import AuthenticationError

from deep_research_utils.ehap import EHAPBase
from deep_research_utils.logger_config import get_logger

logger = get_logger(__name__, console_output=True)

T = TypeVar('T')


def _before_retry_callback(retry_state: RetryCallState) -> None:
    """Log before retry attempt with token refresh details."""
    logger.warning(
        f"Retry attempt {retry_state.attempt_number} after AuthenticationError. "
        f"Refreshing EHAP token..."
    )


def _create_retry_decorator(max_attempts: int = 2):
    """Create a tenacity retry decorator for EHAP token errors."""
    return retry(
        retry=retry_if_exception_type(AuthenticationError),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=_before_retry_callback,
        reraise=True
    )


@_create_retry_decorator(max_attempts=2)
def llm_invoke(
    llm: Any,
    ehap: Optional[EHAPBase],
    messages: List[Dict[str, str]],
    llm_reinitializer: Optional[Callable[[], Any]] = None,
    **invoke_kwargs
) -> tuple[Any, Any]:
    """
    Invoke LLM with automatic EHAP token refresh on 401 AuthenticationError.
    
    This function proactively checks token expiry before invocation and 
    automatically retries with a fresh token if a 401 error occurs.
    
    Args:
        llm: LangChain LLM instance (e.g., ChatOpenAI)
        ehap: EHAPBase instance for token management
        messages: List of message dicts in LangChain format
        llm_reinitializer: Optional callback to reinitialize LLM with fresh token.
                          If provided, this will be called on retry instead of 
                          just refreshing the token.
        **invoke_kwargs: Additional arguments passed to llm.invoke()
    
    Returns:
        Tuple of (LLM response, potentially updated LLM instance)
        The LLM instance is returned to allow caller to update their reference
        if token refresh occurred.
    
    Raises:
        AuthenticationError: If retry with fresh token also fails
        
    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from deep_research_utils import EHAPBase
        >>> 
        >>> ehap = EHAPBase()
        >>> llm = ChatOpenAI(api_key=ehap.get_token(), ...)
        >>> 
        >>> response, updated_llm = llm_invoke(
        ...     llm=llm,
        ...     ehap=ehap,
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
        >>> llm = updated_llm  # Update reference if token was refreshed
    """
    # Externally-supplied LLM (orchestrator builder, test stub, etc.): no EHAP
    # token to manage and no reinitializer that knows how to mint one. Skip the
    # retry machinery and invoke directly, mirroring BaseAgent._invoke_with_token_retry.
    if ehap is None:
        return llm.invoke(messages, **invoke_kwargs), llm

    # Check if we need to reinitialize LLM with fresh token
    # Always get fresh token from Redis before invocation to ensure we have the latest token
    if llm_reinitializer:
        # Get current token from cache
        from deep_research_utils.cache_utils import get_token_cache_obj
        import time

        token_data = get_token_cache_obj()

        if not token_data or not token_data.get('access_token'):
            logger.info("Token not cached. Reinitializing LLM with fresh token.")
            llm = llm_reinitializer()
            logger.debug("LLM reinitialized with fresh token")
        else:
            # Check if LLM's token matches the cached token (detect revoked tokens)
            cached_iat = token_data.get('issued_at', 0)
            
            # Extract issued_at from LLM's embedded token to compare ages
            llm_iat = 0
            if hasattr(llm, 'openai_api_key') and llm.openai_api_key:
                try:
                    import jwt
                    # Handle SecretStr (Pydantic v2)
                    llm_token_str = llm.openai_api_key.get_secret_value() if hasattr(llm.openai_api_key, 'get_secret_value') else str(llm.openai_api_key)
                    llm_payload = jwt.decode(llm_token_str, options={"verify_signature": False})
                    llm_iat = llm_payload.get('iat', 0)
                except Exception as e:
                    logger.debug(f"Could not extract iat from LLM token: {e}")
            
            # If cached token is newer than LLM's token, reinitialize (handles Redis cron updates)
            if cached_iat > 0 and llm_iat > 0 and cached_iat > llm_iat:
                logger.warning(
                    f"LLM has stale token (issued at {llm_iat}) vs cached token (issued at {cached_iat}). "
                    f"Reinitializing with newer token."
                )
                llm = llm_reinitializer()
                logger.info("LLM reinitialized with newer cached token")
            else:
                # Check if token has expiry information (from memory cache or EHAP-fetched token)
                expiry_timestamp = token_data.get('expiry_timestamp', 0)
                
                if expiry_timestamp > 0:
                    # Token has expiry info - check if expired or about to expire (within 60 seconds)
                    current_time = time.time()
                    time_until_expiry = expiry_timestamp - current_time
                    
                    if time_until_expiry < 60:
                        logger.warning(f"Token expired or expiring soon (in {time_until_expiry:.0f}s). Reinitializing LLM with fresh token.")
                        # Force token refresh
                        ehap.force_token_refresh()
                        llm = llm_reinitializer()
                        logger.info("LLM reinitialized with fresh token")
                    else:
                        logger.debug(f"Token valid. Expires in {time_until_expiry:.0f} seconds")
                else:
                    # No expiry info available
                    token_source = token_data.get('source', 'unknown')
                    logger.debug(f"Using token from {token_source}")
    
    try:
        result = llm.invoke(messages, **invoke_kwargs)

        # Clear bypass flag after successful call so next request can use Redis again
        from deep_research_utils.cache_utils import clear_bypass_flag
        clear_bypass_flag()

        return result, llm
    except AuthenticationError:
        # Force token refresh for retry
        logger.warning("AuthenticationError caught. Forcing token refresh for retry.")
        if ehap is not None:
            ehap.force_token_refresh()
        
        # If reinitializer provided, recreate LLM with fresh token and retry immediately
        if llm_reinitializer:
            llm = llm_reinitializer()
            logger.debug("LLM reinitialized with fresh token for retry")
            
            # Retry with new LLM client (don't re-raise, do actual retry here)
            try:
                result = llm.invoke(messages, **invoke_kwargs)
                
                # Clear bypass flag after successful retry
                from deep_research_utils.cache_utils import clear_bypass_flag
                clear_bypass_flag()
                
                logger.info("Retry with fresh token succeeded")
                return result, llm
            except AuthenticationError as retry_error:
                # If still failing after refresh, re-raise to trigger tenacity retry
                logger.error("AuthenticationError persists after token refresh")
                raise retry_error
        else:
            # No reinitializer, re-raise to trigger tenacity retry with same LLM
            raise


@_create_retry_decorator(max_attempts=2)
def structured_llm_invoke(
    llm: Any,
    ehap: Optional[EHAPBase],
    messages: List[Dict[str, str]],
    schema: Type[T],
    llm_reinitializer: Optional[Callable[[], Any]] = None,
    **invoke_kwargs
) -> tuple[T, Any]:
    """
    Invoke LLM with structured output and automatic EHAP token refresh on 401.
    
    Uses LangChain's with_structured_output() to enforce a Pydantic schema.
    Proactively checks token expiry and retries with fresh token on 401 errors.
    
    Args:
        llm: LangChain LLM instance (e.g., ChatOpenAI)
        ehap: EHAPBase instance for token management
        messages: List of message dicts in LangChain format
        schema: Pydantic model class for structured output
        llm_reinitializer: Optional callback to reinitialize LLM with fresh token
        **invoke_kwargs: Additional arguments passed to structured_llm.invoke()
    
    Returns:
        Tuple of (Pydantic schema instance, potentially updated LLM instance)
        The LLM instance is returned to allow caller to update their reference
        if token refresh occurred.
    
    Raises:
        AuthenticationError: If retry with fresh token also fails
        ValidationError: If LLM output doesn't match schema
        
    Example:
        >>> from pydantic import BaseModel
        >>> from langchain_openai import ChatOpenAI
        >>> from deep_research_utils import EHAPBase
        >>> 
        >>> class IntentSchema(BaseModel):
        ...     intent: str
        ...     confidence: float
        >>> 
        >>> ehap = EHAPBase()
        >>> llm = ChatOpenAI(api_key=ehap.get_token(), ...)
        >>> 
        >>> result, updated_llm = structured_llm_invoke(
        ...     llm=llm,
        ...     ehap=ehap,
        ...     messages=[{"role": "user", "content": "Analyze this"}],
        ...     schema=IntentSchema
        ... )
        >>> llm = updated_llm  # Update reference if token was refreshed
        >>> print(result.intent, result.confidence)
    """
    # Externally-supplied LLM (orchestrator builder, test stub, etc.): no EHAP
    # token to manage and no reinitializer that knows how to mint one. Skip the
    # retry machinery and invoke directly, mirroring BaseAgent._invoke_with_token_retry.
    if ehap is None:
        structured_llm = llm.with_structured_output(schema)
        return structured_llm.invoke(messages, **invoke_kwargs), llm

    # Check if we need to reinitialize LLM with fresh token
    # Always get fresh token from Redis before invocation to ensure we have the latest token
    if llm_reinitializer:
        # Get current token from cache
        from deep_research_utils.cache_utils import get_token_cache_obj
        import time

        token_data = get_token_cache_obj()

        if not token_data or not token_data.get('access_token'):
            logger.info("Token not cached. Reinitializing LLM with fresh token.")
            llm = llm_reinitializer()
            logger.debug("LLM reinitialized with fresh token")
        else:
            # Check if LLM's token matches the cached token (detect revoked tokens)
            cached_iat = token_data.get('issued_at', 0)
            
            # Extract issued_at from LLM's embedded token to compare ages
            llm_iat = 0
            if hasattr(llm, 'openai_api_key') and llm.openai_api_key:
                try:
                    import jwt
                    # Handle SecretStr (Pydantic v2)
                    llm_token_str = llm.openai_api_key.get_secret_value() if hasattr(llm.openai_api_key, 'get_secret_value') else str(llm.openai_api_key)
                    llm_payload = jwt.decode(llm_token_str, options={"verify_signature": False})
                    llm_iat = llm_payload.get('iat', 0)
                except Exception as e:
                    logger.debug(f"Could not extract iat from LLM token: {e}")
            
            # If cached token is newer than LLM's token, reinitialize (handles Redis cron updates)
            if cached_iat > 0 and llm_iat > 0 and cached_iat > llm_iat:
                logger.warning(
                    f"LLM has stale token (issued at {llm_iat}) vs cached token (issued at {cached_iat}). "
                    f"Reinitializing with newer token."
                )
                llm = llm_reinitializer()
                logger.info("LLM reinitialized with newer cached token")
            else:
                # Check if token has expiry information (from memory cache or EHAP-fetched token)
                expiry_timestamp = token_data.get('expiry_timestamp', 0)
                
                if expiry_timestamp > 0:
                    # Token has expiry info - check if expired or about to expire (within 60 seconds)
                    current_time = time.time()
                    time_until_expiry = expiry_timestamp - current_time
                    
                    if time_until_expiry < 60:
                        logger.warning(f"Token expired or expiring soon (in {time_until_expiry:.0f}s). Reinitializing LLM with fresh token.")
                        # Force token refresh
                        ehap.force_token_refresh()
                        llm = llm_reinitializer()
                        logger.info("LLM reinitialized with fresh token")
                    else:
                        logger.debug(f"Token valid. Expires in {time_until_expiry:.0f} seconds")
                else:
                    # No expiry info available
                    token_source = token_data.get('source', 'unknown')
                    logger.debug(f"Using token from {token_source}")
    
    try:
        structured_llm = llm.with_structured_output(schema)
        result = structured_llm.invoke(messages, **invoke_kwargs)

        # Clear bypass flag after successful call so next request can use Redis again
        from deep_research_utils.cache_utils import clear_bypass_flag
        clear_bypass_flag()

        return result, llm
    except AuthenticationError:
        # Force token refresh for retry
        logger.warning("AuthenticationError caught. Forcing token refresh for retry.")
        if ehap is not None:
            ehap.force_token_refresh()
        
        # If reinitializer provided, recreate LLM with fresh token and retry immediately
        if llm_reinitializer:
            llm = llm_reinitializer()
            logger.debug("LLM reinitialized with fresh token for retry")
            
            # Retry with new LLM client (don't re-raise, do actual retry here)
            try:
                structured_llm = llm.with_structured_output(schema)
                result = structured_llm.invoke(messages, **invoke_kwargs)
                
                # Clear bypass flag after successful retry
                from deep_research_utils.cache_utils import clear_bypass_flag
                clear_bypass_flag()
                
                logger.info("Retry with fresh token succeeded")
                return result, llm
            except AuthenticationError as retry_error:
                # If still failing after refresh, re-raise to trigger tenacity retry
                logger.error("AuthenticationError persists after token refresh")
                raise retry_error
        else:
            # No reinitializer, re-raise to trigger tenacity retry with same LLM
            raise


@_create_retry_decorator(max_attempts=2)
def structured_llm_invoke_with_tokens(
    llm: Any,
    ehap: Optional[EHAPBase],
    messages: List[Dict[str, str]],
    schema: Type[T],
    llm_reinitializer: Optional[Callable[[], Any]] = None,
    **invoke_kwargs
) -> tuple[T, Any, Any]:
    """
    Invoke LLM with structured output and return raw response for token tracking.
    
    This is identical to structured_llm_invoke but returns a 3-tuple including
    the raw response object for token usage metadata extraction.
    
    Args:
        llm: LangChain LLM instance (e.g., ChatOpenAI)
        ehap: EHAPBase instance for token management
        messages: List of message dicts in LangChain format
        schema: Pydantic model class for structured output
        llm_reinitializer: Optional callback to reinitialize LLM with fresh token
        **invoke_kwargs: Additional arguments passed to structured_llm.invoke()
    
    Returns:
        Tuple of (Pydantic schema instance, potentially updated LLM instance, raw response)
        The raw response contains usage_metadata for token tracking.
    
    Raises:
        AuthenticationError: If retry with fresh token also fails
        ValidationError: If LLM output doesn't match schema
        
    Example:
        >>> from pydantic import BaseModel
        >>> from langchain_openai import ChatOpenAI
        >>> from deep_research_utils import EHAPBase
        >>> 
        >>> class IntentSchema(BaseModel):
        ...     intent: str
        ...     confidence: float
        >>> 
        >>> ehap = EHAPBase()
        >>> llm = ChatOpenAI(api_key=ehap.get_token(), ...)
        >>> 
        >>> result, updated_llm, raw_response = structured_llm_invoke_with_tokens(
        ...     llm=llm,
        ...     ehap=ehap,
        ...     messages=[{"role": "user", "content": "Analyze this"}],
        ...     schema=IntentSchema
        ... )
        >>> llm = updated_llm  # Update reference if token was refreshed
        >>> print(result.intent, result.confidence)
        >>> # Extract tokens from raw_response
        >>> usage = getattr(raw_response, "usage_metadata", None)
    """
    # Externally-supplied LLM (orchestrator builder, test stub, etc.): no EHAP
    # token to manage and no reinitializer that knows how to mint one. Skip the
    # retry machinery and invoke directly, mirroring BaseAgent._invoke_with_token_retry.
    if ehap is None:
        structured_llm = llm.with_structured_output(schema, include_raw=True)
        response = structured_llm.invoke(messages, **invoke_kwargs)
        if isinstance(response, dict):
            parsed = response.get("parsed") or schema()
            raw = response.get("raw")
        else:
            parsed = response
            raw = response
        return parsed, llm, raw

    # Check if we need to reinitialize LLM with fresh token
    # Always get fresh token from Redis before invocation to ensure we have the latest token
    if llm_reinitializer:
        # Get current token from cache
        from deep_research_utils.cache_utils import get_token_cache_obj
        import time

        token_data = get_token_cache_obj()

        if not token_data or not token_data.get('access_token'):
            logger.info("Token not cached. Reinitializing LLM with fresh token.")
            llm = llm_reinitializer()
            logger.debug("LLM reinitialized with fresh token")
        else:
            # Check if token has expiry information (from memory cache or EHAP-fetched token)
            expiry_timestamp = token_data.get('expiry_timestamp', 0)

            if expiry_timestamp > 0:
                # Token has expiry info - check if expired or about to expire (within 60 seconds)
                current_time = time.time()
                time_until_expiry = expiry_timestamp - current_time

                if time_until_expiry < 60:
                    logger.warning(f"Token expired or expiring soon (in {time_until_expiry:.0f}s). Reinitializing LLM with fresh token.")
                    # Force token refresh
                    ehap.force_token_refresh()
                    llm = llm_reinitializer()
                    logger.info("LLM reinitialized with fresh token")
                else:
                    logger.debug(f"Token valid. Expires in {time_until_expiry:.0f} seconds")
            else:
                # Token from Redis cron job (no expiry info) - use as-is, rely on 401 retry if expired
                token_source = token_data.get('source', 'unknown')
                logger.debug(f"Using token from {token_source} (no expiry info available)")

    try:
        structured_llm = llm.with_structured_output(schema, include_raw=True)
        response = structured_llm.invoke(messages, **invoke_kwargs)

        # Extract parsed result and raw response
        if isinstance(response, dict):
            parsed = response.get("parsed") or schema()
            raw = response.get("raw")
        else:
            parsed = response
            raw = response

        # Clear bypass flag after successful call so next request can use Redis again
        from deep_research_utils.cache_utils import clear_bypass_flag
        clear_bypass_flag()

        return parsed, llm, raw
    except AuthenticationError:
        # Force token refresh for retry
        logger.warning("AuthenticationError caught. Forcing token refresh for retry.")
        if ehap is not None:
            ehap.force_token_refresh()
        
        # If reinitializer provided, recreate LLM with fresh token and retry immediately
        if llm_reinitializer:
            llm = llm_reinitializer()
            logger.debug("LLM reinitialized with fresh token for retry")
            
            # Retry with new LLM client (don't re-raise, do actual retry here)
            try:
                structured_llm = llm.with_structured_output(schema, include_raw=True)
                response = structured_llm.invoke(messages, **invoke_kwargs)
                
                # Extract parsed result and raw response
                if isinstance(response, dict):
                    parsed = response.get("parsed") or schema()
                    raw = response.get("raw")
                else:
                    parsed = response
                    raw = response
                
                # Clear bypass flag after successful retry
                from deep_research_utils.cache_utils import clear_bypass_flag
                clear_bypass_flag()
                
                logger.info("Retry with fresh token succeeded")
                return parsed, llm, raw
            except AuthenticationError as retry_error:
                # If still failing after refresh, re-raise to trigger tenacity retry
                logger.error("AuthenticationError persists after token refresh")
                raise retry_error
        else:
            # No reinitializer, re-raise to trigger tenacity retry with same LLM
            raise


@_create_retry_decorator(max_attempts=2)
def post_req(
    ehap: EHAPBase,
    endpoint: str,
    body: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    files: Optional[Any] = None,
    stream: bool = False,
    **request_kwargs
) -> bytes:
    """
    Make POST request to EHAP API with automatic token refresh on 401.
    
    Uses EHAPBase's sendHttpRequest which already handles token refresh,
    but adds tenacity retry logic for additional robustness.
    
    Args:
        ehap: EHAPBase instance for token management and HTTP requests
        endpoint: API endpoint path (e.g., "/api/v1/resource")
        body: JSON body for POST request (optional)
        params: Query parameters (optional)
        files: Files to upload (optional)
        stream: Whether to stream the response (default: False)
        **request_kwargs: Additional arguments (currently unused, for future extension)
    
    Returns:
        Response content as bytes
    
    Raises:
        AuthenticationError: If retry with fresh token also fails
        requests.HTTPError: For non-401 HTTP errors
        
    Example:
        >>> from deep_research_utils import EHAPBase
        >>> 
        >>> ehap = EHAPBase()
        >>> response = post_req(
        ...     ehap=ehap,
        ...     endpoint="/api/v1/analyze",
        ...     body={"query": "Show me data"}
        ... )
        >>> print(response.decode('utf-8'))
    """
    try:
        # EHAPBase.sendHttpRequest already handles token refresh internally,
        # but we add tenacity retry for additional robustness
        return ehap.sendHttpRequest(
            data=body or "",
            files=files or "",
            params=params or "",
            method="POST",
            endpoint=endpoint,
            stream=stream
        )
    except AuthenticationError:
        # Force token refresh for retry
        logger.warning("AuthenticationError caught. Forcing token refresh for retry.")
        ehap.force_token_refresh()
        
        # Re-raise to trigger tenacity retry
        raise


# Convenience function for backward compatibility with base_agent.py
def invoke_with_token_retry(
    llm: Any,
    messages: List[Dict[str, str]],
    ehap: EHAPBase,
    llm_reinitializer: Callable[[], Any],
    logger_instance: Optional[logging.Logger] = None,
    **invoke_kwargs
) -> tuple[Any, Any]:
    """
    Backward-compatible wrapper for base_agent.py migration.
    
    This is a convenience function that matches the signature expected by
    BaseAgent._invoke_with_token_retry() for easier migration.
    
    Args:
        llm: LangChain LLM instance
        messages: List of message dicts
        ehap: EHAPBase instance
        llm_reinitializer: Callback to reinitialize LLM
        logger_instance: Optional logger (unused, kept for compatibility)
        **invoke_kwargs: Additional invoke arguments
    
    Returns:
        Tuple of (LLM response, potentially updated LLM instance)
    """
    return llm_invoke(
        llm=llm,
        ehap=ehap,
        messages=messages,
        llm_reinitializer=llm_reinitializer,
        **invoke_kwargs
    )

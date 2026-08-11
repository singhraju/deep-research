import os
import time
import requests
from dotenv import load_dotenv
from typing import Optional, Union, Dict, Any

from deep_research_utils.cache_utils import (
    get_token_cache_obj, 
    set_token_cache, 
    clear_token_cache
)
from deep_research_utils.logger_config import get_logger
from deep_research_utils.app_constant import AppConstants

logger = get_logger(__name__)

# Load environment variables from .env file
# Users should have .env in their project root or set environment variables directly
load_dotenv()


class EHAPBase:
    """Base class for handling authentication and HTTP requests with token expiry management."""

    _token_cache = get_token_cache_obj()

    def __init__(self, base_url: Optional[str] = None,
                 client_id: Optional[str] = None,
                 client_secret: Optional[str] = None,
                 verify: Optional[Union[str, bool]] = None):
        self.base_url = base_url or AppConstants.EHAP_BASE_URL
        self.client_id = client_id or AppConstants.EHAP_CLIENT_ID
        self.client_secret = client_secret or AppConstants.EHAP_CLIENT_SECRET
        # Handle SSL verification: if SSL_CERT_FILE is "false" or "False", disable verification
        verify_value = verify if verify is not None else AppConstants.SSL_CERT_FILE
        if verify_value and isinstance(verify_value, str) and verify_value.lower() == 'false':
            self.verify = False
        else:
            self.verify = verify_value if verify_value else True  # Default to True if not set

    def _fetch_new_token(self) -> Optional[str]:
        """Authenticate using client credentials and store in Redis/memory cache with configurable TTL."""
        auth_url = f"{self.base_url}/oauth2/token"
        headers = {"Content-Type": "application/json"}
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials"
        }
        logger.info(f"Requesting new access token from {auth_url} with client_id: {self.client_id}")
        try:
            response = requests.post(auth_url, json=data, headers=headers, verify=False)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to obtain access token: {e}")
            raise
        
        resp_json = response.json()
        token = resp_json.get("access_token")
        
        # Extract issued_at from JWT for token age comparison
        try:
            import jwt
            payload = jwt.decode(token, options={"verify_signature": False})
            iat = payload.get('iat', 0)
        except Exception as e:
            logger.warning(f"Failed to extract issued_at from token: {e}")
            iat = 0
        
        # Use configurable cache TTL from AppConstants (default: 90 minutes)
        cache_ttl = AppConstants.CACHE_TTL
        expiry_timestamp = time.time() + cache_ttl
        
        # Simplified token data structure
        token_data = {
            'access_token': token,
            'expiry_timestamp': expiry_timestamp,  # For Redis cache TTL
            'issued_at': iat  # For token age comparison to detect revoked tokens
        }
        
        # Use Redis cache function with configurable TTL
        if set_token_cache(token_data):
            logger.info(f"Access token stored in cache for {cache_ttl // 60} minutes (until {time.ctime(expiry_timestamp)})")
        else:
            logger.warning("Failed to store token in cache, storing in class cache as fallback")
            # Fallback to class-level cache
            self.__class__._token_cache.update(token_data)
        
        logger.info(f"Access token generated successfully and cached for {cache_ttl // 60} minutes")
        return token

    def _call_api(self, data="", files="", params="", method="", endpoint="", stream=False) -> requests.Response:
        """Call the EHAP API, ensuring a valid token. Fetch new token if not in cache."""
        # Get current token from Redis/memory cache
        token_data = get_token_cache_obj()
        token = token_data.get('access_token')
        
        # If no token in cache, fetch a new one
        if not token:
            logger.info("Access token not in cache. Fetching a new token.")
            token = self._fetch_new_token()

        url = f"{self.base_url}{endpoint}"
        logger.debug(f"Calling EHAP url: {url}")
        headers = {
            "Authorization": f"Bearer {token}",
        }
        method = "POST"
        try:
            response = requests.request(
                method=method, url=url, headers=headers, json=data, files=files, params=params, stream=stream,
                verify=self.verify
            )
        except Exception as e:
            logger.error(f"HTTP request error: {e}")
            raise
            
        if response.status_code == 401:
            logger.warning("Received 401 Unauthorized. Token may have been revoked. Refreshing and retrying.")
            # Clear the Redis/memory cache to force a fresh token
            if not clear_token_cache():
                logger.warning("Failed to clear Redis cache, clearing class cache as fallback")
                # Fallback to class-level cache clearing
                self.__class__._token_cache.clear()
            
            token = self._fetch_new_token()
            headers["Authorization"] = f"Bearer {token or ''}"
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data,
                    files=files,
                    params=params,
                    stream=stream,
                    verify=self.verify
                )
            except Exception as e:
                logger.error(f"HTTP request error after token refresh: {e}")
                raise
        return response

    def sendHttpRequest(self, data="", files="", params="", method="", endpoint="", stream=False) -> Optional[bytes]:
        response = self._call_api(
            data=data,
            files=files,
            params=params,
            method=method,
            endpoint=endpoint,
            stream=stream
        )
        response.raise_for_status()
        return response.content

    def get_token(self) -> Optional[str]:
        """
        Get the current valid access token from Redis/memory cache.
        Fetch new token if not in cache (Redis handles configurable TTL automatically).
        """
        # Get token from Redis/memory cache
        from deep_research_utils.cache_utils import _token_cache_instance
        
        token_data = get_token_cache_obj()
        token = token_data.get('access_token')
        token_source = token_data.get('source', 'unknown')
        
        # If no token in cache at all, fetch a new one
        if not token:
            logger.info("Token not in cache. Fetching new token.")
            token = self._fetch_new_token()
        # If token is from Redis but memory cache is empty, it means force_token_refresh() was called
        # In this case, fetch a fresh token from EHAP instead of using the potentially expired Redis token
        elif token_source == 'redis_cron' and not _token_cache_instance._memory_cache:
            logger.warning("Memory cache cleared but Redis token exists. Fetching fresh token from EHAP.")
            token = self._fetch_new_token()
        
        return token
    
    def is_token_cached(self) -> bool:
        """
        Check if token exists in Redis/memory cache.
        
        Returns:
            True if token is cached, False if needs to be fetched
        """
        token_data = get_token_cache_obj()
        token = token_data.get('access_token')
        return bool(token)
    
    def force_token_refresh(self) -> None:
        """
        Clear memory token cache and set bypass flag to skip Redis on next get_token() call.
        
        This method does not fetch the token immediately - it only invalidates
        the memory cache and sets a flag to bypass Redis (since Redis token is invalid).
        The next call to get_token() will fetch a fresh token from EHAP.
        
        Example:
            >>> ehap = EHAPBase()
            >>> ehap.force_token_refresh()  # Clears cache and sets bypass flag
            >>> token = ehap.get_token()     # Fetches fresh token from EHAP (skips Redis)
        """
        logger.info("Clearing token cache and setting Redis bypass flag...")
        
        # Clear memory cache and set bypass flag to skip Redis on next get_token()
        if not clear_token_cache(set_bypass_redis=True):
            logger.warning("Failed to clear cache via TokenCache, clearing class cache as fallback")
            # Fallback to class-level cache clearing
            self.__class__._token_cache.clear()
        else:
            logger.info("Memory cache cleared and Redis bypass flag set - next get_token() will fetch from EHAP")
    
    def get_token_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the current token from Redis/memory cache (for debugging/monitoring).
        
        Returns:
            Dictionary with token status and cache information
        """
        # Get token data from Redis/memory cache
        token_data = get_token_cache_obj()
        token = token_data.get('access_token')
        expiry_timestamp = token_data.get('expiry_timestamp', 0)
        
        return {
            "has_token": bool(token),
            "cache_ttl_minutes": AppConstants.CACHE_TTL // 60 if token else None,
            "cached_until": time.ctime(expiry_timestamp) if expiry_timestamp else None,
            "cache_type": "Redis/Memory" if token_data else "Empty",
        }


EHAP = EHAPBase()

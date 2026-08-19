import json
import logging
import time
import jwt
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager
from datetime import datetime, timezone

from cachetools import TTLCache, cached

from deep_research_utils import AppConstants
from deep_research_utils.logger_config import get_logger

# Set up logger for this module with console output enabled
logger = get_logger(__name__, console_output=True)

# Use plain dict for token cache - fallback when Redis is not enabled
_token_cache: Dict[str, Any] = {}


def is_jwt_is_valid(token: str) -> bool:
    """
    check if JWT is valid or not
    Decode JWT token and log expiration information without verification.
    
    Args:
        token: JWT token string
    """
    try:
        # Decode JWT without verification (we just want to read the payload)
        payload = jwt.decode(token, options={"verify_signature": False})
        
        # Extract timestamps
        iat = payload.get('iat')  # Issued At
        exp = payload.get('exp')  # Expiration
        nbf = payload.get('nbf')  # Not Before
        
        # Check if exp claim exists
        if not exp:
            logger.warning("JWT token missing 'exp' (expiration) claim - treating as invalid")
            return False
        
        # Get current time
        current_time = time.time()
        time_remaining = exp - current_time
        
        # Convert timestamps to readable format (UTC) using timezone-aware datetime
        exp_datetime = datetime.fromtimestamp(exp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        iat_datetime = datetime.fromtimestamp(iat, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC') if iat else 'N/A'
        current_datetime = datetime.fromtimestamp(current_time, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        
        # Log token information
        logger.info(f"JWT Token Information:")
        logger.info(f"Issued At (iat): {iat_datetime} | Expires At (exp): {exp_datetime} (timestamp: {exp}) | Current Time: {current_datetime}")
        
        if time_remaining > 0:
            minutes_remaining = int(time_remaining / 60)
            seconds_remaining = int(time_remaining % 60)
            logger.info(f"Token Status: VALID !!")
            logger.info(f"Time Remaining: {minutes_remaining} minutes {seconds_remaining} seconds ({int(time_remaining)} seconds)")
            return True
        else:
            minutes_expired = int(abs(time_remaining) / 60)
            logger.warning(f"  Token Status: EXPIRED !!")
            logger.warning(f"  Time Remaining: EXPIRED {minutes_expired} minutes ago ({int(time_remaining)} seconds)")
            return False
        
    except jwt.DecodeError as e:
        logger.error(f"Failed to decode JWT token: {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking JWT token: {e}")
        return False



class RedisUtils:
    """
    Redis utility class for managing Redis connections and operations.
    
    Provides methods for creating connections, executing queries, and managing
    the Redis cache with proper error handling following team's connection pattern.
    """
    
    def __init__(self):
        self._redis_client = None
        self._is_connected = False
    
    @property
    def is_redis_enabled(self) -> bool:
        """Check if Redis is enabled in configuration."""
        return AppConstants.REDIS_ENABLED
    
    def create_client(self) -> Optional[Any]:
        """
        Create and return Redis client following team's pattern.
        
        Returns:
            Redis client instance or None if creation fails
        """
        if not self.is_redis_enabled:
            logger.debug("Redis is disabled in configuration")
            return None
        
        try:
            import redis
            
            # Create client using team's pattern with configurable parameters
            client_kwargs = {
                'host': AppConstants.REDIS_HOST,
                'port': AppConstants.REDIS_PORT,
                'db': AppConstants.REDIS_DB,
                'socket_connect_timeout': AppConstants.REDIS_SOCKET_CONNECT_TIMEOUT,
                'socket_timeout': AppConstants.REDIS_SOCKET_TIMEOUT,
                'decode_responses': True  # Automatically decode responses to strings
            }
            
            # Add password if provided
            if AppConstants.REDIS_PASSWORD:
                client_kwargs['password'] = AppConstants.REDIS_PASSWORD
            
            self._redis_client = redis.Redis(**client_kwargs)
            
            # Test connection
            self._redis_client.ping()
            self._is_connected = True
            logger.info(f"Redis client created and connection verified to {AppConstants.REDIS_HOST}:{AppConstants.REDIS_PORT}")
            
            return self._redis_client
            
        except Exception as e:
            logger.error(f"Failed to create Redis client: {e}")
            self._is_connected = False
            return None
    
    def close_connection(self):
        """Close Redis connection and cleanup resources."""
        try:
            if self._redis_client:
                self._redis_client.close()
                self._redis_client = None
                logger.info("Redis client connection closed")
                
            self._is_connected = False
            
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
    
    @contextmanager
    def get_client(self):
        """
        Context manager for Redis client with automatic cleanup.
        
        Usage:
            with redis_utils.get_client() as client:
                if client:
                    client.set("key", "value")
        """
        client = None
        try:
            if not self._is_connected:
                client = self.create_client()
            else:
                client = self._redis_client
            
            yield client
        finally:
            # Don't close here as we want to reuse connections
            pass
    
    # Redis set functionality removed - tokens are managed by external cron job
    def get_value(self, key: str) -> Optional[str]:
        """
        Get a value from Redis using direct key (no prefix).
        
        Args:
            key: Direct Redis key
            
        Returns:
            String value or None if not found/error
        """
        try:
            with self.get_client() as client:
                if not client:
                    return None
                
                # Use direct key (no prefix)
                value = client.get(key)
                if value is None:
                    return None
                
                # Return as string (tokens are stored as strings)
                return value.decode('utf-8') if isinstance(value, bytes) else value
                
        except Exception as e:
            logger.error(f"Failed to get Redis key {key}: {e}")
            return None
    
    
    def delete_value(self, key: str) -> bool:
        """
        Delete a key from Redis using direct key (no prefix).
        
        Args:
            key: Direct Redis key
            
        Returns:
            True if key was deleted, False otherwise
        """
        try:
            with self.get_client() as client:
                if not client:
                    return False
                
                # Use direct key (no prefix)
                result = client.delete(key)
                logger.debug(f"Deleted Redis key: {key}")
                return bool(result)
                
        except Exception as e:
            logger.error(f"Failed to delete Redis key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in Redis using direct key (no prefix).
        
        Args:
            key: Direct Redis key to check
            
        Returns:
            True if key exists, False otherwise
        """
        try:
            with self.get_client() as client:
                if not client:
                    return False
                
                # Use direct key (no prefix)
                return bool(client.exists(key))
                
        except Exception as e:
            logger.error(f"Failed to check Redis key existence {key}: {e}")
            return False
    
    def get_ttl(self, key: str) -> int:
        """
        Get TTL of a key in Redis using direct key (no prefix).
        
        Args:
            key: Direct Redis key
            
        Returns:
            TTL in seconds, -1 if key has no TTL, -2 if key doesn't exist
        """
        try:
            with self.get_client() as client:
                if not client:
                    return -2
                
                # Use direct key (no prefix)
                return client.ttl(key)
                
        except Exception as e:
            logger.error(f"Failed to get TTL for Redis key {key}: {e}")
            return -2


# Global Redis utils instance
_redis_utils = RedisUtils()


class TokenCache:
    """
    Token cache implementation with Redis support and in-memory fallback.
    
    Automatically uses Redis when enabled and available, falls back to
    in-memory dictionary cache otherwise.
    """
    
    def __init__(self):
        self._memory_cache: Dict[str, Any] = {}
        self._use_redis = AppConstants.REDIS_ENABLED
        self._bypass_redis = False  # Flag to temporarily bypass Redis when token is invalid
        
        logger.info(f"TokenCache initializing... (Redis enabled: {self._use_redis})")
        if self._use_redis:
            # Try to initialize Redis connection
            try:
                redis_client = _redis_utils.create_client()
                if redis_client:
                    logger.info("TokenCache initialized with Redis backend")
                else:
                    self._use_redis = False
                    logger.warning("TokenCache falling back to in-memory cache (Redis unavailable)")
            except Exception as e:
                self._use_redis = False
                logger.warning(f"TokenCache falling back to in-memory cache (Redis error: {e})")
        else:
            logger.info("TokenCache initialized with in-memory backend (Redis disabled)")
    
    def set_token(self, token_data: Dict[str, Any]) -> bool:
        """
        Set token data in in-memory cache only.
        Redis tokens are set by external cron job.
        
        Args:
            token_data: Dictionary containing token information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Only store in memory cache - Redis is managed by cron job
            self._memory_cache.clear()
            self._memory_cache.update(token_data)
            
            # NOTE: Do NOT clear bypass flag here - it should persist until after successful LLM call
            # If we clear it immediately, retry will go back to Redis and get the stale token again
            # The flag will be cleared manually after successful API call or on next normal flow
            
            logger.debug("Token stored in memory cache (Redis managed by cron job)")
            return True
        except Exception as e:
            logger.error(f"Failed to set token in memory cache: {e}")
            return False
    
    def get_token(self, bypass_redis: bool = False) -> Dict[str, Any]:
        """
        Get token data from cache.
        Note: token from redis is {'access_token': <value>, 'expires_at': <int value>}
        
        First tries to get token from Redis using REDIS_TOKEN_KEY (unless bypassed),
        then falls back to memory cache.
        
        Args:
            bypass_redis: If True, skip Redis and only check memory cache (used when Redis token is invalid)
        
        Returns:
            Dictionary containing cached token data
        """
        try:
            # Use instance bypass flag or parameter
            should_bypass = bypass_redis or self._bypass_redis
            
            if self._use_redis and AppConstants.REDIS_TOKEN_KEY and not should_bypass:
                # Get token directly from Redis using the specific token key
                token_value = _redis_utils.get_value(AppConstants.REDIS_TOKEN_KEY)
                if token_value is not None:
                    str_token = json.loads(token_value)["access_token"]
                    # Check if token is valid (not expired)
                    if is_jwt_is_valid(str_token):
                        # Extract expiry and issued-at timestamps from JWT for proactive checks in ehap_retry.py
                        try:
                            payload = jwt.decode(str_token, options={"verify_signature": False})
                            exp = payload.get('exp', 0)
                            iat = payload.get('iat', 0)
                        except Exception as e:
                            logger.warning(f"Failed to extract timestamps from Redis token: {e}")
                            exp = 0
                            iat = 0
                        
                        logger.debug(f"Token retrieved from Redis using key: {AppConstants.REDIS_TOKEN_KEY}")
                        return {
                            'access_token': str_token,
                            'expiry_timestamp': exp,  # Enables proactive LLM reinitialization
                            'issued_at': iat,  # Enables token age comparison to detect revoked tokens
                            'source': 'redis_cron'
                        }
                    else:
                        # Token exists but is expired - set bypass flag and fall through to EHAP
                        logger.warning(f"Redis token is expired. Setting bypass flag to fetch from EHAP.")
                        self._bypass_redis = True
                else:
                    logger.debug(f"No token found in Redis for key: {AppConstants.REDIS_TOKEN_KEY}")
            elif should_bypass:
                logger.debug("Bypassing Redis due to invalid token, using memory cache only")
            
            # Check memory cache as fallback
            if self._memory_cache:
                logger.debug("Token retrieved from memory cache fallback")
                return self._memory_cache
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get token from cache: {e}")
            return self._memory_cache if self._memory_cache else {}
    
    def clear_token(self, set_bypass_redis: bool = False) -> bool:
        """
        Clear token data from memory cache only.
        Redis tokens are managed by external cron job.
        
        Args:
            set_bypass_redis: If True, set flag to bypass Redis on next get_token() call
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Only clear memory cache - Redis is managed by cron job
            self._memory_cache.clear()
            
            # Set bypass flag if requested (used when Redis token is invalid)
            if set_bypass_redis:
                self._bypass_redis = True
                logger.warning("Redis bypass flag set - next get_token() will skip Redis and fetch from EHAP")
            
            logger.debug("Token cleared from memory cache (Redis managed by cron job)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear token from memory cache: {e}")
            return False
    
    def clear_bypass_flag(self) -> None:
        """
        Clear the Redis bypass flag after successful LLM call.
        This allows the system to go back to using Redis tokens on next request.
        """
        if self._bypass_redis:
            self._bypass_redis = False
            logger.debug("Redis bypass flag cleared - will use Redis tokens on next request")
    
    def is_token_expired(self) -> bool:
        """
        Check if the cached token is expired.
        
        Returns:
            True if token is expired or not found, False otherwise
        """
        try:
            token_data = self.get_token()
            if not token_data or 'expiry_timestamp' not in token_data:
                return True
            
            current_time = time.time()
            expiry_timestamp = token_data.get('expiry_timestamp', 0)
            
            is_expired = current_time >= expiry_timestamp
            if is_expired:
                logger.debug("Cached token is expired")
            
            return is_expired
            
        except Exception as e:
            logger.error(f"Failed to check token expiry: {e}")
            return True


# Global token cache instance
_token_cache_instance = TokenCache()


def get_token_cache_obj() -> Dict[str, Any]:
    """
    Get token cache object for backward compatibility.
    
    Returns:
        Dictionary containing cached token data
    """
    return _token_cache_instance.get_token()


def set_token_cache(token_data: Dict[str, Any]) -> bool:
    """
    Set token data in cache.
    
    Args:
        token_data: Dictionary containing token information
        
    Returns:
        True if successful, False otherwise
    """
    return _token_cache_instance.set_token(token_data)


def clear_token_cache(set_bypass_redis: bool = False) -> bool:
    """
    Clear token data from cache.
    
    Args:
        set_bypass_redis: If True, set flag to bypass Redis on next get_token() call
    
    Returns:
        True if successful, False otherwise
    """
    return _token_cache_instance.clear_token(set_bypass_redis=set_bypass_redis)


def clear_bypass_flag() -> None:
    """
    Clear the Redis bypass flag after successful LLM call.
    This allows the system to go back to using Redis tokens on next request.
    """
    _token_cache_instance.clear_bypass_flag()


def is_token_expired() -> bool:
    """
    Check if the cached token is expired.
    
    Returns:
        True if token is expired or not found, False otherwise
    """
    return _token_cache_instance.is_token_expired()


def get_redis_utils() -> RedisUtils:
    """
    Get global Redis utils instance.
    
    Returns:
        RedisUtils instance for direct Redis operations
    """
    return _redis_utils


def cleanup_cache():
    """
    Cleanup cache resources and connections.
    Call this when shutting down the application.
    """
    try:
        _redis_utils.close_connection()
        _token_cache_instance._memory_cache.clear()
        logger.info("Cache resources cleaned up")
    except Exception as e:
        logger.error(f"Error during cache cleanup: {e}")

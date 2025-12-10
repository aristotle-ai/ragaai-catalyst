"""
auth_manager.py - Centralized authentication and token management

This module provides authentication services for the entire RagaAI-Catalyst SDK.
It handles token generation, refresh, expiration tracking, and credential management.
"""

import logging
import os
import threading
import time
from typing import Dict, Optional, Union

import requests

logger = logging.getLogger("RagaAICatalyst.AuthManager")


class AuthManager:
    """
    Centralized authentication manager for RagaAI-Catalyst SDK.
    
    Handles:
    - Token generation and refresh
    - Token expiration tracking
    - Automatic background token refresh
    - Credential management
    - Authentication header generation
    """
    
    # Class-level state
    BASE_URL = None
    TIMEOUT = 10  # Default timeout in seconds
    TOKEN_EXPIRY_TIME = 6 * 60 * 60  # Default 6 hours in seconds
    
    _access_key = None
    _secret_key = None
    _token_expiry = None
    _token_refresh_lock = threading.Lock()
    _refresh_thread = None
    
    @classmethod
    def initialize(
        cls,
        access_key: str,
        secret_key: str,
        base_url: Optional[str] = None,
        token_expiry_hours: float = 6,
    ):
        """
        Initialize the AuthManager with credentials and configuration.
        
        Args:
            access_key: RagaAI Catalyst access key
            secret_key: RagaAI Catalyst secret key
            base_url: Optional custom base URL
            token_expiry_hours: Token expiration time in hours (default: 6)
        """
        cls._access_key = access_key
        cls._secret_key = secret_key
        cls.TOKEN_EXPIRY_TIME = token_expiry_hours * 60 * 60
        
        # Set environment variables for backward compatibility
        os.environ["RAGAAI_CATALYST_ACCESS_KEY"] = access_key
        os.environ["RAGAAI_CATALYST_SECRET_KEY"] = secret_key
        
        # Set base URL
        if base_url:
            cls.BASE_URL = base_url
        elif os.getenv("RAGAAI_CATALYST_BASE_URL"):
            cls.BASE_URL = os.getenv("RAGAAI_CATALYST_BASE_URL")
        else:
            cls.BASE_URL = "https://catalyst.raga.ai/api"
    
    @classmethod
    def _get_credentials(cls) -> tuple[str, str]:
        """
        Get access key and secret key from class variables or environment.
        
        Returns:
            Tuple of (access_key, secret_key)
        """
        access_key = cls._access_key or os.getenv("RAGAAI_CATALYST_ACCESS_KEY")
        secret_key = cls._secret_key or os.getenv("RAGAAI_CATALYST_SECRET_KEY")
        return access_key, secret_key
    
    @classmethod
    def _refresh_token_async(cls):
        """Refresh token in background thread."""
        try:
            cls.get_token(force_refresh=True)
        except Exception as e:
            logger.error(f"Background token refresh failed: {str(e)}")
    
    @classmethod
    def _schedule_token_refresh(cls):
        """Schedule a token refresh to happen 20 seconds before expiration."""
        if not cls._token_expiry:
            return
        
        # Calculate when to refresh (20 seconds before expiration)
        current_time = time.time()
        refresh_buffer = min(
            20, cls.TOKEN_EXPIRY_TIME * 0.05
        )  # 20 seconds or 5% of expiry time, whichever is smaller
        time_until_refresh = max(
            cls._token_expiry - current_time - refresh_buffer, 1
        )  # At least 1 second
        
        def delayed_refresh():
            # Sleep until it's time to refresh
            time.sleep(time_until_refresh)
            logger.debug("Scheduled token refresh triggered")
            cls._refresh_token_async()
        
        # Start a new thread for the delayed refresh
        if not cls._refresh_thread or not cls._refresh_thread.is_alive():
            cls._refresh_thread = threading.Thread(target=delayed_refresh)
            cls._refresh_thread.daemon = True
            cls._refresh_thread.start()
            logger.debug(f"Token refresh scheduled in {time_until_refresh:.1f} seconds")
    
    @classmethod
    def get_token(cls, force_refresh: bool = False) -> Union[str, None]:
        """
        Retrieves or refreshes a token using the provided credentials.
        
        Args:
            force_refresh: If True, forces a token refresh regardless of expiration
            
        Returns:
            A string representing the token if successful, None otherwise
        """
        with cls._token_refresh_lock:
            current_token = os.getenv("RAGAAI_CATALYST_TOKEN")
            current_time = time.time()
            
            # Check if we need to refresh the token
            if (
                not force_refresh
                and current_token
                and cls._token_expiry
                and current_time < cls._token_expiry
            ):
                return current_token
            
            access_key, secret_key = cls._get_credentials()
            if not access_key or not secret_key:
                logger.error("Access key or secret key is not set")
                return None
            
            headers = {"Content-Type": "application/json"}
            json_data = {"accessKey": access_key, "secretKey": secret_key}
            
            start_time = time.time()
            endpoint = f"{cls.BASE_URL}/token"
            
            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=json_data,
                    timeout=cls.TIMEOUT,
                )
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(
                    f"API Call: [POST] {endpoint} | Status: {response.status_code} | Time: {elapsed_ms:.2f}ms"
                )
                
                # Handle specific status codes before raising an error
                if response.status_code == 400:
                    token_response = response.json()
                    if token_response.get("message") == "Please enter valid credentials":
                        logger.error(
                            "Authentication failed. Invalid credentials provided. Please check your Access key and Secret key. \n"
                            "To view or create new keys, navigate to Settings -> Authenticate in the RagaAI Catalyst dashboard."
                        )
                
                response.raise_for_status()
                token_response = response.json()
                
                if not token_response.get("success", False):
                    logger.error(
                        "Token retrieval was not successful: %s",
                        token_response.get("message", "Unknown error"),
                    )
                    return None
                
                token = token_response.get("data", {}).get("token")
                if token:
                    os.environ["RAGAAI_CATALYST_TOKEN"] = token
                    cls._token_expiry = time.time() + cls.TOKEN_EXPIRY_TIME
                    logger.debug(
                        f"Token refreshed successfully. Next refresh in {cls.TOKEN_EXPIRY_TIME / 3600:.1f} hours"
                    )
                    
                    # Schedule token refresh 20 seconds before expiration
                    cls._schedule_token_refresh()
                    
                    return token
                else:
                    logger.error("Token not found in response")
                    return None
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Token request failed: {str(e)}")
                return None
    
    @classmethod
    def ensure_valid_token(cls) -> Union[str, None]:
        """
        Ensures a valid token is available.
        
        Handles different scenarios:
        - Missing token: Synchronous retrieval (fail fast)
        - Expired token: Synchronous refresh (since token is needed immediately)
        - Token approaching expiry: Background refresh while returning current token
        
        Returns:
            A string representing the valid token if successful, None otherwise
        """
        current_token = os.getenv("RAGAAI_CATALYST_TOKEN")
        current_time = time.time()
        
        # Case 1: No token - synchronous retrieval (fail fast)
        if not current_token:
            return cls.get_token(force_refresh=True)
        
        # Case 2: Token expired - synchronous refresh (since we need a valid token now)
        if not cls._token_expiry or current_time >= cls._token_expiry:
            logger.info("Token expired, refreshing synchronously")
            return cls.get_token(force_refresh=True)
        
        # Case 3: Token valid but approaching expiry (less than 10% of lifetime remaining)
        # Start background refresh but return current token
        token_remaining_time = cls._token_expiry - current_time
        if token_remaining_time < (cls.TOKEN_EXPIRY_TIME * 0.1):
            if not cls._refresh_thread or not cls._refresh_thread.is_alive():
                logger.info("Token approaching expiry, starting background refresh")
                cls._refresh_thread = threading.Thread(
                    target=cls._refresh_token_async
                )
                cls._refresh_thread.daemon = True
                cls._refresh_thread.start()
        
        # Return current token (which is valid)
        return current_token
    
    @classmethod
    def get_auth_header(cls) -> Dict[str, str]:
        """
        Returns a dictionary containing the Authorization header with a valid token.
        
        This method should be used instead of directly accessing os.getenv("RAGAAI_CATALYST_TOKEN").
        
        Returns:
            A dictionary with the Authorization header if successful, empty dict otherwise
        """
        token = cls.ensure_valid_token()
        if token:
            return {"Authorization": f"Bearer {token}"}
        return {}

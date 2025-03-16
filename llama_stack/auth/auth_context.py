# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
from typing import Dict, List, Optional, Any, Union
import httpx
import json
from contextlib import asynccontextmanager

from llama_stack.auth.types import (
    TokenInfo,
    AuthorizationRequest,
    AuthorizationResponse,
    TokenExchangeRequest,
    TokenExchangeResponse,
    PermissionDeniedError,
    InvalidTokenError,
)

logger = logging.getLogger(__name__)


class AuthContext:
    """
    Interface for validating tokens, checking permissions, and accessing authentication context.
    This class delegates to Authorino for authorization decisions and token validation.
    """

    def __init__(
        self,
        authorino_url: str,
        auth_token: Optional[str] = None,
        auth_config_name: Optional[str] = None,
    ):
        """
        Initialize the AuthContext.

        Args:
            authorino_url: URL of the Authorino service
            auth_token: Optional JWT token for authentication
            auth_config_name: Name of the AuthConfig resource in Authorino
        """
        self.authorino_url = authorino_url
        self.auth_token = auth_token
        self.auth_config_name = auth_config_name
        self.token_info: Optional[TokenInfo] = None
        self._http_client = httpx.AsyncClient()

    async def validate_token(self, token: str) -> TokenInfo:
        """
        Validate a JWT token with Authorino.

        Args:
            token: JWT token to validate

        Returns:
            TokenInfo containing the validated token information

        Raises:
            InvalidTokenError: If the token is invalid or expired
        """
        try:
            url = f"{self.authorino_url}/auth/validate"
            headers = {"Authorization": f"Bearer {token}"}
            
            if self.auth_config_name:
                headers["X-Authorino-AuthConfig"] = self.auth_config_name
                
            response = await self._http_client.post(url, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"Token validation failed: {response.text}")
                raise InvalidTokenError(f"Token validation failed: {response.status_code}")
                
            data = response.json()
            self.token_info = TokenInfo(
                subject=data.get("sub", ""),
                issuer=data.get("iss", ""),
                audience=data.get("aud", []),
                scopes=data.get("scope", "").split() if isinstance(data.get("scope"), str) else data.get("scope", []),
                roles=data.get("roles", []),
                metadata=data.get("metadata", {}),
                expiration=data.get("exp", 0),
                raw_token=token
            )
            return self.token_info
            
        except httpx.RequestError as e:
            logger.error(f"Error validating token: {str(e)}")
            raise InvalidTokenError(f"Error validating token: {str(e)}")

    async def check_permission(
        self, resource: str, action: str, context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if the current token has permission to perform an action on a resource.

        Args:
            resource: The resource being accessed
            action: The action being performed
            context: Additional context for the authorization decision

        Returns:
            True if authorized, False otherwise

        Raises:
            PermissionDeniedError: If the user does not have permission
            InvalidTokenError: If no token is available or the token is invalid
        """
        if not self.token_info and not self.auth_token:
            raise InvalidTokenError("No authentication token available")
            
        if not self.token_info and self.auth_token:
            await self.validate_token(self.auth_token)
            
        try:
            url = f"{self.authorino_url}/auth/authorize"
            headers = {
                "Authorization": f"Bearer {self.token_info.raw_token}",
                "Content-Type": "application/json"
            }
            
            if self.auth_config_name:
                headers["X-Authorino-AuthConfig"] = self.auth_config_name
                
            request = AuthorizationRequest(
                resource=resource,
                action=action,
                subject={
                    "id": self.token_info.subject,
                    "roles": self.token_info.roles,
                    "scopes": self.token_info.scopes,
                },
                context=context or {}
            )
            
            response = await self._http_client.post(
                url, 
                headers=headers,
                content=request.model_dump_json()
            )
            
            if response.status_code != 200:
                logger.error(f"Authorization check failed: {response.text}")
                raise PermissionDeniedError(f"Authorization check failed: {response.status_code}")
                
            auth_response = AuthorizationResponse(**response.json())
            
            if not auth_response.allowed:
                raise PermissionDeniedError(auth_response.reason or "Permission denied")
                
            return True
            
        except httpx.RequestError as e:
            logger.error(f"Error checking permission: {str(e)}")
            raise PermissionDeniedError(f"Error checking permission: {str(e)}")

    async def exchange_token(
        self, 
        audience: str, 
        scope: Optional[str] = None,
        resource: Optional[Union[str, List[str]]] = None
    ) -> str:
        """
        Exchange the current token for a downscoped token.

        Args:
            audience: The intended audience for the new token
            scope: Optional space-separated list of scopes for the new token
            resource: Optional resource(s) the token should be limited to

        Returns:
            A downscoped JWT token

        Raises:
            InvalidTokenError: If no token is available or the token is invalid
        """
        if not self.token_info and not self.auth_token:
            raise InvalidTokenError("No authentication token available")
            
        if not self.token_info and self.auth_token:
            await self.validate_token(self.auth_token)
            
        try:
            url = f"{self.authorino_url}/auth/token"
            headers = {
                "Content-Type": "application/json"
            }
            
            if self.auth_config_name:
                headers["X-Authorino-AuthConfig"] = self.auth_config_name
                
            request = TokenExchangeRequest(
                subject_token=self.token_info.raw_token,
                audience=audience,
                scope=scope,
                resource=resource
            )
            
            response = await self._http_client.post(
                url, 
                headers=headers,
                content=request.model_dump_json()
            )
            
            if response.status_code != 200:
                logger.error(f"Token exchange failed: {response.text}")
                raise InvalidTokenError(f"Token exchange failed: {response.status_code}")
                
            exchange_response = TokenExchangeResponse(**response.json())
            return exchange_response.access_token
            
        except httpx.RequestError as e:
            logger.error(f"Error exchanging token: {str(e)}")
            raise InvalidTokenError(f"Error exchanging token: {str(e)}")

    @asynccontextmanager
    async def ensure_permission(self, resource: str, action: str, context: Optional[Dict[str, Any]] = None):
        """
        Context manager that ensures the user has permission to perform an action.

        Args:
            resource: The resource being accessed
            action: The action being performed
            context: Additional context for the authorization decision

        Raises:
            PermissionDeniedError: If the user does not have permission
        """
        await self.check_permission(resource, action, context)
        try:
            yield
        finally:
            pass  # No cleanup needed

    def get_subject_id(self) -> str:
        """Get the subject ID from the token."""
        if not self.token_info:
            raise InvalidTokenError("No validated token available")
        return self.token_info.subject

    def get_roles(self) -> List[str]:
        """Get the roles from the token."""
        if not self.token_info:
            raise InvalidTokenError("No validated token available")
        return self.token_info.roles

    def get_scopes(self) -> List[str]:
        """Get the scopes from the token."""
        if not self.token_info:
            raise InvalidTokenError("No validated token available")
        return self.token_info.scopes

    def get_metadata(self) -> Dict[str, Any]:
        """Get the metadata from the token."""
        if not self.token_info:
            raise InvalidTokenError("No validated token available")
        return self.token_info.metadata

    async def close(self):
        """Close the HTTP client."""
        await self._http_client.aclose()

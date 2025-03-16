# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel

from llama_stack.auth.types import TokenInfo


class AuthenticationRequest(BaseModel):
    """Request to authenticate with a provider."""
    provider_id: str
    credentials: Dict[str, Any]


class AuthenticationResponse(BaseModel):
    """Response from authentication."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_in: int = 3600


class TokenValidationRequest(BaseModel):
    """Request to validate a token."""
    token: str
    provider_id: Optional[str] = None


class TokenValidationResponse(BaseModel):
    """Response from token validation."""
    valid: bool
    token_info: Optional[TokenInfo] = None
    error: Optional[str] = None


class TokenRefreshRequest(BaseModel):
    """Request to refresh a token."""
    refresh_token: str
    provider_id: Optional[str] = None


class TokenRefreshResponse(BaseModel):
    """Response from token refresh."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_in: int = 3600


class TokenExchangeRequest(BaseModel):
    """Request to exchange a token."""
    token: str
    audience: str
    scope: Optional[str] = None
    resource: Optional[Union[str, List[str]]] = None


class TokenExchangeResponse(BaseModel):
    """Response from token exchange."""
    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600
    scope: Optional[str] = None


class AuthorizationRequest(BaseModel):
    """Request to check authorization."""
    resource: str
    action: str
    token: str
    context: Optional[Dict[str, Any]] = None


class AuthorizationResponse(BaseModel):
    """Response from authorization check."""
    allowed: bool
    reason: Optional[str] = None


class Auth:
    """
    Authentication and authorization API.
    This API provides endpoints for authentication, token validation, token refresh,
    token exchange, and authorization checks.
    """

    async def authenticate(self, request: AuthenticationRequest) -> AuthenticationResponse:
        """
        Authenticate with a provider.

        Args:
            request: Authentication request

        Returns:
            Authentication response with tokens
        """
        raise NotImplementedError("This method should be implemented by a provider")

    async def validate_token(self, request: TokenValidationRequest) -> TokenValidationResponse:
        """
        Validate a token.

        Args:
            request: Token validation request

        Returns:
            Token validation response
        """
        raise NotImplementedError("This method should be implemented by a provider")

    async def refresh_token(self, request: TokenRefreshRequest) -> TokenRefreshResponse:
        """
        Refresh a token.

        Args:
            request: Token refresh request

        Returns:
            Token refresh response with new tokens
        """
        raise NotImplementedError("This method should be implemented by a provider")

    async def exchange_token(self, request: TokenExchangeRequest) -> TokenExchangeResponse:
        """
        Exchange a token for a downscoped token.

        Args:
            request: Token exchange request

        Returns:
            Token exchange response with new token
        """
        raise NotImplementedError("This method should be implemented by a provider")

    async def check_authorization(self, request: AuthorizationRequest) -> AuthorizationResponse:
        """
        Check if a token has permission to perform an action on a resource.

        Args:
            request: Authorization request

        Returns:
            Authorization response
        """
        raise NotImplementedError("This method should be implemented by a provider")

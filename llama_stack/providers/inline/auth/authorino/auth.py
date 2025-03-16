# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
from typing import Dict, List, Optional, Any, Union, Type

from llama_stack.apis.auth import (
    Auth,
    AuthenticationRequest,
    AuthenticationResponse,
    TokenValidationRequest,
    TokenValidationResponse,
    TokenRefreshRequest,
    TokenRefreshResponse,
    TokenExchangeRequest,
    TokenExchangeResponse,
    AuthorizationRequest,
    AuthorizationResponse,
)
from llama_stack.auth.auth_context import AuthContext
from llama_stack.auth.auth_provider import AuthProvider, KeycloakAuthProvider, SpireAuthProvider
from llama_stack.auth.config import AuthConfig, AuthProviderConfig
from llama_stack.auth.types import InvalidTokenError, PermissionDeniedError, TokenInfo
from llama_stack.providers.inline.auth.authorino.config import AuthorinoProviderConfig

logger = logging.getLogger(__name__)


class AuthorinoAuth(Auth):
    """
    Authorino-based implementation of the Auth API.
    """

    def __init__(self, config: AuthorinoProviderConfig):
        """
        Initialize the Authorino auth provider.

        Args:
            config: Authorino provider configuration
        """
        self.config = config
        self.auth_providers: Dict[str, AuthProvider] = {}
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize authentication providers from configuration."""
        for provider_id, provider_config in self.config.providers.items():
            provider_type = provider_config.provider_type
            if provider_type == "keycloak":
                self.auth_providers[provider_id] = KeycloakAuthProvider(provider_config.config)
            elif provider_type == "spire":
                self.auth_providers[provider_id] = SpireAuthProvider(provider_config.config)
            else:
                logger.warning(f"Unknown auth provider type: {provider_type}")

    def _get_provider(self, provider_id: Optional[str] = None) -> AuthProvider:
        """
        Get an authentication provider by ID.

        Args:
            provider_id: Provider ID, or None to use the default provider

        Returns:
            The authentication provider

        Raises:
            ValueError: If the provider is not found
        """
        if not provider_id:
            provider_id = self.config.default_provider

        if not provider_id or provider_id not in self.auth_providers:
            available_providers = ", ".join(self.auth_providers.keys())
            raise ValueError(
                f"Provider '{provider_id}' not found. Available providers: {available_providers}"
            )

        return self.auth_providers[provider_id]

    def _create_auth_context(self, token: Optional[str] = None) -> AuthContext:
        """
        Create an AuthContext instance.

        Args:
            token: Optional token to initialize the context with

        Returns:
            An AuthContext instance
        """
        return AuthContext(
            authorino_url=self.config.url,
            auth_token=token,
            auth_config_name=self.config.auth_config_name,
        )

    async def authenticate(self, request: AuthenticationRequest) -> AuthenticationResponse:
        """
        Authenticate with a provider.

        Args:
            request: Authentication request

        Returns:
            Authentication response with tokens
        """
        try:
            provider = self._get_provider(request.provider_id)
            token = await provider.authenticate(request.credentials)

            # For providers that support refresh tokens
            refresh_token = ""
            if hasattr(provider, "get_refresh_token"):
                refresh_token = await provider.get_refresh_token()

            return AuthenticationResponse(
                access_token=token,
                refresh_token=refresh_token,
                token_type="Bearer",
                expires_in=3600,  # Default expiration
            )
        except InvalidTokenError as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during authentication: {str(e)}")
            raise InvalidTokenError(f"Authentication failed: {str(e)}")

    async def validate_token(self, request: TokenValidationRequest) -> TokenValidationResponse:
        """
        Validate a token.

        Args:
            request: Token validation request

        Returns:
            Token validation response
        """
        try:
            auth_context = self._create_auth_context()
            token_info = await auth_context.validate_token(request.token)
            return TokenValidationResponse(
                valid=True,
                token_info=token_info,
            )
        except InvalidTokenError as e:
            logger.warning(f"Token validation failed: {str(e)}")
            return TokenValidationResponse(
                valid=False,
                error=str(e),
            )
        except Exception as e:
            logger.error(f"Unexpected error during token validation: {str(e)}")
            return TokenValidationResponse(
                valid=False,
                error=f"Token validation failed: {str(e)}",
            )
        finally:
            if auth_context:
                await auth_context.close()

    async def refresh_token(self, request: TokenRefreshRequest) -> TokenRefreshResponse:
        """
        Refresh a token.

        Args:
            request: Token refresh request

        Returns:
            Token refresh response with new tokens
        """
        try:
            provider = self._get_provider(request.provider_id)
            token_data = await provider.refresh_token(request.refresh_token)
            return TokenRefreshResponse(
                access_token=token_data["access_token"],
                refresh_token=token_data.get("refresh_token"),
                token_type="Bearer",
                expires_in=token_data.get("expires_in", 3600),
            )
        except InvalidTokenError as e:
            logger.error(f"Token refresh failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during token refresh: {str(e)}")
            raise InvalidTokenError(f"Token refresh failed: {str(e)}")

    async def exchange_token(self, request: TokenExchangeRequest) -> TokenExchangeResponse:
        """
        Exchange a token for a downscoped token.

        Args:
            request: Token exchange request

        Returns:
            Token exchange response with new token
        """
        try:
            auth_context = self._create_auth_context(request.token)
            downscoped_token = await auth_context.exchange_token(
                audience=request.audience,
                scope=request.scope,
                resource=request.resource,
            )
            return TokenExchangeResponse(
                access_token=downscoped_token,
                token_type="Bearer",
                expires_in=3600,  # Default expiration
                scope=request.scope,
            )
        except InvalidTokenError as e:
            logger.error(f"Token exchange failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during token exchange: {str(e)}")
            raise InvalidTokenError(f"Token exchange failed: {str(e)}")
        finally:
            if auth_context:
                await auth_context.close()

    async def check_authorization(self, request: AuthorizationRequest) -> AuthorizationResponse:
        """
        Check if a token has permission to perform an action on a resource.

        Args:
            request: Authorization request

        Returns:
            Authorization response
        """
        try:
            auth_context = self._create_auth_context(request.token)
            await auth_context.check_permission(
                resource=request.resource,
                action=request.action,
                context=request.context,
            )
            return AuthorizationResponse(
                allowed=True,
            )
        except PermissionDeniedError as e:
            logger.warning(f"Permission denied: {str(e)}")
            return AuthorizationResponse(
                allowed=False,
                reason=str(e),
            )
        except InvalidTokenError as e:
            logger.warning(f"Invalid token during authorization check: {str(e)}")
            return AuthorizationResponse(
                allowed=False,
                reason=f"Invalid token: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Unexpected error during authorization check: {str(e)}")
            return AuthorizationResponse(
                allowed=False,
                reason=f"Authorization check failed: {str(e)}",
            )
        finally:
            if auth_context:
                await auth_context.close()


def get_provider_impl(config: AuthorinoProviderConfig, deps: Dict[str, Any]) -> Auth:
    """
    Get the Authorino auth provider implementation.

    Args:
        config: Authorino provider configuration
        deps: Dependencies

    Returns:
        The auth provider implementation
    """
    return AuthorinoAuth(config)

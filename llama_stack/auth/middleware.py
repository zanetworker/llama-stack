# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
import re
from typing import Callable, Dict, List, Optional, Any, Union

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from llama_stack.auth.auth_context import AuthContext
from llama_stack.auth.config import AuthConfig
from llama_stack.auth.types import InvalidTokenError, PermissionDeniedError

logger = logging.getLogger(__name__)


class TokenExchangeMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware to handle token validation, interaction with Authorino,
    and context creation.
    """

    def __init__(
        self,
        app: ASGIApp,
        auth_config: AuthConfig,
        auth_context_factory: Callable[[str, Optional[str]], AuthContext] = None,
    ):
        """
        Initialize the middleware.

        Args:
            app: The ASGI application
            auth_config: Authentication and authorization configuration
            auth_context_factory: Optional factory function to create AuthContext instances
        """
        super().__init__(app)
        self.auth_config = auth_config
        self.auth_context_factory = auth_context_factory or self._default_auth_context_factory
        self._compile_path_patterns()

    def _compile_path_patterns(self):
        """Compile path patterns for protected and public paths."""
        self.protected_patterns = [re.compile(pattern) for pattern in self.auth_config.protected_paths]
        self.public_patterns = [re.compile(pattern) for pattern in self.auth_config.public_paths]

    def _default_auth_context_factory(self, token: str, provider_id: Optional[str] = None) -> AuthContext:
        """
        Default factory function to create AuthContext instances.

        Args:
            token: The authentication token
            provider_id: Optional provider ID

        Returns:
            An AuthContext instance
        """
        return AuthContext(
            authorino_url=self.auth_config.authorino.url,
            auth_token=token,
            auth_config_name=self.auth_config.authorino.auth_config_name,
        )

    def _is_path_protected(self, path: str) -> bool:
        """
        Check if a path is protected.

        Args:
            path: The request path

        Returns:
            True if the path is protected, False otherwise
        """
        # If no patterns are defined, default to protecting all paths
        if not self.protected_patterns and not self.public_patterns:
            return True

        # Check if the path matches any public pattern
        for pattern in self.public_patterns:
            if pattern.match(path):
                return False

        # Check if the path matches any protected pattern
        for pattern in self.protected_patterns:
            if pattern.match(path):
                return True

        # If protected patterns are defined but none match, default to not protected
        return len(self.protected_patterns) == 0

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request.

        Args:
            request: The incoming request
            call_next: Function to call the next middleware or route handler

        Returns:
            The response
        """
        # Check if the path is protected
        if not self._is_path_protected(request.url.path):
            return await call_next(request)

        # Extract token from header
        auth_header = request.headers.get(self.auth_config.token_header)
        if not auth_header:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required"},
            )

        # Check token format
        if self.auth_config.token_prefix:
            if not auth_header.startswith(f"{self.auth_config.token_prefix} "):
                return JSONResponse(
                    status_code=401,
                    content={"detail": f"Invalid token format. Expected: {self.auth_config.token_prefix} <token>"},
                )
            token = auth_header[len(self.auth_config.token_prefix) + 1:]
        else:
            token = auth_header

        # Get provider ID from header or use default
        provider_id = request.headers.get("X-Auth-Provider") or self.auth_config.default_provider

        try:
            # Create auth context
            auth_context = self.auth_context_factory(token, provider_id)

            # Validate token
            await auth_context.validate_token(token)

            # Add auth context to request state
            request.state.auth_context = auth_context

            # Process the request
            response = await call_next(request)

            return response

        except InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            return JSONResponse(
                status_code=401,
                content={"detail": f"Invalid token: {str(e)}"},
            )
        except PermissionDeniedError as e:
            logger.warning(f"Permission denied: {str(e)}")
            return JSONResponse(
                status_code=403,
                content={"detail": f"Permission denied: {str(e)}"},
            )
        except Exception as e:
            logger.error(f"Error in auth middleware: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error during authentication"},
            )


def setup_auth_middleware(app: FastAPI, auth_config: AuthConfig):
    """
    Set up the authentication middleware.

    Args:
        app: The FastAPI application
        auth_config: Authentication and authorization configuration
    """
    app.add_middleware(TokenExchangeMiddleware, auth_config=auth_config)

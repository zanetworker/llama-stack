# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel


class InvalidTokenError(Exception):
    """Raised when a token is invalid or expired."""
    pass


class PermissionDeniedError(Exception):
    """Raised when a user does not have permission to access a resource."""
    pass


class AuthorizationRequest(BaseModel):
    """Request to check authorization with Authorino."""
    resource: str
    action: str
    subject: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


class AuthorizationResponse(BaseModel):
    """Response from Authorino authorization check."""
    allowed: bool
    reason: Optional[str] = None
    downscoped_token: Optional[str] = None


class TokenInfo(BaseModel):
    """Information extracted from a validated token."""
    subject: str
    issuer: str
    audience: List[str]
    scopes: List[str]
    roles: List[str]
    metadata: Dict[str, Any]
    expiration: int
    raw_token: str


class TokenExchangeRequest(BaseModel):
    """Request to exchange a token for a downscoped token."""
    subject_token: str
    subject_token_type: str = "urn:ietf:params:oauth:token-type:jwt"
    requested_token_type: str = "urn:ietf:params:oauth:token-type:jwt"
    audience: str
    scope: Optional[str] = None
    resource: Optional[Union[str, List[str]]] = None


class TokenExchangeResponse(BaseModel):
    """Response from a token exchange request."""
    access_token: str
    issued_token_type: str
    token_type: str
    expires_in: int
    scope: Optional[str] = None

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class AuthorinoConfig(BaseModel):
    """Configuration for Authorino authorization service."""
    url: str = Field(..., description="URL of the Authorino service")
    auth_config_name: Optional[str] = Field(
        None, 
        description="Name of the AuthConfig resource in Authorino"
    )


class AuthProviderConfig(BaseModel):
    """Base configuration for authentication providers."""
    provider_type: str = Field(..., description="Type of authentication provider")
    config: Dict[str, Any] = Field(..., description="Provider-specific configuration")


class AuthConfig(BaseModel):
    """Configuration for authentication and authorization."""
    authorino: AuthorinoConfig = Field(
        ..., 
        description="Configuration for Authorino authorization service"
    )
    providers: Dict[str, AuthProviderConfig] = Field(
        ..., 
        description="Map of authentication provider configurations"
    )
    default_provider: Optional[str] = Field(
        None, 
        description="Default authentication provider to use"
    )
    token_header: str = Field(
        "Authorization", 
        description="HTTP header containing the authentication token"
    )
    token_prefix: str = Field(
        "Bearer", 
        description="Prefix for the authentication token in the header"
    )
    enable_token_exchange: bool = Field(
        True, 
        description="Whether to enable token exchange for downscoping"
    )
    protected_paths: List[str] = Field(
        default_factory=list,
        description="List of API paths that require authentication"
    )
    public_paths: List[str] = Field(
        default_factory=list,
        description="List of API paths that do not require authentication"
    )

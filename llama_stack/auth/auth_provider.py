# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union

import httpx
import jwt
from pydantic import BaseModel

from llama_stack.auth.types import TokenInfo, InvalidTokenError

logger = logging.getLogger(__name__)


class AuthProvider(ABC):
    """
    Base class for authentication providers.
    Implementations handle the specifics of authenticating with different identity providers.
    """

    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> str:
        """
        Authenticate with the provider and return a token.

        Args:
            credentials: Provider-specific credentials

        Returns:
            A JWT token

        Raises:
            InvalidTokenError: If authentication fails
        """
        pass

    @abstractmethod
    async def validate_token(self, token: str) -> TokenInfo:
        """
        Validate a token and extract its information.

        Args:
            token: The token to validate

        Returns:
            TokenInfo containing the validated token information

        Raises:
            InvalidTokenError: If the token is invalid or expired
        """
        pass

    @abstractmethod
    async def refresh_token(self, refresh_token: str) -> Dict[str, str]:
        """
        Refresh an expired token.

        Args:
            refresh_token: The refresh token

        Returns:
            A dictionary containing the new access token and refresh token

        Raises:
            InvalidTokenError: If the refresh token is invalid or expired
        """
        pass


class KeycloakConfig(BaseModel):
    """Configuration for Keycloak authentication provider."""
    realm_url: str
    client_id: str
    client_secret: Optional[str] = None


class KeycloakAuthProvider(AuthProvider):
    """
    Authentication provider for Keycloak (OAuth 2.0/OIDC).
    """

    def __init__(self, config: KeycloakConfig):
        """
        Initialize the Keycloak authentication provider.

        Args:
            config: Keycloak configuration
        """
        self.config = config
        self.http_client = httpx.AsyncClient()
        self.token_url = f"{config.realm_url}/protocol/openid-connect/token"
        self.userinfo_url = f"{config.realm_url}/protocol/openid-connect/userinfo"
        self.jwks_url = f"{config.realm_url}/protocol/openid-connect/certs"
        self.jwks = None

    async def authenticate(self, credentials: Dict[str, Any]) -> str:
        """
        Authenticate with Keycloak using username/password or other grant types.

        Args:
            credentials: Dictionary containing authentication details
                - grant_type: The OAuth 2.0 grant type (password, client_credentials, etc.)
                - username: Username (for password grant)
                - password: Password (for password grant)
                - scope: Optional space-separated list of scopes

        Returns:
            A JWT access token

        Raises:
            InvalidTokenError: If authentication fails
        """
        try:
            grant_type = credentials.get("grant_type", "password")
            
            data = {
                "client_id": self.config.client_id,
                "grant_type": grant_type,
            }
            
            if self.config.client_secret:
                data["client_secret"] = self.config.client_secret
                
            if grant_type == "password":
                data["username"] = credentials.get("username")
                data["password"] = credentials.get("password")
            elif grant_type == "refresh_token":
                data["refresh_token"] = credentials.get("refresh_token")
                
            if "scope" in credentials:
                data["scope"] = credentials.get("scope")
                
            response = await self.http_client.post(
                self.token_url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            if response.status_code != 200:
                logger.error(f"Keycloak authentication failed: {response.text}")
                raise InvalidTokenError(f"Authentication failed: {response.status_code}")
                
            token_data = response.json()
            return token_data["access_token"]
            
        except httpx.RequestError as e:
            logger.error(f"Error authenticating with Keycloak: {str(e)}")
            raise InvalidTokenError(f"Error authenticating with Keycloak: {str(e)}")

    async def _get_jwks(self):
        """Get the JSON Web Key Set from Keycloak."""
        if self.jwks is None:
            response = await self.http_client.get(self.jwks_url)
            if response.status_code != 200:
                raise InvalidTokenError(f"Failed to get JWKS: {response.status_code}")
            self.jwks = response.json()
        return self.jwks

    async def validate_token(self, token: str) -> TokenInfo:
        """
        Validate a JWT token from Keycloak.

        Args:
            token: The JWT token to validate

        Returns:
            TokenInfo containing the validated token information

        Raises:
            InvalidTokenError: If the token is invalid or expired
        """
        try:
            # Get the unverified header to find the key ID
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            
            if not kid:
                raise InvalidTokenError("Token header missing key ID (kid)")
                
            # Get the JWKS
            jwks = await self._get_jwks()
            
            # Find the matching key
            key = None
            for k in jwks.get("keys", []):
                if k.get("kid") == kid:
                    key = k
                    break
                    
            if not key:
                raise InvalidTokenError(f"No matching key found for kid: {kid}")
                
            # Construct the public key
            public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)
            
            # Verify the token
            payload = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                audience=self.config.client_id,
                options={"verify_exp": True}
            )
            
            # Extract token information
            return TokenInfo(
                subject=payload.get("sub", ""),
                issuer=payload.get("iss", ""),
                audience=[payload.get("aud")] if isinstance(payload.get("aud"), str) else payload.get("aud", []),
                scopes=payload.get("scope", "").split() if isinstance(payload.get("scope"), str) else payload.get("scope", []),
                roles=payload.get("realm_access", {}).get("roles", []),
                metadata=payload,
                expiration=payload.get("exp", 0),
                raw_token=token
            )
            
        except jwt.PyJWTError as e:
            logger.error(f"Error validating token: {str(e)}")
            raise InvalidTokenError(f"Invalid token: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error validating token: {str(e)}")
            raise InvalidTokenError(f"Error validating token: {str(e)}")

    async def refresh_token(self, refresh_token: str) -> Dict[str, str]:
        """
        Refresh an expired token.

        Args:
            refresh_token: The refresh token

        Returns:
            A dictionary containing the new access token and refresh token

        Raises:
            InvalidTokenError: If the refresh token is invalid or expired
        """
        try:
            data = {
                "client_id": self.config.client_id,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            }
            
            if self.config.client_secret:
                data["client_secret"] = self.config.client_secret
                
            response = await self.http_client.post(
                self.token_url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            if response.status_code != 200:
                logger.error(f"Token refresh failed: {response.text}")
                raise InvalidTokenError(f"Token refresh failed: {response.status_code}")
                
            token_data = response.json()
            return {
                "access_token": token_data["access_token"],
                "refresh_token": token_data["refresh_token"],
                "expires_in": token_data["expires_in"]
            }
            
        except httpx.RequestError as e:
            logger.error(f"Error refreshing token: {str(e)}")
            raise InvalidTokenError(f"Error refreshing token: {str(e)}")

    async def close(self):
        """Close the HTTP client."""
        await self.http_client.aclose()


class SpireConfig(BaseModel):
    """Configuration for SPIRE authentication provider."""
    socket_path: str = "/tmp/spire-agent.sock"


class SpireAuthProvider(AuthProvider):
    """
    Authentication provider for SPIFFE/SPIRE for workload identity.
    """

    def __init__(self, config: SpireConfig):
        """
        Initialize the SPIRE authentication provider.

        Args:
            config: SPIRE configuration
        """
        self.config = config
        # Import here to avoid dependency issues if SPIRE is not used
        try:
            import spiffe.workloadapi
            self.spiffe = spiffe.workloadapi
        except ImportError:
            logger.error("SPIFFE Python SDK not installed. Install with 'pip install pyspiffe'")
            raise ImportError("SPIFFE Python SDK not installed. Install with 'pip install pyspiffe'")

    async def authenticate(self, credentials: Dict[str, Any]) -> str:
        """
        Get a JWT-SVID from the SPIRE agent.

        Args:
            credentials: Not used for SPIRE

        Returns:
            A JWT-SVID token

        Raises:
            InvalidTokenError: If authentication fails
        """
        try:
            # Create a new workload API client
            client = self.spiffe.WorkloadApiClient(self.config.socket_path)
            
            # Fetch JWT-SVID
            jwt_svid = client.fetch_jwt_svid(audience=["llama-stack"])
            
            # Return the token
            return jwt_svid.token
            
        except Exception as e:
            logger.error(f"Error getting JWT-SVID from SPIRE: {str(e)}")
            raise InvalidTokenError(f"Error getting JWT-SVID from SPIRE: {str(e)}")

    async def validate_token(self, token: str) -> TokenInfo:
        """
        Validate a JWT-SVID token.

        Args:
            token: The JWT-SVID token to validate

        Returns:
            TokenInfo containing the validated token information

        Raises:
            InvalidTokenError: If the token is invalid or expired
        """
        try:
            # Create a new workload API client
            client = self.spiffe.WorkloadApiClient(self.config.socket_path)
            
            # Get the bundle for validation
            bundle = client.fetch_x509_bundle()
            
            # Validate the JWT-SVID
            jwt_bundle = self.spiffe.jwtbundle.JwtBundle(bundle.trust_domain, {})
            jwt_svid = self.spiffe.jwtsvid.JwtSvid.parse_insecure(token, ["llama-stack"])
            
            # Extract token information
            payload = jwt.decode(token, options={"verify_signature": False})
            
            return TokenInfo(
                subject=jwt_svid.spiffe_id.to_string(),
                issuer=payload.get("iss", ""),
                audience=jwt_svid.audiences,
                scopes=[],  # SPIRE doesn't use scopes in the same way
                roles=[],   # SPIRE doesn't use roles in the same way
                metadata=payload,
                expiration=payload.get("exp", 0),
                raw_token=token
            )
            
        except Exception as e:
            logger.error(f"Error validating JWT-SVID: {str(e)}")
            raise InvalidTokenError(f"Invalid JWT-SVID: {str(e)}")

    async def refresh_token(self, refresh_token: str) -> Dict[str, str]:
        """
        SPIRE doesn't use refresh tokens. Just get a new JWT-SVID.

        Args:
            refresh_token: Not used

        Returns:
            A dictionary containing the new access token

        Raises:
            InvalidTokenError: If getting a new JWT-SVID fails
        """
        try:
            # Just get a new token
            token = await self.authenticate({})
            return {
                "access_token": token,
                "refresh_token": "",  # No refresh token for SPIRE
                "expires_in": 3600    # Default expiration
            }
            
        except Exception as e:
            logger.error(f"Error refreshing JWT-SVID: {str(e)}")
            raise InvalidTokenError(f"Error refreshing JWT-SVID: {str(e)}")

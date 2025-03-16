# Llama Stack Authentication and Authorization

This module provides a zero-trust access control system for Llama Stack using Authorino as the core authorization engine. It enhances security across all components (vector databases, inference services, agents, tools) by ensuring that every request is explicitly authorized.

## Features

- **Zero-Trust Security Model**: Every request is authenticated and authorized
- **Industry Standards**: Leverages OAuth 2.0, OIDC, and SPIFFE for authentication
- **Multi-Tenancy Support**: Securely isolate data and resources for different users or organizations
- **Least Privilege Enforcement**: Agents and services have only necessary permissions
- **Compliance Ready**: Helps meet requirements for SOC2, HIPAA, GDPR, etc.
- **Detailed Audit Trails**: Track all access attempts and authorization decisions

## Architecture

The auth system consists of the following components:

1. **AuthContext**: Interface for validating tokens, checking permissions, and accessing context
2. **AuthProvider**: Interface for authentication providers (Keycloak, SPIRE)
3. **TokenExchangeMiddleware**: FastAPI middleware for token validation and context creation
4. **Auth API**: Endpoints for authentication, token validation, and authorization checks

## Configuration

### Build.yaml Configuration

To enable authentication and authorization in your Llama Stack deployment, add the auth provider to your `build.yaml` file:

```yaml
version: '2'
distribution_spec:
  providers:
    auth:
      - provider_id: authorino
        provider_type: inline::authorino
        config:
          url: ${env.AUTHORINO_URL}
          auth_config_name: llama-stack-auth
          providers:
            keycloak:
              provider_type: keycloak
              config:
                realm_url: ${env.KEYCLOAK_REALM_URL}
                client_id: ${env.KEYCLOAK_CLIENT_ID}
                client_secret: ${env.KEYCLOAK_CLIENT_SECRET}
          default_provider: keycloak
          protected_paths:
            - "^/agents/.*"
            - "^/vector_dbs/.*"
          public_paths:
            - "^/auth/.*"
            - "^/health$"
```

### Authorino AuthConfig

Create an Authorino AuthConfig resource to define your authorization policies:

```yaml
apiVersion: authorino.kuadrant.io/v1beta3
kind: AuthConfig
metadata:
  name: llama-stack-auth
  namespace: llama-stack
spec:
  hosts:
    - llama-stack-service.example.com

  authentication:
    "keycloak-auth":
      jwt:
        issuerUrl: "https://your-keycloak-server/auth/realms/your-realm"
        audiences:
          - "llama-stack-client"

  authorization:
    "rbac-policy":
      patternMatching:
        patterns:
          - selector: "auth.identity.metadata.realm_access.roles"
            operator: incl
            value: "data-scientist"
```

## Client Usage

### Basic Authentication

```python
from llama_stack_client import LlamaStackClient

# Initialize the client with authentication
client = LlamaStackClient(
    base_url="http://localhost:8080",
    auth_config={
        "provider": "keycloak",
        "token": "your-jwt-token"
    }
)

# Use the client as usual
results = client.vector_dbs.query(
    vector_db_id="secure_docs",
    query="sensitive information",
    top_k=5
)
```

### Agent with Restricted Tools

```python
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent

# Initialize client with auth
client = LlamaStackClient(
    base_url="http://localhost:8080",
    auth_config={
        "provider": "keycloak",
        "token": "your-jwt-token",
        "scopes": ["agent:execute", "tools:read"]
    }
)

# Create agent with auth context
agent = Agent(
    client,
    model="meta-llama/Llama-3-70b-chat",
    instructions="You are a helpful assistant that can use tools to answer questions.",
    tools=["builtin::code_interpreter", "builtin::rag/knowledge_search"],
    auth_context={
        "allowed_tools": ["builtin::code_interpreter"],  # Tool-level restrictions
        "data_access_level": "restricted"
    }
)

# Create session and execute with auth checks
session = agent.create_session("secure-session")
response = agent.create_turn(
    session,
    [{"role": "user", "content": "Search our codebase for auth examples"}],
)
```

## Error Handling

The auth module provides two main exception types:

- `InvalidTokenError`: Raised when a token is invalid or expired
- `PermissionDeniedError`: Raised when a user does not have permission to access a resource

Example error handling:

```python
from llama_stack.auth.types import PermissionDeniedError, InvalidTokenError

try:
    # Authenticated API call
    results = client.vector_dbs.query(...)
except PermissionDeniedError as e:
    print(f"Permission denied: {e}")
except InvalidTokenError as e:
    print(f"Invalid token: {e}")
```

## Extending the Auth System

### Adding a New Authentication Provider

1. Create a new class that implements the `AuthProvider` interface
2. Register the provider in the `_initialize_providers` method of `AuthorinoAuth`
3. Update the build.yaml configuration to include the new provider

### Adding New Authorization Policies

1. Update the Authorino AuthConfig resource to include new policies
2. Use the `check_permission` method of `AuthContext` to enforce the policies in your code

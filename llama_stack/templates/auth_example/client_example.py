#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""
Example client code demonstrating how to use the Llama Stack client with authentication.
"""

import asyncio
import os
from typing import Dict, List, Optional, Any

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack.auth.types import PermissionDeniedError, InvalidTokenError


async def vector_db_example(client: LlamaStackClient, auth_token: str):
    """
    Example of accessing a vector database with RBAC.
    
    Args:
        client: The Llama Stack client
        auth_token: JWT token for authentication
    """
    try:
        # Initialize the client with authentication configuration
        client = LlamaStackClient(
            base_url="http://localhost:8080",
            auth_config={
                "provider": "keycloak",  # Use the 'keycloak' provider from build.yaml
                "token": auth_token
            }
        )

        # Register a vector database
        client.vector_dbs.register(
            vector_db_id="secure_docs",
            embedding_model="all-MiniLM-L6-v2",
            embedding_dimension=384,
            provider_id="faiss"
        )

        # Query the vector database
        results = client.vector_dbs.query(
            vector_db_id="secure_docs",
            query="sensitive information",
            top_k=5
        )
        
        print("Vector DB query results:", results)
        
    except PermissionDeniedError as e:
        print(f"Permission denied: {e}")
    except InvalidTokenError as e:
        print(f"Invalid token: {e}")
    except Exception as e:
        print(f"Error: {e}")


async def agent_with_restricted_tools_example(client: LlamaStackClient, auth_token: str):
    """
    Example of using an agent with restricted tools.
    
    Args:
        client: The Llama Stack client
        auth_token: JWT token for authentication
    """
    try:
        # Initialize the client with authentication configuration
        client = LlamaStackClient(
            base_url="http://localhost:8080",
            auth_config={
                "provider": "keycloak",
                "token": auth_token,
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

        # Create session
        session = agent.create_session("secure-session")

        # Execute with auth checks
        response = agent.create_turn(
            session,
            [{"role": "user", "content": "Search our codebase for auth examples"}],
        )
        
        print("Agent response:", response)
        
    except PermissionDeniedError as e:
        print(f"Permission denied: {e}")
    except InvalidTokenError as e:
        print(f"Invalid token: {e}")
    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Main function to run the examples."""
    # Get auth token from environment or login
    auth_token = os.environ.get("LLAMA_STACK_AUTH_TOKEN")
    
    if not auth_token:
        # In a real application, you would implement a proper login flow
        # This is just a placeholder
        print("No auth token found. Please set LLAMA_STACK_AUTH_TOKEN environment variable.")
        return
    
    # Create client
    client = LlamaStackClient(base_url="http://localhost:8080")
    
    # Run examples
    await vector_db_example(client, auth_token)
    await agent_with_restricted_tools_example(client, auth_token)


if __name__ == "__main__":
    asyncio.run(main())

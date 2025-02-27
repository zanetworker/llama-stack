import os

def create_http_client():
    from llama_stack_client import LlamaStackClient
    return LlamaStackClient(base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}")

def create_library_client(template="ollama"):
    from llama_stack import LlamaStackAsLibraryClient
    client = LlamaStackAsLibraryClient(template)
    client.initialize()
    return client

import os
from termcolor import cprint

from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types import Document
from llama_stack.apis.inference import (
    UserMessage,
)

client = create_http_client()  # or create_http_client() depending on the environment you picked

# Register a memory bank
bank_id = "my_documents"
response = client.memory_banks.register(
    memory_bank_id=bank_id,
    params={
        "memory_bank_type": "vector",
        "embedding_model": "all-MiniLM-L6-v2",
        "chunk_size_in_tokens": 512
    }
)

# Insert documents
documents = [
    {
        "document_id": "doc1",
        "content": "Your document text here",
        "mime_type": "text/plain"
    }
]
client.memory.insert(bank_id, documents)

# Query documents
results = client.memory.query(
    bank_id=bank_id,
    query="What do you know about...",
)
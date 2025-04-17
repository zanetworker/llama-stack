import os
import asyncio
from termcolor import cprint

def create_http_client():
    from llama_stack_client import LlamaStackClient
    return LlamaStackClient(base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}")

from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types import Document
from llama_stack.apis.inference import UserMessage

# Initialize client
client = create_http_client()

# Documents to be used for RAG
urls = ["chat.rst", "llama3.rst", "datasets.rst", "lora_finetune.rst"]
documents = [
    Document(
        document_id=f"num-{i}",
        content=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}",
        mime_type="text/plain",
        metadata={},
    )
    for i, url in enumerate(urls)
]

print(documents)

# Register a vector database with the updated API
vector_db_id = "test-vector-db"
client.vector_dbs.register(
    vector_db_id=vector_db_id,
    embedding_model="all-MiniLM-L6-v2",
    embedding_dimension=384,
    provider_id="faiss"  # Added this required parameter
)

# Use the RAG tool for document ingestion
client.tool_runtime.rag_tool.insert(
    documents=documents,
    vector_db_id=vector_db_id,
    chunk_size_in_tokens=512,
)

agent_config = AgentConfig(
    model=os.environ["INFERENCE_MODEL"],
    instructions="You are a helpful assistant specialized in analyzing PyTorch documentation.",
    enable_session_persistence=True,
    toolgroups=[
        {
            "name": "builtin::rag",
            "args": {
                "vector_db_ids": [vector_db_id],
            }
        }
    ],
    tool_choice="auto"
)

rag_agent = Agent(client, agent_config)
session_id = rag_agent.create_session("test-session")

user_prompts = [
    "What are the top 5 topics that were explained? Only list succinct bullet points.",
]

# Run the agent loop with the updated API
for prompt in user_prompts:
    cprint(f'User> {prompt}', 'green')
    response = rag_agent.create_turn(
        messages=[UserMessage(role="user", content=prompt)],
        session_id=str(session_id),
        documents=documents,  # Keep this as is since it's used for reference
    )
    for log in EventLogger().log(response):
        log.print()
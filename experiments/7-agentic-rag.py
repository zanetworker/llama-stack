from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from termcolor import cprint
from llama_stack_client.types import Document
from llama_stack_client import LlamaStackClient
import httpx
import os

# Initialize client with timeout settings
client = LlamaStackClient(
    base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}",
    # timeout=httpx.Timeout(
    #     connect=5.0,    # connection timeout
    #     read=30.0,      # read timeout
    #     write=5.0,      # write timeout
    #     pool=10.0       # pool timeout
    # ),
    # limits=httpx.Limits(
    #     max_connections=5,
    #     max_keepalive_connections=5,
    #     keepalive_expiry=5.0
    # )
)
model_id = os.environ['INFERENCE_MODEL']

# Reduce number of documents and use smaller documents
urls = ["chat.rst", "llama3.rst"]  # Only use 2 documents for testing

# Create smaller documents with limited content
documents = [
    Document(
        document_id=f"num-{i}",
        content=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}",
        mime_type="text/plain",
        metadata={},
    )
    for i, url in enumerate(urls)
]

vector_db_id = "test-vector-db"

# Clean up existing vector db if it exists
try:
    client.vector_dbs.delete(vector_db_id=vector_db_id)
except:
    pass

# Register vector db
client.vector_dbs.register(
    vector_db_id=vector_db_id,
    embedding_model="all-MiniLM-L6-v2",
    embedding_dimension=384,
)

# Insert documents with smaller chunk size
client.tool_runtime.rag_tool.insert(
    documents=documents,
    vector_db_id=vector_db_id,
    chunk_size_in_tokens=512,  #
)

# Ensure the agent is configured to use the RAG tool effectively
agent_config = AgentConfig(
    model=model_id,
    instructions="""You are a concise assistant. Use the retrieved context to answer queries. 
                   Focus on providing brief and relevant information based on the context.""",
    enable_session_persistence=False,
    max_infer_iters=2,
    toolgroups=[
        {
            "name": "builtin::rag",
            "args": {
                "vector_db_ids": [vector_db_id],
                "top_k": 2,
                "min_score": 0.7
            }
        }
    ],
)

# Create agent and session
rag_agent = Agent(client, agent_config)
session_id = rag_agent.create_session("test-session")

# Use a more focused prompt
user_prompts = [
    "Based on the retrieved context, list the 2 most important topics mentioned. Be very brief.",
]

for prompt in user_prompts:
    cprint(f'User> {prompt}', 'green')
    try:
        response = rag_agent.create_turn(
            messages=[{"role": "user", "content": prompt}],
            session_id=session_id,
        )
        
        # Handle streaming response with timeout protection
        for log in EventLogger().log(response):
            try:
                log.print()
            except Exception as e:
                cprint(f"Error printing log: {str(e)}", "red")
                continue
                
    except httpx.ReadTimeout:
        cprint("Response timed out. Try reducing the context or simplifying the query.", "red")
    except Exception as e:
        cprint(f"Error: {str(e)}", "red")
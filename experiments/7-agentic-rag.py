from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from termcolor import cprint
from llama_stack_client.types import Document
from llama_stack_client import LlamaStackClient
import httpx
import os
import asyncio

async def fetch_document_content(url: str) -> str:
    """Fetch document content from URL"""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code == 200:
            return response.text
    return ""

async def main():
    # Initialize client with timeout settings
    client = LlamaStackClient(
        base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}",
        # timeout=httpx.Timeout(
        #     connect=5.0,
        #     read=60.0,  # Increased read timeout
        #     write=5.0,
        #     pool=10.0
        # )
    )
    model_id = os.environ['INFERENCE_MODEL']

    # URLs for documents
    base_url = "https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/"
    urls = ["chat.rst", "llama3.rst"]

    # Fetch actual content from URLs
    documents = []
    for i, url in enumerate(urls):
        full_url = f"{base_url}{url}"
        content = await fetch_document_content(full_url)
        if content:
            doc = Document(
                document_id=f"num-{i}",
                content=content,  # Use actual content instead of URL
                mime_type="text/plain",
                metadata={"url": full_url},
            )
            documents.append(doc)
            cprint(f"Loaded document {i} with {len(content)} characters", "green")

    vector_db_id = "test-vector-db"

    # Clean up existing vector db
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

    # Insert documents with optimized chunk size
    client.tool_runtime.rag_tool.insert(
        documents=documents,
        vector_db_id=vector_db_id,
        chunk_size_in_tokens=256,  # Increased chunk size for better context
    )

    # Configure agent with context window settings
    agent_config = AgentConfig(
        model=model_id,
        instructions="""You are a concise assistant. Your task is to:
                       1. Use ONLY the retrieved context to answer queries
                       2. Provide brief, focused responses
                       3. If you can't find relevant information in the context, say so
                       4. Do not make up or infer information not present in the context""",
        enable_session_persistence=False,
        max_infer_iters=2,
        toolgroups=[
            {
                "name": "builtin::rag",
                "args": {
                    "vector_db_ids": [vector_db_id],
                    "query_config": {
                        "max_tokens_in_context": 1800,  # Stay within Ollama's limits
                        "max_chunks": 3,
                        "query_generator_config": {
                            "type": "default",
                            "separator": "\n"
                        }
                    }
                }
            }
        ],
    )

    # Create agent and session
    rag_agent = Agent(client, agent_config)
    session_id = rag_agent.create_session("test-session")

    # Use a more focused prompt
    prompt = "Based on the retrieved context, what are the 2 most important topics mentioned? List them as bullet points."
    
    cprint(f'\nUser> {prompt}', 'green')
    try:
        response = rag_agent.create_turn(
            messages=[{
                "role": "user",
                "content": prompt
            }],
            session_id=session_id,
            documents=documents  # Include documents in the turn
        )
        
        # Handle streaming response
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

if __name__ == "__main__":
    asyncio.run(main())
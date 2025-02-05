import os
from termcolor import cprint
import httpx
import asyncio
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import Document
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from typing import List, Dict, Any

class RAGImplementation:
    def __init__(self, base_url: str, port: str):
        self.client = LlamaStackClient(base_url=f"http://localhost:{port}")
        self.vector_db_id = "test-vector-db"
        self.urls = ["chat.rst", "llama3.rst", "datasets.rst", "lora_finetune.rst"]
        self.base_url = base_url

    async def fetch_documents(self) -> List[Document]:
        """Fetch documents from URLs asynchronously"""
        documents = []
        async with httpx.AsyncClient() as http_client:
            for i, url in enumerate(self.urls):
                full_url = f"{self.base_url}{url}"
                print(f"Fetching {full_url}")
                try:
                    response = await http_client.get(full_url)
                    if response.status_code == 200:
                        doc = Document(
                            document_id=f"num-{i}",
                            content=response.text,
                            mime_type="text/plain",
                            metadata={"url": full_url},
                        )
                        documents.append(doc)
                        cprint(f"Document {i} loaded, length: {len(response.text)}", "green")
                    else:
                        cprint(f"Failed to fetch {full_url}: Status {response.status_code}", "red")
                except Exception as e:
                    cprint(f"Error fetching {full_url}: {str(e)}", "red")
        return documents

    def setup_vector_db(self):
        """Initialize vector database"""
        try:
            self.client.vector_dbs.delete(vector_db_id=self.vector_db_id)
        except:
            pass

        cprint("\nRegistering vector database...", "cyan")
        self.client.vector_dbs.register(
            vector_db_id=self.vector_db_id,
            embedding_model="all-MiniLM-L6-v2",
            embedding_dimension=384,
            provider_id="faiss",
        )

    def method2_rag_tool(self, documents: List[Document]):
        """Method 2: RAG Tool Implementation with improved error handling"""
        cprint("\n=== Method 2: RAG Tool ===", "yellow")

        try:
            # First, ensure documents are properly chunked before insertion
            self.client.tool_runtime.rag_tool.insert(
                documents=documents,
                vector_db_id=self.vector_db_id,
                chunk_size_in_tokens=512,  # Reduced chunk size for better stability
                overlap_size_in_tokens=50   # Add overlap for context continuity
            )

            # Query with more specific configuration
            query_config = {
                "max_tokens_in_context": 2048,
                "max_chunks": 3,
                "similarity_threshold": 0.7,
                "query_generator_config": {
                    "type": "default",
                    "separator": "\n"
                }
            }

            results = self.client.tool_runtime.rag_tool.query(
                content=[{
                    "type": "text",
                    "text": "What are the key learning objectives across these documents?"
                }],
                vector_db_ids=[self.vector_db_id],
                query_config=query_config
            )

            cprint("\nRAG Tool Query Results:", "green")
            if hasattr(results, 'content'):
                for item in results.content:
                    if hasattr(item, 'text'):
                        print(f"\n{item.text}")

        except httpx.HTTPError as e:
            cprint(f"HTTP Error in rag_tool method: {str(e)}", "red")
            if hasattr(e, 'response'):
                cprint(f"Response content: {e.response.content}", "red")
        except Exception as e:
            cprint(f"Error in rag_tool method: {str(e)}", "red")

    def method3_agent_rag(self, documents: List[Document]):
        """Method 3: Agent-based RAG Implementation with fixed configuration"""
        cprint("\n=== Method 3: Agent-based RAG ===", "yellow")

        try:
            model = os.environ.get('INFERENCE_MODEL', "Llama3.2-3B-Instruct")
            
            # Create agent configuration dictionary instead of AgentConfig object
            agent_config = {
                "model": model,
                "instructions": """You are a concise assistant. Your task is to:
                              1. Use ONLY the retrieved context to answer queries
                              2. Provide brief, focused responses
                              3. If you can't find relevant information in the context, say so
                              4. Do not make up or infer information not present in the context""",
                "enable_session_persistence": False,
                "max_infer_iters": 2,
                "toolgroups": [
                    {
                        "name": "builtin::rag",
                        "args": {
                            "vector_db_ids": [self.vector_db_id],
                            "top_k": 3,
                            "query_config": {
                                "max_tokens_in_context": 2048,
                                "max_chunks": 3,
                                "query_generator_config": {
                                    "type": "default",
                                    "separator": "\n"
                                }
                            }
                        }
                    }
                ]
            }

            # Create agent and session
            agent = Agent(self.client, agent_config)
            session_id = agent.create_session("rag_session")

            prompt = "Based on the retrieved context, what are the 6 most important topics mentioned? List them as bullet points."
            
            cprint(f'\nUser> {prompt}', 'green')
            try:
                response = agent.create_turn(
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    session_id=session_id,
                    documents=documents
                )
                
                for log in EventLogger().log(response):
                    try:
                        log.print()
                    except Exception as e:
                        cprint(f"Error printing log: {str(e)}", "red")
                        continue
                    
            except httpx.ReadTimeout:
                cprint("Response timed out. Try reducing the context or simplifying the query.", "red")
            except Exception as e:
                cprint(f"Error in response handling: {str(e)}", "red")

        except Exception as e:
            cprint(f"Error in agent_rag method setup: {str(e)}", "red")

async def main():
    # Initialize RAG implementation
    rag = RAGImplementation(
        base_url="https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/",
        port=os.environ['LLAMA_STACK_PORT']
    )

    # Fetch documents
    documents = await rag.fetch_documents()
    cprint(f"\nLoaded {len(documents)} documents", "cyan")

    # Setup vector database
    rag.setup_vector_db()

    # Run the fixed methods
    rag.method2_rag_tool(documents)
    rag.method3_agent_rag(documents)

if __name__ == "__main__":
    asyncio.run(main())
from dataclasses import dataclass
from typing import List, Dict, Optional, AsyncGenerator
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types import Document
import httpx
import os
import asyncio
from termcolor import cprint


@dataclass
class ComponentConfig:
    """Configuration for a pipeline component"""
    name: str
    depends_on: List[str]
    config: Dict


class RAGPipeline:
    def __init__(
        self,
        base_url: Optional[str] = None,
        port: Optional[str] = None,
        model_id: Optional[str] = None
    ):
        """Initialize RAG Pipeline with components"""
        self.base_url = base_url or f"http://localhost:{port or os.environ['LLAMA_STACK_PORT']}"
        self.model_id = model_id or os.environ['INFERENCE_MODEL']
        self.client = LlamaStackClient(base_url=self.base_url)
        
        # Set up components and their dependencies
        self.setup_components()
        
    def setup_components(self):
        """Define pipeline components and their relationships"""
        self.components = {
            "document_loader": ComponentConfig(
                name="document_loader",
                depends_on=[],
                config={
                    "timeout": httpx.Timeout(connect=5.0, read=60.0, write=5.0, pool=10.0)
                }
            ),
            "vector_store": ComponentConfig(
                name="vector_store",
                depends_on=["document_loader"],
                config={
                    "vector_db_id": "test-vector-db",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "embedding_dimension": 384,
                    "chunk_size": 512
                }
            ),
            "rag_agent": ComponentConfig(
                name="rag_agent",
                depends_on=["vector_store"],
                config={
                    "instructions": """You are a concise assistant. Your task is to:
                                     1. Use ONLY the retrieved context to answer queries
                                     2. Provide brief, focused responses
                                     3. If you can't find relevant information in the context, say so
                                     4. Do not make up or infer information not present in the context""",
                    "enable_session_persistence": False,
                    "max_infer_iters": 2,
                    "context_window": 4096,
                    "max_chunks": 4
                }
            )
        }
        
    async def fetch_document_content(self, url: str) -> str:
        """Fetch document content from URL using document_loader component"""
        async with httpx.AsyncClient(timeout=self.components["document_loader"].config["timeout"]) as client:
            response = await client.get(url)
            if response.status_code == 200:
                return response.text
        return ""

    async def load_documents(self, urls: List[str]) -> List[Document]:
        """Load documents using document_loader component"""
        documents = []
        for i, url in enumerate(urls):
            content = await self.fetch_document_content(url)
            if content:
                doc = Document(
                    document_id=f"num-{i}",
                    content=content,
                    mime_type="text/plain",
                    metadata={"url": url}
                )
                documents.append(doc)
                cprint(f"Loaded document {i} with {len(content)} characters", "green")
        return documents

    def setup_vector_store(self, vector_db_id: str):
        """Set up vector store component"""
        # Clean up existing vector db
        try:
            self.client.vector_dbs.delete(vector_db_id=vector_db_id)
        except:
            pass

        # Register vector db
        self.client.vector_dbs.register(
            vector_db_id=vector_db_id,
            embedding_model=self.components["vector_store"].config["embedding_model"],
            embedding_dimension=self.components["vector_store"].config["embedding_dimension"]
        )

    def index_documents(self, documents: List[Document], vector_db_id: str):
        """Index documents using vector store component"""
        self.client.tool_runtime.rag_tool.insert(
            documents=documents,
            vector_db_id=vector_db_id,
            chunk_size_in_tokens=self.components["vector_store"].config["chunk_size"]
        )

    def create_agent(self, vector_db_id: str) -> Agent:
        """Create RAG agent component"""
        agent_config = AgentConfig(
            model=self.model_id,
            instructions=self.components["rag_agent"].config["instructions"],
            enable_session_persistence=self.components["rag_agent"].config["enable_session_persistence"],
            max_infer_iters=self.components["rag_agent"].config["max_infer_iters"],
            toolgroups=[
                {
                    "name": "builtin::rag",
                    "args": {
                        "vector_db_ids": [vector_db_id],
                        "top_k": 4,
                        "query_config": {
                            "max_tokens_in_context": self.components["rag_agent"].config["context_window"],
                            "max_chunks": self.components["rag_agent"].config["max_chunks"],
                            "query_generator_config": {
                                "type": "default",
                                "separator": "\n"
                            }
                        }
                    }
                }
            ]
        )
        return Agent(self.client, agent_config)

    async def process_response(self, response) -> AsyncGenerator:
        """Process streaming response from RAG agent"""
        for log in EventLogger().log(response):
            try:
                yield log
            except Exception as e:
                cprint(f"Error processing log: {str(e)}", "red")


async def main():
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # URLs for documents
    base_url = "https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/"
    urls = [f"{base_url}{url}" for url in ["chat.rst", "llama3.rst"]]

    try:
        # Load documents
        documents = await pipeline.load_documents(urls)
        
        # Set up and index documents
        vector_db_id = pipeline.components["vector_store"].config["vector_db_id"]
        pipeline.setup_vector_store(vector_db_id)
        pipeline.index_documents(documents, vector_db_id)

        # Create agent and session
        rag_agent = pipeline.create_agent(vector_db_id)
        session_id = rag_agent.create_session("test-session")

        # Query the pipeline
        prompt = "Based on the retrieved context, what are the 2 most important topics mentioned? List them as bullet points."
        cprint(f'\nUser> {prompt}', 'green')
        
        response = rag_agent.create_turn(
            messages=[{"role": "user", "content": prompt}],
            session_id=session_id,
            documents=documents
        )

        async for log in pipeline.process_response(response):
            log.print()

    except httpx.ReadTimeout:
        cprint("Response timed out. Try reducing the context or simplifying the query.", "red")
    except Exception as e:
        cprint(f"Error: {str(e)}", "red")

if __name__ == "__main__":
    asyncio.run(main())
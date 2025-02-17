import os
import asyncio
from termcolor import cprint
from typing import Optional, List
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import Document
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from termcolor import colored


class LlamaStackDemo:
    def __init__(self, port: Optional[str] = None):
        """Initialize LlamaStack client with configuration"""
        self.base_url = f"http://localhost:{port or os.environ.get('LLAMA_STACK_PORT', '8080')}"
        self.client = LlamaStackClient(base_url=self.base_url)
        self.model_id = os.environ.get('INFERENCE_MODEL', 'llama2')
        self.vector_db_id = "demo-vector-db"

    def setup_vector_store(self):
        """Initialize vector database"""
        try:
            # Clean up existing vector db
            try:
                self.client.vector_dbs.delete(vector_db_id=self.vector_db_id)
            except:
                pass

            # Register new vector db
            self.client.vector_dbs.register(
                vector_db_id=self.vector_db_id,
                embedding_model="all-MiniLM-L6-v2",
                embedding_dimension=384,
                provider_id="faiss"
            )
            cprint(f"Vector store setup complete: {self.vector_db_id}", "green")
        except Exception as e:
            cprint(f"Error setting up vector store: {e}", "red")
            raise

    def ingest_documents(self, documents: List[Document]):
        """Ingest documents into vector store"""
        try:
            self.client.tool_runtime.rag_tool.insert(
                documents=documents,
                vector_db_id=self.vector_db_id,
                chunk_size_in_tokens=512,
            )
            cprint(f"Ingested {len(documents)} documents", "green")
        except Exception as e:
            cprint(f"Error ingesting documents: {e}", "red")
            raise

    def setup_rag_agent(self) -> tuple[Agent, str]:
        """Configure and create RAG agent"""
        try:
            agent_config = AgentConfig(
                model=self.model_id,
                instructions="""You are a helpful assistant that can:
                1. Use RAG to access relevant information
                2. Provide clear, accurate responses
                3. Acknowledge when information is not available""",
                enable_session_persistence=False,
                toolgroups=[
                    {
                        "name": "builtin::rag",
                        "args": {
                            "vector_db_ids": [self.vector_db_id],
                            "query_config": {
                                "max_tokens_in_context": 2048,
                                "max_chunks": 3
                            }
                        }
                    }
                ]
            )

            agent = Agent(self.client, agent_config)
            session_id = agent.create_session("demo-session")
            return agent, session_id

        except Exception as e:
            cprint(f"Error setting up RAG agent: {e}", "red")
            raise

    def list_available_models(self):
        """List available models"""
        try:
            models = self.client.models.list()
            cprint("\nAvailable Models:", "blue")
            for model in models:
                if model.model_type == "llm":
                    cprint(f"- {model.identifier}", "green")
        except Exception as e:
            cprint(f"Error listing models: {e}", "red")
            raise

    def run_chat_completion(self, prompt: str):
        """Run basic chat completion"""
        try:
            response = self.client.inference.chat_completion(
                model_id=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            cprint("\nChat Completion Response:", "blue")
            cprint(response.completion_message.content, "green")
        except Exception as e:
            cprint(f"Error in chat completion: {e}", "red")
            raise

    def run_rag_query(self, agent: Agent, session_id: str, query: str):
        """Run RAG query through agent"""
        try:
            response = agent.create_turn(
                messages=[{"role": "user", "content": query}],
                session_id=session_id
            )
            
            cprint("\nRAG Query Response:", "blue")
            for log in EventLogger().log(response):
                log.print()
        except Exception as e:
            cprint(f"Error in RAG query: {e}", "red")
            raise

async def main():
    # Initialize demo
    demo = LlamaStackDemo()
    
    # List available models
    demo.list_available_models()
    
    # Run basic chat completion
    demo.run_chat_completion("Write a haiku about coding")
    
    # Setup RAG components
    demo.setup_vector_store()
    
    # Create sample documents
    documents = [
        Document(
            document_id="doc1",
            content="LlamaStack is a framework for building AI applications.",
            mime_type="text/plain"
        ),
        Document(
            document_id="doc2",
            content="RAG (Retrieval Augmented Generation) enhances LLM responses with external knowledge.",
            mime_type="text/plain"
        )
    ]
    
    # Ingest documents
    demo.ingest_documents(documents)
    
    # Setup and run RAG agent
    agent, session_id = demo.setup_rag_agent()
    demo.run_rag_query(agent, session_id, "What is LlamaStack and RAG?")

if __name__ == "__main__":
    asyncio.run(main())
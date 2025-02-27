import os
from llama_stack_client.types import Document
from dataclasses import dataclass
from typing import List, Optional
from llama_stack_client.lib.agents.event_logger import EventLogger

@dataclass
class LlamaStackSettings:
    base_url: str
    vector_db_id: str
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    chunk_size: int = 512
    model: str = os.getenv("INFERENCE_MODEL", "llama2")
    instructions: str = "You are a helpful assistant"

class LlamaStackPipeline:
    def __init__(self, settings: LlamaStackSettings):
        self.settings = settings
        self.client = self._init_client()
        
    def _init_client(self):
        from llama_stack_client import LlamaStackClient
        return LlamaStackClient(base_url=self.settings.base_url)
            
    def init_vector_store(self):
        """Initialize and register vector database"""
        self.client.vector_dbs.register(
            vector_db_id=self.settings.vector_db_id,
            embedding_model=self.settings.embedding_model,
            embedding_dimension=self.settings.embedding_dimension,
        )
        
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        self.client.tool_runtime.rag_tool.insert(
            documents=documents,
            vector_db_id=self.settings.vector_db_id,
            chunk_size_in_tokens=self.settings.chunk_size,
        )
        
    def init_agent(self, custom_instructions: Optional[str] = None):
        """Initialize the RAG agent"""
        from llama_stack_client.lib.agents.agent import Agent
        from llama_stack_client.types.agent_create_params import AgentConfig
        
        agent_config = AgentConfig(
            model=self.settings.model,
            instructions=custom_instructions or self.settings.instructions,
            enable_session_persistence=False,
            toolgroups=[
                {
                    "name": "builtin::rag",
                    "args": {
                        "vector_db_ids": [self.settings.vector_db_id],
                    }
                }
            ],
        )
        self.agent = Agent(self.client, agent_config)
        self._session_id = self.agent.create_session("pipeline-session")
        
    def query(self, prompt: str):
        """Run a query through the RAG pipeline"""
        if not hasattr(self, 'agent') or not hasattr(self, '_session_id'):
            raise ValueError("Agent not initialized. Call init_agent() first.")
            
        response = self.agent.create_turn(
            messages=[{"role": "user", "content": prompt}],
            session_id=self._session_id,
        )
        
        # Log and return the response
        for log in EventLogger().log(response):
            log.print()
        return response

# Usage example
if __name__ == "__main__":
    # Initialize settings
    settings = LlamaStackSettings(
        base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}",
        vector_db_id="test-vector-db"
    )

    # Create sample documents
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

    # Initialize pipeline
    pipeline = LlamaStackPipeline(settings)
    
    # Set up vector store and add documents
    pipeline.init_vector_store()
    pipeline.add_documents(documents)
    
    # Initialize agent
    pipeline.init_agent("You are a helpful assistant specialized in analyzing PyTorch documentation.")
    
    # Run a query
    query = "What are the top 5 topics that were explained? Only list succinct bullet points."
    response_generator = pipeline.query(query)
    
    # Process the streaming response
    assistant_response = []
    for response_chunk in response_generator:
        for event in EventLogger().log(response_chunk):
            # Look for message completion events
            if event.event_type == "TURN_COMPLETE":
                assistant_response.append(event.payload.turn.output_message.content)
    
    print("\nFinal Answer:")
    print("".join(assistant_response))